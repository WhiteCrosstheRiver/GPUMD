/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
 * This file contains the definitions of the Adam optimizer.
 * Use the Adam optimizer to fit potential parameters.
 * Reference:
 * Adam: A method for stochastic optimization
 * doi: 10.48550/arXiv.1412.6980
------------------------------------------------------------------------------*/
 
#include "adam.cuh"
#include "parameters.cuh"
#include "utilities/error.cuh"
#include <chrono>

static __global__ void initialize_curand_states(curandState* state, int N, int seed)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    curand_init(seed, n, 0, &state[n]);
  }
}

static __global__ void update_moments(
  const int N,
  const float beta1,
  const float beta2,
  const float* __restrict__ gradients,
  float* __restrict__ m,
  float* __restrict__ v)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    const float g = gradients[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
  }
}

static __global__ void apply_updates(
  const int N,
  const float lr,
  const float beta1_t,
  const float beta2_t,
  const float eps,
  const float weight_decay,
  const int* mask,
  const float* __restrict__ m,
  const float* __restrict__ v,
  float* __restrict__ parameters)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N && (mask == nullptr || mask[i])) {
    // calculate bias correction
    const float m_hat = m[i] / (1.0f - beta1_t);
    const float v_hat = v[i] / (1.0f - beta2_t);
    
    // update parameters
    parameters[i] -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * parameters[i]);
  }
}

__global__ void adam_update(
    const int N,
    const float step_size,  // = lr * sqrt(1 - beta2^t)/(1 - beta1^t)
    const float beta1_power,// = beta1^t
    const float beta2_power,// = beta2^t
    const float eps,
    float* __restrict__ m,
    float* __restrict__ v,
    float* __restrict__ param)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float bc1 = 1.0f - beta1_power;
        float bc2 = 1.0f - beta2_power;

        float m_hat = m[i] / bc1;   
        float v_hat = v[i] / bc2;   

        // param[i] -= step_size * (m_hat / (sqrt(v_hat) + eps));
        param[i] = param[i] - step_size * (m_hat / (sqrt(v_hat) + eps));
    }
}

__global__ void compute_gradient_norm(const float* grad, int n, float* norm_result)
{
  __shared__ float shared_sum[1024];
  const int tid = threadIdx.x;
  shared_sum[tid] = 0.0f;
  
  for (int i = tid; i < n; i += blockDim.x) {
    shared_sum[tid] += grad[i] * grad[i];
  }
  __syncthreads();
  
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    __syncthreads();
  }
  
  if (tid == 0) {
    *norm_result = shared_sum[0];
  }
}

__global__ void apply_gradient_clipping(float* grad, int n, float norm, float max_norm)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (norm > max_norm) {
    float scale = max_norm / norm;
    if (tid < n) {
      grad[tid] *= scale;
    }
  }
}

void clip_gradients(int step, float* d_grad, int size) 
{
  static float avg_norm = -1.0f; 

  float* d_norm;
  CHECK(cudaMalloc(&d_norm, sizeof(float)));
  CHECK(cudaMemset(d_norm, 0, sizeof(float)));
  
  // calculate the norm of the gradient
  int block_size = 1024;
  compute_gradient_norm<<<1, block_size>>>(d_grad, size, d_norm);
  
  // apply gradient clipping
  int grid_size = (size + block_size - 1) / block_size;
  float h_norm;
  CHECK(cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost));
  h_norm = sqrt(h_norm);
  
  if (avg_norm < 0) {
    avg_norm = h_norm;
  } else {
    avg_norm = 0.9f * avg_norm + 0.1f * h_norm;
  }
  
  const float alpha = 1.0f;
  const float MAX_CLIP_NORM = 10.0f; //modified by parameters?
  float max_norm = avg_norm * alpha;
  if (max_norm > MAX_CLIP_NORM) max_norm = MAX_CLIP_NORM;
  
  apply_gradient_clipping<<<grid_size, block_size>>>(d_grad, size, h_norm, max_norm);
  
  CHECK(cudaFree(d_norm));
}

Adam::Adam(Parameters& para)
{
  // initialize the parameters
  number_of_variables = para.number_of_variables;
  weight_decay = para.weight_decay;
  lambda1 = para.lambda_1;
  lambda2 = para.lambda_2;
  loss_epsilon = para.fcvp_epsilon;
  project_output = para.fcvp_interval > 0;

  // initialize the CPU vectors
  parameters.resize(number_of_variables);
  m.resize(number_of_variables, 0.0f);
  v.resize(number_of_variables, 0.0f);

  // initialize the GPU vectors
  cudaSetDevice(0);
  gpu_parameters.resize(number_of_variables);
  if (project_output) {
    std::vector<int> mask(number_of_variables, 1);
    const int block = (para.dim + 2) * para.num_neurons1 + 1;
    for (int t = 0; t < para.num_types; ++t) {
      for (int j = 0; j <= para.num_neurons1; ++j) {
        mask[t * block + (para.dim + 1) * para.num_neurons1 + j] = 0;
      }
    }
    update_mask.resize(number_of_variables);
    update_mask.copy_from_host(mask.data());
  }
  gpu_m.resize(number_of_variables, 0.0f);
  gpu_v.resize(number_of_variables, 0.0f);
  curand_states.resize(number_of_variables);
  initialize_curand_states<<<(number_of_variables - 1) / 128 + 1, 128>>>(
    curand_states.data(), number_of_variables, para.seed);
  GPU_CHECK_KERNEL
}

static __global__ void gpu_create_paramters(
  const int dim,
  const int num_types,
  const int num_neurons1,
  const int number_of_variables_ann,
  const int number_of_variables,
  const float* energy_shift,
  curandState* g_state,
  float* g_parameters)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n < number_of_variables_ann) {
    curandState state = g_state[n];
    int type_idx = (n * num_types) / number_of_variables_ann;
    int param_idx = n % ((dim + 2) * num_neurons1 + 1);
    if (param_idx < dim * num_neurons1) {
      float std = sqrt(2.0f / (dim + num_neurons1));
      g_parameters[n] = curand_normal(&state) * std;
    } else if (param_idx < (dim + 1) * num_neurons1) {
      g_parameters[n] = 0.01f * curand_normal(&state);
    } else if (param_idx < (dim + 2) * num_neurons1) {
      float std = sqrt(2.0f / (num_neurons1 + 1));
      g_parameters[n] = curand_normal(&state) * std;
    } else {
      float mean = energy_shift[type_idx];
      g_parameters[n] = mean + 0.01f * curand_normal(&state);
    }
  } else if (n < number_of_variables) {
    curandState state = g_state[n];
    float r1 = curand_normal(&state);
    float r2 = curand_uniform(&state) - 0.5f;
    g_parameters[n] = 0.1f * r1 + r2;
  }
}

void Adam::initialize_parameters(Parameters& para)
{
  FILE* fid_restart = fopen("gnep.restart", "r");
  if (fid_restart == NULL) {
    cudaSetDevice(0); // normally use GPU-0
    gpu_create_paramters<<<(number_of_variables - 1) / 128 + 1, 128>>>(
      para.dim,
      para.num_types,
      para.num_neurons1,
      para.number_of_variables_ann,
      para.number_of_variables,
      para.energy_shift_gpu.data(),
      curand_states.data(),
      gpu_parameters.data());
    GPU_CHECK_KERNEL
    gpu_parameters.copy_to_host(parameters.data());
  } else {
    for (int n = 0; n < number_of_variables; ++n) {
        int count = fscanf(fid_restart, "%f", &parameters[n]);
        PRINT_SCANF_ERROR(count, 1, "Reading error for gnep.restart.");
    }
    fclose(fid_restart);
    cudaSetDevice(0); // normally use GPU-0
    gpu_parameters.copy_from_host(parameters.data());
  }
}

static __global__ void prepare_gradients(
  int n, const int* mask, const float* parameters, float l1, float l2, float* gradient)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = parameters[i];
    gradient[i] =
      (mask == nullptr || mask[i]) ? gradient[i] + l1 * ((x > 0) - (x < 0)) + l2 * x : 0.0f;
  }
}

void Adam::sync_parameters()
{
  CHECK(cudaSetDevice(0));
  gpu_parameters.copy_from_host(parameters.data());
}

void Adam::update(float lr, float* gradients) {
  CHECK(cudaSetDevice(0));
  const int block_size = 256;
  const int grid_size = (number_of_variables + block_size - 1) / block_size + 1;

  const int* mask = project_output ? update_mask.data() : nullptr;
  if (project_output || lambda1 > 0 || lambda2 > 0) {
    double norm2 = 0;
    if (lambda2 > 0) {
      for (float x : parameters) {
        norm2 += double(x) * x;
      }
    }
    const double norm = std::sqrt(norm2 / number_of_variables + loss_epsilon * loss_epsilon);
    prepare_gradients<<<grid_size, block_size>>>(
      number_of_variables,
      mask,
      gpu_parameters.data(),
      lambda1 / number_of_variables,
      lambda2 / (number_of_variables * norm),
      gradients);
    GPU_CHECK_KERNEL
  }
  clip_gradients(step, gradients, number_of_variables);
  update_moments<<<grid_size, block_size>>>(
    number_of_variables,
    beta1,
    beta2,
    gradients,
    gpu_m.data(),
    gpu_v.data()
  );
  GPU_CHECK_KERNEL
  
  // calculate bias correction
  const float beta1_t = pow(beta1, step + 1);
  const float beta2_t = pow(beta2, step + 1);

  // apply parameter updates
  apply_updates<<<grid_size, block_size>>>(
    number_of_variables,
    lr,
    beta1_t,
    beta2_t,
    eps,
    weight_decay,
    mask,
    gpu_m.data(),
    gpu_v.data(),
    gpu_parameters.data());
  GPU_CHECK_KERNEL

  gpu_parameters.copy_to_host(parameters.data());
  step++; 
}

void Adam::output_parameters(Parameters& para) {
  cudaSetDevice(0); 
  gpu_parameters.copy_to_host(parameters.data());
  FILE* fid = fopen("gnep.restart", "w");
  for (int i = 0; i < number_of_variables; ++i) {
    fprintf(fid, "%15.7e\n", parameters[i]);
  }
  fclose(fid);
}

float* Adam::get_parameters() { return parameters.data(); }
