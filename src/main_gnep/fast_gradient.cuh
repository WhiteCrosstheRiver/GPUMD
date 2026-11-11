/*----------------------------------------------------------------------------80
Fast loss_rmse gradient for gnep (included at the end of gnep.cu).

The force and virial terms of the loss gradient are parameter derivatives of a
directional derivative of the energy. With per-atom force weights v_i and per-
configuration virial weights D (the same prefactors as accumulate_gradients),

  sum_i v_i . dF_i/dtheta + sum_c D_c : dW_c/dtheta = d/dtheta dE[delta],
  delta_12 = (v_1 - v_2) - (D r_12),   dE[delta] = sum_pairs dU_1/dr_12 . delta_12,

because F_1 += f_12, F_2 -= f_12 and W_ab = -sum r_a f_b with f_12 = dU_1/dr_12.
dE[delta] is a central difference of E along the pair displacement delta, so the
whole gradient needs only energy-parameter gradients at r and r +- h delta:

  g = sum_i omega_i dU_i/dtheta (r) + [dE/dtheta(r + h delta) - dE/dtheta(r - h delta)] / 2h.

An energy-parameter gradient is one descriptor pass, one ANN pass whose weight
gradients are a DGEMM (Q^T B), and one pair pass for the descriptor coefficients.
Nothing per-atom of size num_ann x dim is stored. Precondition: find_force with
require_grad = false has just run on every shard with the same parameters. The
cached descriptors, Fp and sum_fxyz are overwritten.

GNEP_FD_H (default 1e-2) is the largest pair displacement h max|delta| in Angstrom.
------------------------------------------------------------------------------*/

#include <cublas_v2.h>
#include <cstring>

struct GNEP_FastGradientScratch {
  GPU_Vector<float> delta[6]; // radial x, y, z; angular x, y, z
  GPU_Vector<float> moved[6];
  GPU_Vector<float> v;     // 3N force weights
  GPU_Vector<float> omega; // N energy weights
  GPU_Vector<double> Q;    // N x (dim + 1), last column 1
  GPU_Vector<double> B;    // N x (2H + 1) num_types
  GPU_Vector<double> G;    // (dim + 1) x (2H + 1) num_types
  GPU_Vector<double> grad; // num_para
  GPU_Vector<unsigned int> max_bits;
  cublasHandle_t handle = nullptr;
  ~GNEP_FastGradientScratch()
  {
    if (handle) {
      cublasDestroy(handle);
    }
  }
};

namespace
{

template <typename T>
void fg_reserve(GPU_Vector<T>& vector, size_t size)
{
  if (vector.size() < size) {
    vector.resize(size);
  }
}

static __global__ void fg_weights(
  const int N,
  const int global_num_configurations,
  const int global_num_atoms,
  const float lambda_e,
  const float lambda_f,
  const float force_delta,
  const int* __restrict__ g_batch_idx,
  const int* __restrict__ g_Na,
  const int* __restrict__ g_type,
  const float* __restrict__ g_weight,
  const float* __restrict__ g_type_weight,
  const float* __restrict__ g_diff_e,
  const float* __restrict__ g_force,
  const float* __restrict__ g_force_ref,
  float* g_omega,
  float* g_v)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }
  const int c = g_batch_idx[n1];
  const float squared_weight = g_weight[c] * g_weight[c];
  const int Na = g_Na[c];
  g_omega[n1] = g_diff_e[c] * squared_weight * 2.0f * lambda_e / global_num_configurations / Na;
  const float fr[3] = {g_force_ref[n1], g_force_ref[n1 + N], g_force_ref[n1 + 2 * N]};
  float type_weight = g_type_weight[g_type[n1]];
  if (force_delta > 0.0f) {
    const float magnitude = sqrt(fr[0] * fr[0] + fr[1] * fr[1] + fr[2] * fr[2]);
    type_weight *= sqrt(force_delta / (force_delta + magnitude));
  }
  type_weight *= type_weight;
  const float force_factor = squared_weight * 2.0f * lambda_f / 3 / global_num_atoms;
  for (int d = 0; d < 3; ++d) {
    g_v[n1 + d * N] = force_factor * type_weight * (g_force[n1 + d * N] - fr[d]);
  }
}

static __global__ void fg_pair_delta(
  const int N,
  const int global_virial_components,
  const float lambda_v,
  const int* __restrict__ g_NN,
  const int* __restrict__ g_NL,
  const int* __restrict__ g_batch_idx,
  const int* __restrict__ g_Na,
  const int* __restrict__ g_has_virial,
  const float* __restrict__ g_weight,
  const float* __restrict__ g_diff_v,
  const float* __restrict__ g_v,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_dx,
  float* g_dy,
  float* g_dz,
  unsigned int* g_max_bits)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }
  const int c = g_batch_idx[n1];
  const float squared_weight = g_weight[c] * g_weight[c];
  const float per_Nc_v = (g_has_virial[c] && global_virial_components > 0)
                           ? squared_weight * 2.0f * lambda_v / global_virial_components / g_Na[c]
                           : 0.0f;
  float D[6];
  for (int d = 0; d < 6; ++d) {
    D[d] = g_diff_v[c * 6 + d] * per_Nc_v; // xx yy zz xy yz zx
  }
  const float v1[3] = {g_v[n1], g_v[n1 + N], g_v[n1 + 2 * N]};
  float largest = 0.0f;
  for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
    const int index = i1 * N + n1;
    const int n2 = g_NL[index];
    const float r[3] = {g_x12[index], g_y12[index], g_z12[index]};
    // W_xy = -sum r_x f_y etc., so component b of D r collects D_ab r_a.
    const float dx = v1[0] - g_v[n2] - (D[0] * r[0] + D[5] * r[2]);
    const float dy = v1[1] - g_v[n2 + N] - (D[1] * r[1] + D[3] * r[0]);
    const float dz = v1[2] - g_v[n2 + 2 * N] - (D[2] * r[2] + D[4] * r[1]);
    g_dx[index] = dx;
    g_dy[index] = dy;
    g_dz[index] = dz;
    largest = fmaxf(largest, fmaxf(fabsf(dx), fmaxf(fabsf(dy), fabsf(dz))));
  }
  atomicMax(g_max_bits, __float_as_uint(largest)); // non-negative floats order as integers
}

static __global__ void
fg_move(const int size, const float step, const float* x, const float* delta, float* moved)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < size) {
    moved[n] = x[n] + step * delta[n];
  }
}

// Energy-parameter gradient of the ANN for weights omega (per atom, or one scalar):
// Q = [q_scaled, 1], B = omega [w1 tanh', tanh, 1] per type, so Q^T B holds
// d/dw0 (rows d), and with the ones row d/db0 (negated), d/dw1 and d/db1 (negated).
// Fp_out = omega dU/dq (scaled descriptors), the input of the coefficient pass.
static __global__ void fg_ann(
  const int N,
  const int num_types,
  const GNEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  const float* __restrict__ g_omega,
  const float omega_scalar,
  float* g_Fp,
  double* g_Q,
  double* g_B)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }
  const int D = annmb.dim;
  const int H = annmb.num_neurons1;
  const int C = 2 * H + 1;
  const int t = g_type[n1];
  const float omega = g_omega ? g_omega[n1] : omega_scalar;
  float q[MAX_DIM];
  float Fp[MAX_DIM];
  for (int d = 0; d < D; ++d) {
    q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    Fp[d] = 0.0f;
    g_Q[n1 + static_cast<size_t>(d) * N] = q[d];
  }
  g_Q[n1 + static_cast<size_t>(D) * N] = 1.0;
  for (int other = 0; other < num_types; ++other) {
    if (other != t) {
      for (int col = 0; col < C; ++col) {
        g_B[n1 + static_cast<size_t>(other * C + col) * N] = 0.0;
      }
    }
  }
  const float* w0 = annmb.w0[t];
  const float* b0 = annmb.b0[t];
  const float* w1 = annmb.w1[t];
  for (int h = 0; h < H; ++h) {
    float z = 0.0f;
    for (int d = 0; d < D; ++d) {
      z += w0[h * D + d] * q[d];
    }
    const float a = tanh(z - b0[h]);
    const float coefficient = w1[h] * (1.0f - a * a);
    for (int d = 0; d < D; ++d) {
      Fp[d] += coefficient * w0[h * D + d];
    }
    g_B[n1 + static_cast<size_t>(t * C + h) * N] = double(omega) * coefficient;
    g_B[n1 + static_cast<size_t>(t * C + H + h) * N] = double(omega) * a;
  }
  g_B[n1 + static_cast<size_t>(t * C + 2 * H) * N] = omega;
  for (int d = 0; d < D; ++d) {
    g_Fp[n1 + d * N] = omega * Fp[d] * g_q_scaler[d];
  }
}

static __global__ void fg_scatter_ann(
  const int D, const int H, const int num_types, const double* __restrict__ g_G, double* g_grad)
{
  const int per_type = (D + 2) * H + 1;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= per_type * num_types) {
    return;
  }
  const int t = index / per_type;
  const int p = index % per_type;
  const int C = 2 * H + 1;
  const int ld = D + 1;
  const double* G = g_G + static_cast<size_t>(t * C) * ld;
  double value;
  if (p < H * D) {
    value = G[(p % D) + (p / D) * ld]; // w0[h * D + d]
  } else if (p < H * D + H) {
    value = -G[D + (p - H * D) * ld]; // b0
  } else if (p < H * D + 2 * H) {
    value = G[D + (p - H * D) * ld]; // w1: column H + h
  } else {
    value = -G[D + 2 * H * ld]; // b1
  }
  g_grad[index] += value;
}

static __device__ __forceinline__ void
fg_add(double* s_acc, double* g_acc, const bool use_shared, const int index, const float value)
{
  if (use_shared) {
    atomicAdd(&s_acc[index], double(value));
  } else {
    atomicAdd(&g_acc[index], double(value));
  }
}

// Descriptor-coefficient part of sum_i dU_i/dc with g_Fp = omega dU/dq (scaled).
static __global__ void fg_coefficients(
  const int N,
  const GNEP::ParaMB paramb,
  const int* __restrict__ g_type,
  const int* __restrict__ g_NN_radial,
  const int* __restrict__ g_NL_radial,
  const float* __restrict__ g_x12_radial,
  const float* __restrict__ g_y12_radial,
  const float* __restrict__ g_z12_radial,
  const int* __restrict__ g_NN_angular,
  const int* __restrict__ g_NL_angular,
  const float* __restrict__ g_x12_angular,
  const float* __restrict__ g_y12_angular,
  const float* __restrict__ g_z12_angular,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  const int num_c,
  const bool use_shared,
  double* g_grad_c)
{
  extern __shared__ double s_acc[];
  if (use_shared) {
    for (int k = threadIdx.x; k < num_c; k += blockDim.x) {
      s_acc[k] = 0.0;
    }
    __syncthreads();
  }
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    const int t1 = g_type[n1];
    const int T = paramb.num_types;
    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      const int index = i1 * N + n1;
      const int t2 = g_type[g_NL_radial[index]];
      const float x12 = g_x12_radial[index];
      const float y12 = g_y12_radial[index];
      const float z12 = g_z12_radial[index];
      const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      const float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      const float rcinv = 1.0f / rc;
      float fc12;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        const float fp = g_Fp[n1 + n * N];
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          const int c_index =
            (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq + t1 * T + t2;
          fg_add(s_acc, g_grad_c, use_shared, c_index, fp * fn12[k]);
        }
      }
    }
    float Fp[MAX_DIM_ANGULAR];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      const int index = i1 * N + n1;
      const int t2 = g_type[g_NL_angular[index]];
      float r12[3] = {g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
      const float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      const float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      const float rcinv = 1.0f / rc;
      float fc12;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          float value = 0.0f;
          for (int l = 0; l < paramb.num_L; ++l) {
            float qc = 0.0f;
            gnep_higher_body::qc(paramb, N, l, n, d12, r12, fn12[k], &g_sum_fxyz[n1], &qc);
            value += Fp[l * (paramb.n_max_angular + 1) + n] * qc;
          }
          const int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq +
                              t1 * T + t2 + paramb.num_c_radial;
          fg_add(s_acc, g_grad_c, use_shared, c_index, value);
        }
      }
    }
  }
  if (use_shared) {
    __syncthreads();
    for (int k = threadIdx.x; k < num_c; k += blockDim.x) {
      if (s_acc[k] != 0.0) {
        atomicAdd(&g_grad_c[k], s_acc[k]);
      }
    }
  }
}

} // namespace

void GNEP::fast_rmse_gradient(
  Parameters& para,
  std::vector<Dataset>& dataset,
  int global_num_configurations,
  int global_num_atoms,
  int global_virial_components,
  std::vector<double>& gradient)
{
  if (zbl.flexibled) {
    PRINT_INPUT_ERROR("The fast gradient does not support flexible ZBL parameters.");
  }
  const Batch_MSE mse = get_batch_mse(para, dataset, true);
  const double eps2 = para.fcvp_epsilon * para.fcvp_epsilon;
  const float lambda_e = para.lambda_e / (2.0 * std::sqrt(mse.energy + eps2));
  const float lambda_f = para.lambda_f / (2.0 * std::sqrt(mse.force + eps2));
  const float lambda_v =
    mse.virial_configurations ? para.lambda_v / (2.0 * std::sqrt(mse.virial + eps2)) : 0.0f;
  const char* target_text = std::getenv("GNEP_FD_H");
  const double target = target_text ? std::atof(target_text) : 1e-2;

  const int num_para = para.number_of_variables;
  gradient.assign(num_para, 0.0);
  std::vector<double> local(num_para);
  for (int device_id = 0; device_id < static_cast<int>(dataset.size()); ++device_id) {
    CHECK(cudaSetDevice(device_id));
    DeviceContext& context = *contexts[device_id];
    Dataset& data = dataset[device_id];
    GNEP_Data& cache = context.data;
    ANN& ann = context.ann;
    if (!context.fast) {
      context.fast = std::make_shared<GNEP_FastGradientScratch>();
      if (cublasCreate(&context.fast->handle) != CUBLAS_STATUS_SUCCESS) {
        PRINT_INPUT_ERROR("cublasCreate failed for the fast gradient.");
      }
    }
    GNEP_FastGradientScratch& s = *context.fast;
    const int N = data.N;
    const int D = ann.dim;
    const int H = ann.num_neurons1;
    const int T = paramb.num_types;
    const int C = 2 * H + 1;
    const size_t radial_size = static_cast<size_t>(N) * data.max_NN_radial;
    const size_t angular_size = static_cast<size_t>(N) * data.max_NN_angular;
    for (int a = 0; a < 3; ++a) {
      fg_reserve(s.delta[a], std::max<size_t>(1, radial_size));
      fg_reserve(s.moved[a], std::max<size_t>(1, radial_size));
      fg_reserve(s.delta[3 + a], std::max<size_t>(1, angular_size));
      fg_reserve(s.moved[3 + a], std::max<size_t>(1, angular_size));
    }
    fg_reserve(s.v, 3 * static_cast<size_t>(N));
    fg_reserve(s.omega, static_cast<size_t>(N));
    fg_reserve(s.Q, static_cast<size_t>(N) * (D + 1));
    fg_reserve(s.B, static_cast<size_t>(N) * C * T);
    fg_reserve(s.G, static_cast<size_t>(D + 1) * C * T);
    fg_reserve(s.grad, static_cast<size_t>(num_para));
    fg_reserve(s.max_bits, 1);
    CHECK(cudaMemset(s.grad.data(), 0, sizeof(double) * num_para));
    CHECK(cudaMemset(s.max_bits.data(), 0, sizeof(unsigned int)));

    const int block = 64;
    const int grid = (N - 1) / block + 1;
    fg_weights<<<grid, block>>>(
      N,
      global_num_configurations,
      global_num_atoms,
      lambda_e,
      lambda_f,
      para.force_delta,
      data.batch_idx.data(),
      data.Na.data(),
      data.type.data(),
      data.weight_gpu.data(),
      data.type_weight_gpu.data(),
      data.diff_gpu_e.data(),
      data.force.data(),
      data.force_ref_gpu.data(),
      s.omega.data(),
      s.v.data());
    GPU_CHECK_KERNEL
    const GPU_Vector<int>* NN[2] = {&cache.NN_radial, &cache.NN_angular};
    const GPU_Vector<int>* NL[2] = {&cache.NL_radial, &cache.NL_angular};
    GPU_Vector<float>* xyz[6] = {
      &cache.x12_radial,
      &cache.y12_radial,
      &cache.z12_radial,
      &cache.x12_angular,
      &cache.y12_angular,
      &cache.z12_angular};
    for (int list = 0; list < 2; ++list) {
      fg_pair_delta<<<grid, block>>>(
        N,
        global_virial_components,
        lambda_v,
        NN[list]->data(),
        NL[list]->data(),
        data.batch_idx.data(),
        data.Na.data(),
        data.has_virial_gpu.data(),
        data.weight_gpu.data(),
        data.diff_gpu_v.data(),
        s.v.data(),
        xyz[3 * list]->data(),
        xyz[3 * list + 1]->data(),
        xyz[3 * list + 2]->data(),
        s.delta[3 * list].data(),
        s.delta[3 * list + 1].data(),
        s.delta[3 * list + 2].data(),
        s.max_bits.data());
      GPU_CHECK_KERNEL
    }
    unsigned int bits = 0;
    s.max_bits.copy_to_host(&bits, 1);
    float largest = 0.0f;
    std::memcpy(&largest, &bits, sizeof(float));
    const double h = largest > 0.0f ? target / largest : 0.0;

    const int num_c = num_para - ann.num_ann;
    const size_t shared_bytes = sizeof(double) * num_c;
    const bool use_shared = shared_bytes <= 40000;
    double* grad_c = s.grad.data() + ann.num_ann;
    auto energy_gradient = [&](const float* omega, float omega_scalar, bool moved, bool first) {
      fg_ann<<<grid, block>>>(
        N,
        T,
        ann,
        data.type.data(),
        cache.descriptors.data(),
        para.q_scaler_gpu(device_id).data(),
        omega,
        omega_scalar,
        cache.Fp.data(),
        s.Q.data(),
        s.B.data());
      GPU_CHECK_KERNEL
      const double alpha = 1.0;
      const double beta = first ? 0.0 : 1.0;
      if (
        cublasDgemm(
          s.handle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          D + 1,
          C * T,
          N,
          &alpha,
          s.Q.data(),
          N,
          s.B.data(),
          N,
          &beta,
          s.G.data(),
          D + 1) != CUBLAS_STATUS_SUCCESS) {
        PRINT_INPUT_ERROR("cublasDgemm failed in the fast gradient.");
      }
      const float* p[6];
      for (int a = 0; a < 6; ++a) {
        p[a] = moved ? s.moved[a].data() : xyz[a]->data();
      }
      fg_coefficients<<<grid, block, use_shared ? shared_bytes : 0>>>(
        N,
        paramb,
        data.type.data(),
        cache.NN_radial.data(),
        cache.NL_radial.data(),
        p[0],
        p[1],
        p[2],
        cache.NN_angular.data(),
        cache.NL_angular.data(),
        p[3],
        p[4],
        p[5],
        cache.Fp.data(),
        cache.sum_fxyz.data(),
        num_c,
        use_shared,
        grad_c);
      GPU_CHECK_KERNEL
    };

    // Energy term at r, from the descriptors of the preceding find_force.
    energy_gradient(s.omega.data(), 0.0f, false, true);
    // Force and virial terms: central difference of the energy gradient along delta.
    if (h > 0.0) {
      for (const int sign : {1, -1}) {
        for (int a = 0; a < 6; ++a) {
          const size_t size = a < 3 ? radial_size : angular_size;
          if (size > 0) {
            fg_move<<<(size - 1) / 256 + 1, 256>>>(
              static_cast<int>(size), float(sign * h), xyz[a]->data(), s.delta[a].data(),
              s.moved[a].data());
            GPU_CHECK_KERNEL
          }
        }
        find_descriptors_radial<<<(N - 1) / 32 + 1, 32>>>(
          N,
          data.max_NN_radial,
          cache.NN_radial.data(),
          cache.NL_radial.data(),
          paramb,
          ann,
          data.type.data(),
          s.moved[0].data(),
          s.moved[1].data(),
          s.moved[2].data(),
          cache.descriptors.data());
        GPU_CHECK_KERNEL
        find_descriptors_angular<<<(N - 1) / 32 + 1, 32>>>(
          N,
          cache.NN_angular.data(),
          cache.NL_angular.data(),
          paramb,
          ann,
          data.type.data(),
          s.moved[3].data(),
          s.moved[4].data(),
          s.moved[5].data(),
          cache.descriptors.data(),
          cache.sum_fxyz.data());
        GPU_CHECK_KERNEL
        energy_gradient(nullptr, float(sign / (2.0 * h)), true, false);
      }
    }
    fg_scatter_ann<<<(ann.num_ann - 1) / 256 + 1, 256>>>(D, H, T, s.G.data(), s.grad.data());
    GPU_CHECK_KERNEL
    s.grad.copy_to_host(local.data(), num_para);
    for (int i = 0; i < num_para; ++i) {
      gradient[i] += local[i];
    }
  }
  CHECK(cudaSetDevice(0));
}
