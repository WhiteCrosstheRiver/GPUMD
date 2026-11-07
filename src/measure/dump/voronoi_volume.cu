/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "force/neighbor.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include "voronoi_volume.cuh"
#include <cmath>

// Counts every neighbor but stores at most MN, so the caller can grow the list and retry.
static __global__ void gpu_find_voronoi_neighbors(
  const Box box,
  const int N,
  const int MN,
  const int* __restrict__ cell_counts,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  int* NN,
  int* NL,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const double cutoff_square)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }
  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];
  int cell_id, cell_id_x, cell_id_y, cell_id_z;
  find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);
  int z_begin, z_end, y_begin, y_end, x_begin, x_end;
  get_neighbor_cell_offset_range(box.pbc_z, nz, z_begin, z_end);
  get_neighbor_cell_offset_range(box.pbc_y, ny, y_begin, y_end);
  get_neighbor_cell_offset_range(box.pbc_x, nx, x_begin, x_end);

  int count = 0;
  for (int k = z_begin; k < z_end; ++k) {
    const int cell_z = shift_cell_id(cell_id_z, k, nz, box.pbc_z);
    if (cell_z < 0) {
      continue;
    }
    for (int j = y_begin; j < y_end; ++j) {
      const int cell_y = shift_cell_id(cell_id_y, j, ny, box.pbc_y);
      if (cell_y < 0) {
        continue;
      }
      for (int i = x_begin; i < x_end; ++i) {
        const int cell_x = shift_cell_id(cell_id_x, i, nx, box.pbc_x);
        if (cell_x < 0) {
          continue;
        }
        const int neighbor_cell = cell_x + nx * cell_y + nx * ny * cell_z;
        const int num_atoms = cell_counts[neighbor_cell];
        const int first = cell_count_sum[neighbor_cell];
        for (int m = 0; m < num_atoms; ++m) {
          const int n2 = cell_contents[first + m];
          if (n2 == n1) {
            continue;
          }
          double x12 = x[n2] - x1;
          double y12 = y[n2] - y1;
          double z12 = z[n2] - z1;
          apply_mic(box, x12, y12, z12);
          if (x12 * x12 + y12 * y12 + z12 * z12 < cutoff_square) {
            if (count < MN) {
              NL[static_cast<size_t>(count) * N + n1] = n2;
            }
            ++count;
          }
        }
      }
    }
  }
  NN[n1] = count;
}

// One warp per atom. Each lane holds Q directions, so M = 32 * Q. Along direction n the cell
// reaches 1 / max(1/R, max_j 2 (r_j - r_i).n / |r_j - r_i|^2).
template <int Q>
static __global__ void gpu_voronoi_volume(
  const int N,
  const Box box,
  const double R,
  const double* __restrict__ directions,
  const int* __restrict__ NN,
  const int* __restrict__ NL,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  double* __restrict__ volume)
{
  constexpr unsigned FULL = 0xffffffffu;
  constexpr int M = 32 * Q;
  const int lane = threadIdx.x & 31;
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (i >= N) {
    return;
  }

  double q[Q], nx[Q], ny[Q], nz[Q];
#pragma unroll
  for (int a = 0; a < Q; ++a) {
    const int m = lane + 32 * a;
    nx[a] = directions[m];
    ny[a] = directions[m + M];
    nz[a] = directions[m + 2 * M];
    q[a] = 1.0 / R;
  }

  const int count = NN[i];
  const double cutoff_square = 4.0 * R * R;
  for (int k = 0; k < count; ++k) {
    const int j = NL[i + static_cast<size_t>(k) * N];
    double dx = x[j] - x[i];
    double dy = y[j] - y[i];
    double dz = z[j] - z[i];
    apply_mic(box, dx, dy, dz);
    const double d2 = dx * dx + dy * dy + dz * dz;
    if (d2 <= 0.0 || d2 >= cutoff_square) {
      continue;
    }
    const double scale = 2.0 / d2;
#pragma unroll
    for (int a = 0; a < Q; ++a) {
      q[a] = fmax(q[a], scale * (dx * nx[a] + dy * ny[a] + dz * nz[a]));
    }
  }

  double sum = 0.0;
#pragma unroll
  for (int a = 0; a < Q; ++a) {
    const double r = 1.0 / q[a];
    sum += r * r * r;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(FULL, sum, offset);
  }
  if (lane == 0) {
    volume[i] = (4.0 * PI / (3.0 * M)) * sum;
  }
}

void VoronoiVolume::set(const std::string& radius_token, const std::string& direction_token)
{
  double radius = 0.0;
  int directions = 0;
  if (!is_valid_real(radius_token, &radius) || !(radius > 0.0)) {
    PRINT_INPUT_ERROR("Voronoi radius R should be a positive number.\n");
  }
  if (!is_valid_int(direction_token, &directions) || (directions != 128 && directions != 256)) {
    PRINT_INPUT_ERROR("number of Voronoi directions should be 128 or 256.\n");
  }
  if (enabled() && (radius != radius_ || directions != directions_)) {
    PRINT_INPUT_ERROR(
      "dump_xyz volume, stress, stress_norm and pressure must use the same R and number of "
      "directions.\n");
  }
  radius_ = radius;
  directions_ = directions;
  // Fibonacci sphere: nearly uniform directions with equal weights.
  std::vector<double> cpu(3 * directions);
  const double golden = PI * (3.0 - sqrt(5.0));
  for (int m = 0; m < directions; ++m) {
    const double zm = 1.0 - 2.0 * (m + 0.5) / directions;
    const double xy = sqrt(fmax(0.0, 1.0 - zm * zm));
    cpu[m] = xy * cos(m * golden);
    cpu[m + directions] = xy * sin(m * golden);
    cpu[m + 2 * directions] = zm;
  }
  directions_gpu_.resize(3 * directions);
  directions_gpu_.copy_from_host(cpu.data());
}

void VoronoiVolume::compute(Box& box, Atom& atom, std::vector<double>& cpu_volume)
{
  const int N = atom.number_of_atoms;
  const double rc = 2.0 * radius_;
  const double rc_cell_list = 0.5 * rc;
  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);
  const double min_thickness = 4.0 * radius_;
  if (
    (box.pbc_x && box.thickness_x < min_thickness) ||
    (box.pbc_y && box.thickness_y < min_thickness) ||
    (box.pbc_z && box.thickness_z < min_thickness)) {
    PRINT_INPUT_ERROR(
      "A periodic box thickness is smaller than 4R, so the minimum image cannot cover the 2R "
      "neighborhood of the Voronoi volume. Increase the box or decrease R.\n");
  }

  if (static_cast<int>(NN_.size()) != N) {
    NN_.resize(N);
    volume_.resize(N);
    cell_count_.resize(N);
    cell_count_sum_.resize(N);
    cell_contents_.resize(N);
  }
  find_cell_list(
    rc_cell_list, num_bins, box, atom.position_per_atom, cell_count_, cell_count_sum_,
    cell_contents_);

  const double* x = atom.position_per_atom.data();
  const double* y = x + N;
  const double* z = x + 2 * N;
  std::vector<int> cpu_NN(N);
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (NL_.size() < static_cast<size_t>(N) * max_neighbors_) {
      NL_.resize(static_cast<size_t>(N) * max_neighbors_);
    }
    gpu_find_voronoi_neighbors<<<(N - 1) / 256 + 1, 256>>>(
      box, N, max_neighbors_, cell_count_.data(), cell_count_sum_.data(), cell_contents_.data(),
      NN_.data(), NL_.data(), x, y, z, num_bins[0], num_bins[1], num_bins[2], 1.0 / rc_cell_list,
      rc * rc);
    GPU_CHECK_KERNEL
    NN_.copy_to_host(cpu_NN.data());
    int largest = 0;
    for (const int n : cpu_NN) {
      largest = n > largest ? n : largest;
    }
    if (largest <= max_neighbors_) {
      break;
    }
    if (attempt == 1 || largest > 1024) {
      PRINT_INPUT_ERROR(
        "Voronoi neighbor list holds more than 1024 atoms. Decrease R or use a less dense "
        "structure.\n");
    }
    max_neighbors_ = largest;
  }

  constexpr int block = 128;
  const int grid = (N - 1) / (block / 32) + 1;
  if (directions_ == 128) {
    gpu_voronoi_volume<4><<<grid, block>>>(
      N, box, radius_, directions_gpu_.data(), NN_.data(), NL_.data(), x, y, z, volume_.data());
  } else {
    gpu_voronoi_volume<8><<<grid, block>>>(
      N, box, radius_, directions_gpu_.data(), NN_.data(), NL_.data(), x, y, z, volume_.data());
  }
  GPU_CHECK_KERNEL
  cpu_volume.resize(N);
  volume_.copy_to_host(cpu_volume.data());
}
