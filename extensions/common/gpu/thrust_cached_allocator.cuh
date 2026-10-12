/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#pragma once

// Caching allocator for Thrust temporary storage.
//
// With the default thrust::device policy every algorithm that needs scratch
// space (copy_if, exclusive_scan, sort, ...) calls cudaMalloc and cudaFree,
// and cudaFree synchronizes the device. On the per-step skip path that is two
// extra device-wide syncs per MD step. Passing this allocator through
//   thrust::copy_if(gpumd_ext::thrust_policy(alloc), ...)
// keeps the blocks and hands them back on the next call (pattern of the Thrust
// example custom_temporary_allocation.cu). Results are unchanged; only the
// allocation pattern differs. Single-threaded use per instance.

#include "utilities/gpu_macro.cuh"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <thrust/execution_policy.h>

namespace gpumd_ext
{

class ThrustCachedAllocator
{
public:
  typedef char value_type;

  ThrustCachedAllocator() = default;
  ThrustCachedAllocator(const ThrustCachedAllocator&) = delete;
  ThrustCachedAllocator& operator=(const ThrustCachedAllocator&) = delete;
  ~ThrustCachedAllocator() { release(); }

  char* allocate(std::ptrdiff_t num_bytes)
  {
    const size_t n = num_bytes > 0 ? (size_t)num_bytes : 1;
    char* p = nullptr;
    auto it = free_blocks_.lower_bound(n); // smallest cached block that fits
    if (it != free_blocks_.end()) {
      p = it->second;
      const size_t size = it->first;
      free_blocks_.erase(it);
      allocated_.emplace(p, size);
      return p;
    }
    if (gpuMalloc((void**)&p, n) != gpuSuccess) {
      // out of memory with blocks cached: give them back once and retry
      release_free_blocks();
      if (gpuMalloc((void**)&p, n) != gpuSuccess) {
        printf("ThrustCachedAllocator: gpuMalloc of %zu bytes failed.\n", n);
        exit(1);
      }
    }
    allocated_.emplace(p, n);
    return p;
  }

  void deallocate(char* p, size_t)
  {
    auto it = allocated_.find(p);
    if (it == allocated_.end()) {
      return;
    }
    free_blocks_.emplace(it->second, p);
    allocated_.erase(it);
  }

  void release()
  {
    release_free_blocks();
    for (auto& kv : allocated_) {
      gpuFree(kv.first);
    }
    allocated_.clear();
  }

private:
  void release_free_blocks()
  {
    for (auto& kv : free_blocks_) {
      gpuFree(kv.second);
    }
    free_blocks_.clear();
  }

  std::multimap<size_t, char*> free_blocks_;
  std::map<char*, size_t> allocated_;
};

inline auto thrust_policy(ThrustCachedAllocator& alloc)
{
#ifdef USE_HIP
  return thrust::hip::par(alloc);
#else
  return thrust::cuda::par(alloc);
#endif
}

} // namespace gpumd_ext
