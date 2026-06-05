// Copyright (c) 2019
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// ABN 41 687 119 230
//
// Author: Kazys Stepanas
// HIP port by Claude (Anthropic)
#ifndef HUTIL_DECL_H
#define HUTIL_DECL_H

// Whilst the functions below are __host__, the declaration of std::function apparent is __host__ __device__.
// This can yield a situation where a __host__ function is calling a __host__ __device__ function which is not
// allowed. We force the issue by excluding this code when building device code.
#ifndef __HIP_DEVICE_COMPILE__

#include <functional>

// Helper macros for exposing HIP kernel functions.

namespace gputil
{
using SharedMemCalculation = std::function<size_t(size_t)>;
using OptimalGroupSizeCalculation = std::function<int(size_t *, const SharedMemCalculation &)>;
}  // namespace gputil

#define _GPUTIL_DECLARE_KERNEL(kernel_name) const void *kernel_name##Ptr()
#define _GPUTIL_DECLARE_GROUP_CALC(kernel_name) \
  gputil::OptimalGroupSizeCalculation kernel_name##OptimalGroupSizeCalculator()

/// Declaration of the a function which exposes a HIP kernel for use with gputil.
/// Returns a pointer to the HIP kernel.
#define GPUTIL_CUDA_DECLARE_KERNEL(kernel_name) \
  _GPUTIL_DECLARE_KERNEL(kernel_name);          \
  _GPUTIL_DECLARE_GROUP_CALC(kernel_name)


/// Definition of the a function which exposes a HIP kernel for use with gputil.
#define GPUTIL_CUDA_DEFINE_KERNEL(kernel_name)                                                                 \
  __host__ _GPUTIL_DECLARE_KERNEL(kernel_name) { return (const void *)&(kernel_name); }                        \
  __host__ _GPUTIL_DECLARE_GROUP_CALC(kernel_name)                                                             \
  {                                                                                                            \
    return [](size_t *optimal_size, const gputil::SharedMemCalculation &shared_mem_calc) {                     \
      int min_grid_size = 0, optimal_block_size = 0;                                                           \
      hipError_t err = hipSuccess;                                                                             \
      err = ::hipOccupancyMaxPotentialBlockSizeVariableSMem(&min_grid_size, &optimal_block_size, kernel_name,  \
                                                             shared_mem_calc);                                 \
      *optimal_size = size_t(optimal_block_size);                                                              \
      return err;                                                                                              \
    };                                                                                                         \
  }

#else  // !__HIP_DEVICE_COMPILE__

#define GPUTIL_CUDA_DEFINE_KERNEL(kernel_name)

#endif  // !__HIP_DEVICE_COMPILE__

#endif  // HUTIL_DECL_H
