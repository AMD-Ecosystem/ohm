// Copyright (c) 2017
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// ABN 41 687 119 230
//
// Author: Kazys Stepanas
// HIP port by Claude (Anthropic)
#ifndef HIP_IMPORTCL_H
#define HIP_IMPORTCL_H

#include <hip/hip_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdio>

#include "hutil_decl.h"

#define LOCAL_ARG(TYPE, VAR)
#define LOCAL_MEM_ENABLE()       \
  size_t shared_mem_offset_ = 0; \
  extern __shared__ char shared_mem_[]
#define LOCAL_VAR(TYPE, VAR, SIZE)                   \
  TYPE VAR = (TYPE)&shared_mem_[shared_mem_offset_]; \
  shared_mem_offset_ += (SIZE)
#define LM_PER_THREAD(per_thread_size) ((per_thread_size)*blockDim.x * blockDim.y * blockDim.z)

// Kernel
#define __kernel __global__

// Memory
// For HIP compatibility, we use __local on arguments, (removed in HIP) and local for local declarations (convert to
// __shared__)
#define __constant __constant__
#define __global
#define __local
#define local __shared__

// HIP devices are little endian
#define __ENDIAN_LITTLE__ 1

// Useful information:
// https://www.sharcnet.ca/help/index.php/Porting_CUDA_to_OpenCL

/*

Hardware Terminology
CUDA/HIP                      OpenCL
SM (Stream Multiprocessor)    CU (Compute Unit)
Thread                        Work-item
Block                         Work-group
Global memory                 Global memory
Constant memory               Constant memory
Shared memory                 Local memory
Local memory                  Private memory


Qualifiers for Kernel Functions
CUDA/HIP                            OpenCL
__constant__ variable declaration   __constant variable declaration
__device__ function                 No annotation necessary
__device__ variable declaration     __global variable declaration
__global__ function                 __kernel function
__shared__ variable declaration     local variable declaration
argument declaration declaration    __local variable declaration


Kernels Indexing
CUDA/HIP                          OpenCL
gridDim                           get_num_groups()
blockDim                          get_local_size()
blockIdx                          get_group_id()
threadIdx                         get_local_id()
blockIdx * blockDim + threadIdx   get_global_id()
gridDim * blockDim                get_global_size()
CUDA/HIP is using threadIdx.x to get the id for the first dimension while OpenCL is using get_local_id(0).


Kernels Synchronization
CUDA/HIP                OpenCL
__syncthreads()         barrier(CLK_LOCAL_MEM_FENCE)
__threadfence()         No direct equivalent
__threadfence_block()   mem_fence()
No direct equivalent    read_mem_fence()
No direct equivalent    write_mem_fence()

*/

// Synchronisation.
#define barrier(...) __syncthreads()
#define mem_fence    __threadfence_block

typedef unsigned char uchar;
typedef unsigned int uint;
typedef long long longlong;
typedef unsigned long long ulonglong;

inline __device__ unsigned get_num_groups(int i)
{
  switch (i)
  {
  default:
  case 0:
    return gridDim.x;
  case 1:
    return gridDim.y;
  case 2:
    return gridDim.z;
  }
}

inline __device__ unsigned get_local_size(int i)
{
  switch (i)
  {
  default:
  case 0:
    return blockDim.x;
  case 1:
    return blockDim.y;
  case 2:
    return blockDim.z;
  }
}

inline __device__ unsigned get_group_id(int i)
{
  switch (i)
  {
  default:
  case 0:
    return blockIdx.x;
  case 1:
    return blockIdx.y;
  case 2:
    return blockIdx.z;
  }
}

inline __device__ unsigned get_local_id(int i)
{
  switch (i)
  {
  default:
  case 0:
    return threadIdx.x;
  case 1:
    return threadIdx.y;
  case 2:
    return threadIdx.z;
  }
}

inline __device__ unsigned get_global_size(int i)
{
  switch (i)
  {
  default:
  case 0:
    return gridDim.x * blockDim.x;
  case 1:
    return gridDim.y * blockDim.y;
  case 2:
    return gridDim.z * blockDim.z;
  }
}

inline __device__ unsigned get_global_id(int i)
{
  switch (i)
  {
  default:
  case 0:
    return blockIdx.x * blockDim.x + threadIdx.x;
  case 1:
    return blockIdx.y * blockDim.y + threadIdx.y;
  case 2:
    return blockIdx.z * blockDim.z + threadIdx.z;
  }
}

inline __device__ float3 xyz(float4 v)
{
  return make_float3(v.x, v.y, v.z);
}

// // Atomic operation mapping : OpenCL to HIP
// #define atomic_add atomicAdd
// #define atomic_sub atomicSub
// #define atomic_xchg atomicExch
// #define atomic_min atomicMin
// #define atomic_max atomicMax
// #define atomic_inc(addr) atomicAdd(addr, 1)
// #define atomic_dec(addr) atomicSub(addr, 1)
// #define atomic_cmpxchg atomicCAS
// #define atomic_and atomicAnd
// #define atomic_or atomicOr
// #define atomic_xor atomicXor

// HIP provides math constants via hip_runtime.h
// M_PI is typically available

#include "hutil_atomic.h"
#include "hutil_math.h"

#endif  // HIP_IMPORTCL_H
