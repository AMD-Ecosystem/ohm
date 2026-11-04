# Build the CUDA backend with HIP for AMD GPUs. The CUDA sources, targets and options are reused: OHM_FEATURE_CUDA is
# set so the CUDA targets are configured, the .cu files compile as HIP, and gputil/cuda/hip_compat supplies the CUDA
# headers they include.
if(CMAKE_VERSION VERSION_LESS 3.21)
  message(FATAL_ERROR "OHM_FEATURE_HIP requires CMake 3.21 or later.")
endif(CMAKE_VERSION VERSION_LESS 3.21)

set(OHM_FEATURE_CUDA ON)
set(OHM_USE_DEPRECATED_CMAKE_CUDA OFF)

# Compile the .cu files in the given list as HIP. Other files in the list are left alone.
function(ohm_hip_sources)
  foreach(_source ${ARGN})
    if(_source MATCHES "\\.cu$")
      set_source_files_properties(${_source} PROPERTIES LANGUAGE HIP)
    endif()
  endforeach(_source)
endfunction(ohm_hip_sources)

macro(cuda_setup)
  # Use CMAKE_HIP_ARCHITECTURES to select the GPU targets. When unset, enable_language(HIP) detects the GPUs present.
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_HIP_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED})
  # nvcc implicitly includes cuda_runtime.h in every .cu file. Do the same, resolving to gputil/cuda/hip_compat.
  add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:SHELL:-include cuda_runtime.h>")
  find_package(hip REQUIRED)
  set(OHM_CUDA_LIBRARY hip::host)
endmacro(cuda_setup)
