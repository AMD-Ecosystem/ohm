# OHM_FEATURE_HIP is used to enable HIP/ROCm builds.
# This mirrors the structure of OhmCuda.cmake.

set(OHM_FEATURE_HIP_DEFAULT OFF)

# Check if HIP is available
include(CheckLanguage)
check_language(HIP)
if(CMAKE_HIP_COMPILER)
  set(OHM_FEATURE_HIP_DEFAULT ON)
endif(CMAKE_HIP_COMPILER)

option(OHM_FEATURE_HIP "Build ohm library and utilities for HIP/ROCm?" ${OHM_FEATURE_HIP_DEFAULT})

if(OHM_FEATURE_HIP)
  # enable_language(HIP) honors an explicit -DCMAKE_HIP_ARCHITECTURES, otherwise
  # auto-detects the host GPU(s), and errors on a no-GPU build host.
  enable_language(HIP)

  # Match HIP standard to the C++ standard for the project
  set(CMAKE_HIP_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_HIP_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED})

  # HIP runtime library - hip::device for HIP kernel files, hip::host for pure C++ linking HIP
  find_package(hip REQUIRED)
  set(OHM_HIP_LIBRARY hip::device)
  set(OHM_HIP_HOST_LIBRARY hip::host)

  message(STATUS "HIP enabled for architectures: ${CMAKE_HIP_ARCHITECTURES}")
endif(OHM_FEATURE_HIP)
