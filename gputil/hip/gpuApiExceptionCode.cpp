// Copyright (c) 2017
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// ABN 41 687 119 230
//
// Author: Kazys Stepanas

#include "gputil/gpuApiException.h"

#include <hip/hip_runtime.h>

namespace gputil
{
const char *ApiException::errorCodeString(int error_code)
{
  return hipGetErrorString(static_cast<hipError_t>(error_code));
}
}  // namespace gputil
