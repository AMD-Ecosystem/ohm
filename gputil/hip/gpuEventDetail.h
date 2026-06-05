// Copyright (c) 2017
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// ABN 41 687 119 230
//
// Author: Kazys Stepanas
#ifndef GPUEVENTDETAIL_H
#define GPUEVENTDETAIL_H

#include "gpuConfig.h"

#include "gputil/hip/ref.h"

#include <hip/hip_runtime.h>

#include <utility>

namespace gputil
{
void destroyEvent(hipEvent_t event);

struct EventDetail : public Ref<hipEvent_t>
{
  inline EventDetail(hipEvent_t obj, unsigned initial_ref_count, const ReleaseFunc &release)
    : Ref<hipEvent_t>(obj, initial_ref_count, release)
  {}

  explicit inline EventDetail(Ref &&other) noexcept
    : Ref<hipEvent_t>(std::move(other))
  {}

  inline EventDetail(const Ref &other) = delete;

protected:
  inline ~EventDetail() = default;
};
}  // namespace gputil

#endif  // GPUEVENTDETAIL_H
