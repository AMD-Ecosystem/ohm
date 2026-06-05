// Copyright (c) 2017
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// ABN 41 687 119 230
//
// Author: Kazys Stepanas
#include "gputil/gpuEvent.h"

#include "gputil/hip/gpuEventDetail.h"

#include "gputil/gpuApiException.h"
#include "gputil/gpuThrow.h"

#include <hip/hip_runtime.h>

namespace gputil
{
void destroyEvent(hipEvent_t event)
{
  if (event)
  {
    hipError_t err = hipEventDestroy(event);
    event = nullptr;
    GPUAPICHECK2(err, hipSuccess);
  }
}

Event::Event() = default;

Event::Event(const Event &other)
{
  if (other.imp_)
  {
    imp_ = other.imp_;
    imp_->reference();
  }
}


Event::Event(Event &&other) noexcept
  : imp_(other.imp_)
{
  other.imp_ = nullptr;
}


Event::~Event()
{
  release();
}


bool Event::isValid() const
{
  return imp_ && imp_->obj();
}


void Event::release()
{
  if (imp_)
  {
    imp_->release();
    imp_ = nullptr;
  }
}


bool Event::isComplete() const
{
  if (!isValid())
  {
    return true;
  }

  hipError_t status = hipEventQuery(imp_->obj());
  switch (status)
  {
  case hipSuccess:
  case hipErrorNotReady:
    break;
  default:
    GPUAPICHECK(status, hipSuccess, true);
    break;
  }

  return status == hipSuccess;
}


void Event::wait() const
{
  if (imp_ && imp_->obj())
  {
    hipError_t err = hipEventSynchronize(imp_->obj());
    GPUAPICHECK2(err, hipSuccess);
  }
}


void Event::wait(const Event *events, size_t event_count)
{
  for (size_t i = 0; i < event_count; ++i)
  {
    events[i].wait();
  }
}


void Event::wait(const Event **events, size_t event_count)
{
  for (size_t i = 0; i < event_count; ++i)
  {
    events[i]->wait();
  }
}


Event &Event::operator=(const Event &other)
{
  if (this != &other)
  {
    release();
    if (other.imp_)
    {
      imp_ = other.imp_;
      imp_->reference();
    }
  }

  return *this;
}


Event &Event::operator=(Event &&other) noexcept
{
  release();
  imp_ = other.imp_;
  other.imp_ = nullptr;
  return *this;
}


EventDetail *Event::detail()
{
  // Detail requested. Allocate if required.
  if (!imp_)
  {
    hipEvent_t event = nullptr;
    hipError_t err = hipEventCreateWithFlags(&event, hipEventBlockingSync);
    GPUAPICHECK(err, hipSuccess, nullptr);
    imp_ = new EventDetail(event, 1, &destroyEvent);  // NOLINT(cppcoreguidelines-owning-memory)
  }
  return imp_;
}


EventDetail *Event::detail() const
{
  return imp_;
}
}  // namespace gputil
