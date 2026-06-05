// Copyright (c) 2017
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// ABN 41 687 119 230
//
// Author: Kazys Stepanas

#include "gputil/gpuDevice.h"

#include "gputil/hip/gpuDeviceDetail.h"

#include "gputil/gpuApiException.h"
#include "gputil/gpuThrow.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <sstream>

namespace gputil
{
namespace
{
void initDeviceInfo(DeviceInfo &gputil_info, const hipDeviceProp_t &cuda_info)
{
  gputil_info.name = cuda_info.name;
  gputil_info.platform = "HIP";  // cuda_info.name;
  gputil_info.version.major = cuda_info.major;
  gputil_info.version.minor = cuda_info.minor;
  gputil_info.version.patch = 0;
  gputil_info.type = kDeviceGpu;
}

bool selectDevice(DeviceDetail &detail, int device_id = -1)
{
  hipError_t err = hipSuccess;

  if (device_id < 0)
  {
    // Select default device.
    err = hipGetDevice(&device_id);
  }

  if (err == hipSuccess)
  {
    hipDeviceProp_t cuda_info{};
    memset(&cuda_info, 0, sizeof(cuda_info));
    if (hipGetDeviceProperties(&cuda_info, device_id) == hipSuccess)
    {
      detail.device = device_id;
      initDeviceInfo(detail.info, cuda_info);
      return true;
    }
  }

  detail.device = -1;
  detail.info = DeviceInfo();
  detail.default_queue = Queue(nullptr);

  return false;
}
}  // namespace

Device::Device(bool default_device)
  : imp_(std::make_unique<DeviceDetail>())
{
  imp_->device = -1;
  if (default_device)
  {
    selectDevice(*imp_);
  }
}


Device::Device(const DeviceInfo &device_info)
  : imp_(std::make_unique<DeviceDetail>())
{
  select(device_info);
}


Device::Device(int argc, const char **argv, const char *default_device, unsigned device_type_flags)
  : imp_(std::make_unique<DeviceDetail>())
{
  select(argc, argv, default_device, device_type_flags);
}


Device::Device(const Device &other)
  : imp_(std::make_unique<DeviceDetail>())
{
  *imp_ = *other.imp_;
}


Device::Device(Device &&other) noexcept
  : imp_(std::move(other.imp_))
{}


Device::~Device() = default;


unsigned Device::enumerateDevices(std::vector<DeviceInfo> &devices)
{
  int device_count = 0;
  hipGetDeviceCount(&device_count);

  for (int i = 0; i < device_count; ++i)
  {
    hipDeviceProp_t cuda_info{};
    DeviceInfo gputil_info;
    memset(&cuda_info, 0, sizeof(cuda_info));
    if (hipGetDeviceProperties(&cuda_info, i) == hipSuccess)
    {
      initDeviceInfo(gputil_info, cuda_info);
      devices.push_back(gputil_info);
    }
  }

  return static_cast<unsigned>(device_count);
}


const char *Device::name() const
{
  return imp_->name.c_str();
}


const char *Device::description() const
{
  return imp_->name.c_str();
}


const DeviceInfo &Device::info() const
{
  return imp_->info;
}


Queue Device::defaultQueue() const
{
  return imp_->default_queue;
}


Queue Device::createQueue(unsigned /*flags*/) const
{
  hipStream_t stream = nullptr;
  hipError_t err;
  err = hipSetDevice(imp_->device);
  GPUAPICHECK(err, hipSuccess, Queue());
  err = hipStreamCreate(&stream);
  GPUAPICHECK(err, hipSuccess, Queue());
  return Queue(stream);
}


bool Device::select(int argc, const char **argv, const char *default_device, unsigned device_type_flags)
{
  if ((device_type_flags & kGpu) == 0)
  {
    // Only GPU supported for CUDA.
    return false;
  }

  // Resolve constraints:
  // - (Partial) device name match
  // - Compute capability version number.

  std::string name_constraint = default_device ? default_device : "";
  Version version_min;

  const std::string device_arg = "--device";
  const std::string clver_arg = "--clver";
  for (int i = 0; i < argc; ++i)
  {
    if (strncmp(device_arg.c_str(), argv[i], device_arg.size()) == 0)
    {
      // Read value from "--device="
      if (argv[i][device_arg.size()] == '=')
      {
        // Clip '='
        name_constraint = argv[i] + device_arg.size() + 1;
        // Strip quotes
        if (!name_constraint.empty())
        {
          if (name_constraint.front() == '"')
          {
            name_constraint.erase(name_constraint.begin());
            if (!name_constraint.empty())
            {
              if (name_constraint.back() == '"')
              {
                name_constraint.pop_back();
              }
            }
          }
        }
      }
    }
    else if (strncmp(clver_arg.c_str(), argv[i], clver_arg.size()) == 0)
    {
      // Read version from "--clver="
      if (argv[i][clver_arg.size()] == '=')
      {
        char dot;
        std::string ver_str = argv[i] + clver_arg.size() + 1;
        std::istringstream in(ver_str);
        in >> version_min.major;
        in >> dot;
        in >> version_min.minor;
      }
    }
  }

  std::vector<DeviceInfo> devices;
  enumerateDevices(devices);

  std::transform(name_constraint.begin(), name_constraint.end(), name_constraint.begin(), ::tolower);

  const auto name_check = [](const std::string &device_name, const std::string &constraint) -> bool  //
  {
    if (constraint.empty())
    {
      return true;
    }

    std::string device_lower = device_name;
    std::transform(device_lower.begin(), device_lower.end(), device_lower.begin(), ::tolower);
    return device_lower.find(constraint) != std::string::npos;
  };

  const auto ver_check = [](const Version &device_version, const Version &constraint) -> bool  //
  {
    if (device_version.major > constraint.major)
    {
      return true;
    }
    if (device_version.major == constraint.major)
    {
      if (device_version.minor > constraint.minor)
      {
        return true;
      }
      if (device_version.minor == constraint.minor)
      {
        if (device_version.patch > constraint.patch)
        {
          return true;
        }
        if (device_version.patch == constraint.patch)
        {
          return true;
        }
      }
    }

    return false;
  };

  for (auto &&device : devices)
  {
    if (name_check(device.name, name_constraint) && ver_check(device.version, version_min))
    {
      return select(device);
    }
  }

  return false;
}


bool Device::select(const DeviceInfo &device_info)
{
  int device_count = 0;
  hipGetDeviceCount(&device_count);

  for (int i = 0; i < device_count; ++i)
  {
    hipDeviceProp_t cuda_info{};
    DeviceInfo gputil_info;
    memset(&cuda_info, 0, sizeof(cuda_info));
    if (hipGetDeviceProperties(&cuda_info, i) == hipSuccess)
    {
      initDeviceInfo(gputil_info, cuda_info);
      if (gputil_info == device_info)
      {
        return selectDevice(*imp_, i);
      }
    }
  }

  return false;
}


// Lint(KS): required for API compatibility
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void Device::setDebugGpu(DebugLevel /*level*/)
{
  // Ignored for CUDA.
}


// Lint(KS): required for API compatibility
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
Device::DebugLevel Device::debugGpu() const
{
  return kDlOff;
}


// Lint(KS): required for API compatibility
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
bool Device::supportsFeature(const char * /*feature_id*/) const
{
  // TODO(KS): find CUDA feature checking or use compute capabilities.
  return false;
}


// Lint(KS): required for API compatibility
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void Device::addSearchPath(const char * /*path*/)
{
  // Ignored for CUDA.
}


// Lint(KS): required for API compatibility
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
const char *Device::searchPaths() const
{
  return "";
}


bool Device::isValid() const
{
  return imp_ && imp_->device >= 0;
}


uint64_t Device::deviceMemory() const
{
  hipError_t err = hipSuccess;
  hipDeviceProp_t cuda_info{};
  memset(&cuda_info, 0, sizeof(cuda_info));
  err = hipGetDeviceProperties(&cuda_info, imp_->device);
  GPUAPICHECK(err, hipSuccess, 0u);
  return cuda_info.totalGlobalMem;
}


uint64_t Device::maxAllocationSize() const
{
  // Is this right?
  return deviceMemory();
}


bool Device::unifiedMemory() const
{
  hipDeviceProp_t cuda_info{};
  memset(&cuda_info, 0, sizeof(cuda_info));
  hipError_t err = hipGetDeviceProperties(&cuda_info, imp_->device);
  GPUAPICHECK(err, hipSuccess, false);
  // This is a bit of an assumption.
  return cuda_info.integrated != 0;
}


Device &Device::operator=(const Device &other)
{
  if (this != &other)
  {
    *imp_ = *other.imp_;
  }
  return *this;
}


Device &Device::operator=(Device &&other) noexcept
{
  imp_ = std::move(other.imp_);
  return *this;
}
}  // namespace gputil