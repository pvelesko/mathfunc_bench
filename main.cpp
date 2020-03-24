#include <CL/sycl.hpp>
#include "Util.hpp"
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <limits>
#include <chrono>

using namespace cl::sycl;
#define RTYPE double

//float elapsed(event e) {
//  float t0 = e.get_profiling_info<info::event_profiling::command_start>();
//  float t1 = e.get_profiling_info<info::event_profiling::command_end>();
//  float t = (t1 - t0) / 1000000.0f;
//  std::cout << t << "ms" << std::endl;
//  return t;
//}

class CUDASelector : public cl::sycl::device_selector {
  public:
    int operator()(const cl::sycl::device &Device) const override {

      const std::string DriverVersion = Device.get_info<info::device::driver_version>();

      if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos)) {
        return 1;
      };
      return -1;
    }
};

RTYPE diff(RTYPE* ref, RTYPE* x) {
  typedef std::numeric_limits<RTYPE> flt;
  std::cout.precision(flt::max_digits10);
  RTYPE d = abs(ref[0] - x[0]);
  RTYPE p = abs(ref[0] / x[0] * 100 - 100);
  std::cout << std::fixed << "Absolute diff: " << d << std::endl;
  std::cout << std::fixed << "Percent diff: " << p << "%" << std::endl;
  return d;
}

int main(int argc, char** argv) {
  process_args(argc, argv);
  auto property_list =
          //cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
          cl::sycl::property_list{};
  std::string env;
  if (std::getenv("SYCL_DEVICE") != NULL) {
    env = std::string(std::getenv("SYCL_DEVICE"));
  } else {
    env = std::string("");
  }
  queue q;
  std::cout << "Using DEVICE = " << env << std::endl;
  if (!env.compare("gpu") or !env.compare("GPU")) {
    q = cl::sycl::queue(cl::sycl::gpu_selector{}, exception_handler, property_list);
  } else if (!env.compare("cpu") or !env.compare("CPU")) {
    q = cl::sycl::queue(cl::sycl::cpu_selector{}, exception_handler, property_list);
  } else if (!env.compare("host") or !env.compare("HOST")) {
    q = cl::sycl::queue(cl::sycl::host_selector{}, exception_handler, property_list);
  } else if (!env.compare("cuda") or !env.compare("CUDA")) {
    q = cl::sycl::queue(CUDASelector{}, exception_handler, property_list);
  } else {
    q = cl::sycl::queue(cl::sycl::default_selector{}, exception_handler, property_list);
  }

  //init();

  RTYPE* x = static_cast<RTYPE*>(malloc(2 * sizeof(RTYPE)));
  RTYPE* ref = static_cast<RTYPE*>(malloc(2 * sizeof(RTYPE)));
  x[0] = 0;
  ref[0] = 0;
  x[1] = 1.337;
  ref[1] = 1.337;

  for(int i = 0; i < n; i++)
    ref[0] += sqrt(ref[1]);

  int size = n; 
  {
  buffer<RTYPE, 1> x_d(x, range<1>(2));
  auto t0 = std::chrono::high_resolution_clock::now();
  auto e = q.submit([&] (handler& cgh) { // q scope
    auto x = x_d.get_access<access::mode::read_write>(cgh);
    cgh.single_task<class _sqrt>([=] {
      for(int i = 0; i < size; i++)
        x[0] += sqrt(x[1]);
    }); // end task scope
  }); // end q scope
  e.wait();
  auto t1 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t1 - t0 ).count();
  double cost = double(duration) / n;
  std::cout << std::setw(6) << std::setprecision(4) << "sqrt: " << cost << " us/call" << std::endl;
  } // buffer scope
  diff(ref, x);

  return 0;
}
