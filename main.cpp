#include <CL/sycl.hpp>
#include "Util.hpp"
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <limits>
using namespace cl::sycl;


float elapsed(event e) {
  float t0 = e.get_profiling_info<info::event_profiling::command_start>();
  float t1 = e.get_profiling_info<info::event_profiling::command_end>();
  float t = (t1 - t0) / 1000000.0f;
  std::cout << t << "ms" << std::endl;
  return t;
}

float diff(float* ref, float* x) {
  typedef std::numeric_limits<float> flt;
  std::cout.precision(flt::max_digits10);
  float d = fabs(ref[0] - x[0]);
  float p = fabs(ref[0] / x[0] * 100 - 100);
  std::cout << std::fixed << "Absolute diff: " << d << std::endl;
  std::cout << std::fixed << "Percent diff: " << p << "%" << std::endl;
  return d;
}

int main(int argc, char** argv) {
  process_args(argc, argv);
  auto property_list =
          cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
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
  } else {
    q = cl::sycl::queue(cl::sycl::default_selector{}, exception_handler, property_list);
  }

  //init();

  float* x = static_cast<float*>(malloc_shared(n * sizeof(float), q));
  float* ref = static_cast<float*>(malloc_shared(n * sizeof(float), q));
  for (int i = 0; i < n; i++) {
    x[i] = 1.23f;
    ref[i] = 1.23f;
  }
  x[0] = 0;
  ref[0] = 0;

  for(int i = 1; i < n; i++)
    ref[0] += sqrt(ref[i]);
  std::cout << ref[0];

  int size = n;
  auto e = q.submit([&] (handler& cgh) { // q scope
    cgh.single_task<class _sqrt>([=] {
      for(int i = 1; i < size; i++)
        x[0] += cos(x[i]);
    }); // end task scope
  }); // end q scope
  e.wait();
  elapsed(e);
  diff(ref, x);

  return 0;
}
