#include <CL/sycl.hpp>
#include "Util.hpp"
#include <stdlib.h>
#include <iostream>
using namespace cl::sycl;
int main(int argc, char** argv) {
  process_args(argc, argv);
  init();
  int num[3] = {1, 1, 0};

  { // buffer scope
    buffer<int, 1> num_d(num, range<1>(3));
    q.submit([&] (handler& cgh) { // q scope
      auto num = num_d.get_access<access::mode::read_write>(cgh);
      cgh.single_task<class oneplusone>([=] {
        num[2] = num[0] + cl::sycl::ceil(3.2);
      }); // end task scope
    }); // end q scope
  } // end buffer scope

  std::cout << "1 + 1 = " << num[2] << std::endl;

  return 0;
}
