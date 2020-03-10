#include <CL/sycl.hpp>
#include "Util.hpp"
#include <stdlib.h>
#include <iostream>
using namespace cl::sycl;
int main(int argc, char** argv) {
  process_args(argc, argv);
  init();

  q.submit([&] (handler& cgh) { // q scope
    cgh.single_task<class oneplusone>([=] {
    }); // end task scope
  }); // end q scope

  return 0;
}
