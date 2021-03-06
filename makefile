.PHONY: clean all

all: intel

intel: main.cpp Util.hpp
	dpcpp -std=c++14 -fsycl -I./ ./main.cpp -o intel

codeplay: main.cpp Util.hpp
	compute++ -std=c++14 -sycl -sycl-driver -lComputeCpp ./main.cpp -o codeplay

hipsycl: main.cpp Util.hpp
	syclcc-clang-wrapper -std=c++14 ./main.cpp -o hipsycl  -ffast-math

cudasycl: main.cpp Util.hpp
	clang++ -DCUDA -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice ./main.cpp -o cudasycl

clean:
	rm -f ./intel ./codeplay ./hipsycl
