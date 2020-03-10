using namespace cl::sycl;
int n;
queue q;
device dev;
context ctx;

void init() {
  q = queue(gpu_selector{});
  dev = q.get_device();
  ctx = q.get_context();
  std::cout << "Running on "
            << dev.get_info<info::device::name>()
            << std::endl;
};

inline void process_args(int argc, char** argv) {
  if (argc > 1) {
    int n = std::atoi(argv[1]);
  } else {
    n = 3;
  }
  std::cout << "Using N = " << n << std::endl;
}

template<class T>
inline void dump(T* var, std::string name) {
  for(int i = 0; i < n; i++)
    std::cout << name << "[" << i << "] = " << var[i] << std::endl;
}

