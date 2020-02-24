int n;
inline void process_args(int argc, char** argv) {
  if (argc > 1) {
    int n = std::atoi(argv[1]);
  } else {
    n = 3;
  }
  std::cout << "Using N = " << n << std::endl;
}

