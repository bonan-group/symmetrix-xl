#include <cstdio>
#include <cstring>

#include <mpi.h>

#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: mpi_gpu_aware_probe cuda|rocm\n");
    return 2;
  }

  int compile_time = 0;
  int runtime = 0;
  const char *method = "unavailable";
  if (std::strcmp(argv[1], "cuda") == 0) {
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    compile_time = MPIX_CUDA_AWARE_SUPPORT;
    runtime = MPIX_Query_cuda_support();
    method = "MPIX_Query_cuda_support";
#endif
  } else if (std::strcmp(argv[1], "rocm") == 0) {
#if defined(MPIX_ROCM_AWARE_SUPPORT)
    compile_time = MPIX_ROCM_AWARE_SUPPORT;
    runtime = MPIX_Query_rocm_support();
    method = "MPIX_Query_rocm_support";
#endif
  } else {
    std::fprintf(stderr, "unsupported accelerator: %s\n", argv[1]);
    return 2;
  }

  std::printf("method=%s compile_time=%d runtime=%d\n", method, compile_time,
              runtime);
  return std::strcmp(method, "unavailable") != 0 && runtime != 0 ? 0 : 3;
}
