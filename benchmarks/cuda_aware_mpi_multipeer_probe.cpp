#include <cuda_runtime.h>
#include <mpi.h>

#include <array>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace {

bool cuda_ok(cudaError_t status, const char *operation, int rank)
{
  if (status == cudaSuccess) return true;
  std::fprintf(
    stderr, "rank=%d %s failed: %s\n", rank, operation,
    cudaGetErrorString(status));
  return false;
}

bool parse_mebibytes(const char *text, std::size_t &bytes)
{
  errno = 0;
  char *end = nullptr;
  const unsigned long long mebibytes = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') return false;
  constexpr std::size_t bytes_per_mebibyte = 1024ULL * 1024ULL;
  if (mebibytes == 0 || mebibytes > INT_MAX / bytes_per_mebibyte)
    return false;
  bytes = static_cast<std::size_t>(mebibytes) * bytes_per_mebibyte;
  return true;
}

bool any_rank_failed(bool failed, MPI_Comm comm, int rank)
{
  const int local_failed = failed ? 1 : 0;
  int global_failed = 0;
  const int status = MPI_Allreduce(
    &local_failed, &global_failed, 1, MPI_INT, MPI_MAX, comm);
  if (status != MPI_SUCCESS) {
    std::fprintf(stderr, "rank=%d MPI_Allreduce failed: %d\n", rank, status);
    return true;
  }
  return global_failed != 0;
}

}  // namespace

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank = -1;
  int size = 0;
  int local_rank = -1;
  int local_size = 0;
  MPI_Comm local = MPI_COMM_NULL;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_split_type(
    MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
  MPI_Comm_rank(local, &local_rank);
  MPI_Comm_size(local, &local_size);

  std::size_t bytes = 0;
  bool failed = argc != 2 || !parse_mebibytes(argc == 2 ? argv[1] : "", bytes);
  if (failed && rank == 0)
    std::fprintf(stderr, "usage: %s MESSAGE_MIB\n", argv[0]);

  int device_count = 0;
  failed = !cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", rank)
    || failed;
  if (!failed && (device_count < local_size || local_size != size)) {
    std::fprintf(
      stderr,
      "rank=%d expected one node with one GPU per rank; devices=%d "
      "local_size=%d size=%d\n",
      rank, device_count, local_size, size);
    failed = true;
  }
  if (!failed)
    failed = !cuda_ok(cudaSetDevice(local_rank), "cudaSetDevice", rank);

  failed = any_rank_failed(failed, MPI_COMM_WORLD, rank);

  constexpr int repetitions = 4;
  const std::array<int, 3> distances{1, 2, 4};
  for (int repetition = 0; repetition < repetitions && !failed; ++repetition) {
    unsigned char *send = nullptr;
    unsigned char *receive = nullptr;
    failed = !cuda_ok(cudaMalloc(&send, bytes), "cudaMalloc(send)", rank)
      || !cuda_ok(cudaMalloc(&receive, bytes), "cudaMalloc(receive)", rank);
    if (any_rank_failed(failed, MPI_COMM_WORLD, rank)) {
      if (receive != nullptr) cudaFree(receive);
      if (send != nullptr) cudaFree(send);
      failed = true;
      break;
    }

    const auto value = static_cast<unsigned char>(rank + 17 + repetition);
    failed = !cuda_ok(cudaMemset(send, value, bytes), "cudaMemset(send)", rank)
      || !cuda_ok(cudaMemset(receive, 0, bytes), "cudaMemset(receive)", rank)
      || !cuda_ok(
        cudaDeviceSynchronize(), "cudaDeviceSynchronize(pre-MPI)", rank);
    if (any_rank_failed(failed, MPI_COMM_WORLD, rank)) {
      cudaFree(receive);
      cudaFree(send);
      failed = true;
      break;
    }

    for (const int distance : distances) {
      if (distance >= size) continue;
      const int destination = (rank + distance) % size;
      const int source = (rank - distance + size) % size;
      const int mpi_status = MPI_Sendrecv(
        send, static_cast<int>(bytes), MPI_BYTE, destination, 700 + distance,
        receive, static_cast<int>(bytes), MPI_BYTE, source, 700 + distance,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (mpi_status != MPI_SUCCESS) {
        std::fprintf(stderr, "rank=%d MPI_Sendrecv failed: %d\n", rank, mpi_status);
        failed = true;
      }

      std::array<unsigned char, 3> samples{};
      const auto expected = static_cast<unsigned char>(source + 17 + repetition);
      if (!failed) {
        const std::array<std::size_t, 3> offsets{0, bytes / 2, bytes - 1};
        for (std::size_t i = 0; i < offsets.size(); ++i)
          failed = !cuda_ok(
            cudaMemcpy(
              &samples[i], receive + offsets[i], 1, cudaMemcpyDeviceToHost),
            "cudaMemcpy(sample)", rank)
            || failed;
        for (const auto sample : samples) failed = failed || sample != expected;
      }
      if (mpi_status == MPI_SUCCESS && failed)
        std::fprintf(
          stderr,
          "rank=%d repetition=%d distance=%d source=%d expected=%u "
          "samples=%u,%u,%u\n",
          rank, repetition, distance, source, static_cast<unsigned>(expected),
          static_cast<unsigned>(samples[0]), static_cast<unsigned>(samples[1]),
          static_cast<unsigned>(samples[2]));
      failed = any_rank_failed(failed, MPI_COMM_WORLD, rank);
      if (failed) break;
    }

    failed = !cuda_ok(cudaFree(receive), "cudaFree(receive)", rank) || failed;
    failed = !cuda_ok(cudaFree(send), "cudaFree(send)", rank) || failed;
    failed = any_rank_failed(failed, MPI_COMM_WORLD, rank);
  }

  int local_failed = failed ? 1 : 0;
  int any_failed = 0;
  MPI_Allreduce(
    &local_failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  std::printf(
    "rank=%d local_rank=%d repetitions=%d bytes=%zu status=%s\n", rank,
    local_rank, repetitions, bytes, any_failed ? "failed" : "ok");
  std::fflush(stdout);

  MPI_Comm_free(&local);
  MPI_Finalize();
  return any_failed ? 3 : 0;
}
