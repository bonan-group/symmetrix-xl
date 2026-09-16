ARG CUDA_IMAGE=nvidia/cuda:13.3.1-cudnn-devel-rockylinux8@sha256:b0c3200d0bf08f931bc51ba01953eec596d8cec5bbf3748236b23facaec489ce
FROM ${CUDA_IMAGE} AS cuda

FROM quay.io/pypa/manylinux_2_28_x86_64@sha256:531d7aa844bbb0c131d4ab011d3db741c4abc8d498cd5ccc86121046f62303b4

ARG CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
ARG CMAKE_VERSION=4.4.3
ARG CMAKE_SHA256=d6c83076c575bc00b823522ac974bda66d0af05d6ddc30e739c12385cf32c6cc
ARG OPENBLAS_VERSION=0.3.29
ARG OPENBLAS_SHA256=38240eee1b29e2bde47ebb5d61160207dc68668a54cac62c076bb5032013b1eb

COPY --from=cuda ${CUDA_TOOLKIT_PATH}/ /usr/local/cuda/

RUN dnf install -y binutils gcc-gfortran gcc-toolset-13-gcc \
      gcc-toolset-13-gcc-c++ git make patchelf perl-core which \
    && dnf clean all \
    && test -x /usr/local/cuda/bin/nvcc

RUN curl -fsSL -o /tmp/cmake.tar.gz \
      "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
    && echo "${CMAKE_SHA256}  /tmp/cmake.tar.gz" | sha256sum -c - \
    && tar xzf /tmp/cmake.tar.gz -C /opt \
    && ln -sf "/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/cmake" /usr/local/bin/cmake \
    && rm -f /tmp/cmake.tar.gz

RUN curl -fsSL -o /tmp/openblas.tar.gz \
      "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz" \
    && echo "${OPENBLAS_SHA256}  /tmp/openblas.tar.gz" | sha256sum -c - \
    && tar xzf /tmp/openblas.tar.gz -C /tmp \
    && make -s -C "/tmp/OpenBLAS-${OPENBLAS_VERSION}" -j8 \
      TARGET=HASWELL DYNAMIC_ARCH=1 NO_AFFINITY=1 NOFORTRAN=1 NO_SHARED=1 \
      NO_LAPACK=1 NO_LAPACKE=1 \
      USE_OPENMP=0 USE_THREAD=0 NUM_THREADS=1 \
      CFLAGS="-O2 -fPIC" \
    && make -s -C "/tmp/OpenBLAS-${OPENBLAS_VERSION}" \
      TARGET=HASWELL DYNAMIC_ARCH=1 NO_AFFINITY=1 NOFORTRAN=1 NO_SHARED=1 \
      NO_LAPACK=1 NO_LAPACKE=1 \
      USE_OPENMP=0 USE_THREAD=0 NUM_THREADS=1 \
      PREFIX=/opt/symmetrix-openblas install \
    && test -f /opt/symmetrix-openblas/lib/libopenblas.a \
    && test -f /opt/symmetrix-openblas/include/cblas.h \
    && rm -rf /tmp/openblas.tar.gz "/tmp/OpenBLAS-${OPENBLAS_VERSION}"

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/usr/local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

WORKDIR /io
