FROM quay.io/pypa/manylinux_2_28_x86_64@sha256:531d7aa844bbb0c131d4ab011d3db741c4abc8d498cd5ccc86121046f62303b4

ARG CMAKE_VERSION=4.4.3
ARG CMAKE_SHA256=d6c83076c575bc00b823522ac974bda66d0af05d6ddc30e739c12385cf32c6cc
ARG OPENBLAS_VERSION=0.3.29
ARG OPENBLAS_SHA256=38240eee1b29e2bde47ebb5d61160207dc68668a54cac62c076bb5032013b1eb
ARG FFTW_VERSION=3.3.10
ARG FFTW_SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467

RUN dnf install -y binutils gcc-gfortran git make patchelf perl-core which \
    && dnf clean all

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
      TARGET=HASWELL DYNAMIC_ARCH=1 NO_AFFINITY=1 NO_SHARED=0 \
      USE_OPENMP=1 NUM_THREADS=64 CFLAGS="-O2 -fPIC" \
    && make -s -C "/tmp/OpenBLAS-${OPENBLAS_VERSION}" \
      TARGET=HASWELL DYNAMIC_ARCH=1 NO_AFFINITY=1 NO_SHARED=0 \
      USE_OPENMP=1 NUM_THREADS=64 \
      PREFIX=/opt/symmetrix-openblas install \
    && test -f /opt/symmetrix-openblas/lib/libopenblas.so \
    && rm -rf /tmp/openblas.tar.gz "/tmp/OpenBLAS-${OPENBLAS_VERSION}"

# The CPU wheel publishes the reciprocal-space D3 correction, so the build
# requires an FFTW3 provider (SYMMETRIX_PARTICLE_MESH=ON) and auditwheel
# grafts these libraries into the manylinux wheel.
# FFTW builds one precision per configure, so double and float need
# separate out-of-tree passes into the same prefix.
RUN curl -fsSL -o /tmp/fftw.tar.gz \
      "https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz" \
    && echo "${FFTW_SHA256}  /tmp/fftw.tar.gz" | sha256sum -c - \
    && tar xzf /tmp/fftw.tar.gz -C /tmp \
    && mkdir /tmp/fftw-double /tmp/fftw-single \
    && cd /tmp/fftw-double \
    && "/tmp/fftw-${FFTW_VERSION}/configure" --prefix=/opt/symmetrix-fftw \
      --enable-shared --disable-static --enable-openmp \
      CFLAGS="-O2 -fPIC -fopenmp" \
    && make -s -j8 \
    && make -s install \
    && cd /tmp/fftw-single \
    && "/tmp/fftw-${FFTW_VERSION}/configure" --prefix=/opt/symmetrix-fftw \
      --enable-shared --disable-static --enable-openmp --enable-float \
      CFLAGS="-O2 -fPIC -fopenmp" \
    && make -s -j8 \
    && make -s install \
    && test -f /opt/symmetrix-fftw/lib/libfftw3.so \
    && test -f /opt/symmetrix-fftw/lib/libfftw3f.so \
    && test -f /opt/symmetrix-fftw/lib/libfftw3_omp.so \
    && test -f /opt/symmetrix-fftw/lib/libfftw3f_omp.so \
    && test -f /opt/symmetrix-fftw/include/fftw3.h \
    && rm -rf /tmp/fftw.tar.gz "/tmp/fftw-${FFTW_VERSION}" /tmp/fftw-double \
      /tmp/fftw-single

ENV CMAKE_PREFIX_PATH=/opt/symmetrix-openblas:/opt/symmetrix-fftw \
    PATH=/usr/local/bin:${PATH}

WORKDIR /io
