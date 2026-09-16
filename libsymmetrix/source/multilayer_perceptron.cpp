#include <cmath>
#include <stdexcept>
#include <utility>

#include "cblas.hpp"

#include "multilayer_perceptron.hpp"

template <typename Precision>
MultilayerPerceptronT<Precision>::MultilayerPerceptronT()
{
    // TODO: add sanity checks to default constructor
}

template <typename Precision>
MultilayerPerceptronT<Precision>::MultilayerPerceptronT(
    std::vector<int> shape,
    std::vector<std::vector<Precision>> weights,
    Precision activation_scale_factor)
    : shape(shape),
      weights(weights),
      activation_scale_factor(activation_scale_factor)
{
    // TODO: check and sanitize input
    // TODO: double check that something sensible happens here for i==shape.size()-1
    for (int i=0; i<shape.size(); ++i) {
        node_values.push_back(std::vector<Precision>(shape[i]));
        node_derivs.push_back(std::vector<Precision>(shape.back()*shape[i]));
        node_activation_derivs.push_back(std::vector<Precision>(shape[i]));
    }
}

template <typename Precision>
auto MultilayerPerceptronT<Precision>::evaluate(
    std::vector<Precision> input
    )-> std::vector<Precision>
{
    // TODO: Check/sanitize input
    // Reshape node arrays and send input to nodes
    for (int i=0; i<shape.size(); ++i) {
        node_values[i].resize(shape[i]);
    }
    std::copy(input.begin(), input.end(), node_values[0].begin());
    // Evaluate layers
    for (int l=0; l<shape.size()-2; ++l) {
        symmetrix_blas_gemv<Precision>(
            CblasRowMajor,            // const CBLAS_LAYOUT Layout
            CblasNoTrans,             // const CBLAS_TRANSPOSE trans
            shape[l+1],               // const MKL_INT m
            shape[l],                 // const MKL_INT n
            1.0,                      // const Precision alpha
            weights[l].data(),        // const Precision *a
            shape[l],                 // const MKL_INT lda
            node_values[l].data(),    // const Precision *x
            1,                        // const MKL_INT incx
            0.0,                      // const Precision beta
            node_values[l+1].data(),  // Precision *y
            1);                       // const MKL_INT incy
        for (int i=0; i<shape[l+1]; ++i) {
            const Precision x = node_values[l+1][i];
            node_values[l+1][i] = activation_scale_factor*x/(1.0+std::exp(-x));
        }
    }
    // Evaluate final layer (no nonlinearity)
    symmetrix_blas_gemv<Precision>(
        CblasRowMajor,                 // const CBLAS_LAYOUT Layout
        CblasNoTrans,                  // const CBLAS_TRANSPOSE trans
        shape.end()[-1],               // const MKL_INT m
        shape.end()[-2],               // const MKL_INT n
        1.0,                           // const Precision alpha
        weights.back().data(),         // const Precision *a
        shape.end()[-2],               // const MKL_INT lda
        node_values.end()[-2].data(),  // const Precision *x
        1,                             // const MKL_INT incx
        0.0,                           // const Precision beta
        node_values.end()[-1].data(),  // Precision *y
        1);                            // const MKL_INT incy
    return node_values.back();
}

template <typename Precision>
auto  MultilayerPerceptronT<Precision>::evaluate_gradient(
    std::vector<Precision> input
    )-> std::tuple<std::vector<Precision>,std::vector<Precision>>
{
    // TODO: check/sanitize input
    // Reshape node arrays and send input to nodes
    for (int i=0; i<shape.size(); ++i) {
        node_values[i].resize(shape[i]);
        node_activation_derivs[i].resize(shape[i]);
        node_derivs[i].resize(shape.back()*shape[i]);
    }
    std::copy(input.begin(), input.end(), node_values[0].begin());
    // Evaluate layers
    for (int l=0; l<shape.size()-2; ++l) {
        symmetrix_blas_gemv<Precision>(
            CblasRowMajor,            // const CBLAS_LAYOUT Layout
            CblasNoTrans,             // const CBLAS_TRANSPOSE trans
            shape[l+1],               // const MKL_INT m
            shape[l],                 // const MKL_INT n
            1.0,                      // const Precision alpha
            weights[l].data(),        // const Precision *a
            shape[l],                 // const MKL_INT lda
            node_values[l].data(),    // const Precision *x
            1,                        // const MKL_INT incx
            0.0,                      // const Precision beta
            node_values[l+1].data(),  // Precision *y
            1);                       // const MKL_INT incy
        for (int i=0; i<shape[l+1]; ++i) {
            const Precision x = node_values[l+1][i];
            const Precision sigmoid = 1.0/(1.0+std::exp(-x));
            node_values[l+1][i] = activation_scale_factor*x*sigmoid;
            node_activation_derivs[l+1][i] = activation_scale_factor*sigmoid
                + activation_scale_factor*x*sigmoid*(1-sigmoid);
        }
    }
    // Evaluate final layer (no nonlinearity)
    symmetrix_blas_gemv<Precision>(
        CblasRowMajor,                 // const CBLAS_LAYOUT Layout
        CblasNoTrans,                  // const CBLAS_TRANSPOSE trans
        shape.end()[-1],               // const MKL_INT m
        shape.end()[-2],               // const MKL_INT n
        1.0,                           // const Precision alpha
        weights.back().data(),         // const Precision *a
        shape.end()[-2],               // const MKL_INT lda
        node_values.end()[-2].data(),  // const Precision *x
        1,                             // const MKL_INT incx
        0.0,                           // const Precision beta
        node_values.end()[-1].data(),  // Precision *y
        1);                            // const MKL_INT incy
    // Differentiate backwards
    node_derivs.end()[-2] = weights.back();
    for (int l=shape.size()-3; l>=0; --l) {
        for (int i=0; i<shape.back(); ++i) {
            for (int j=0; j<shape[l+1]; ++j) {
                node_derivs[l+1][i*shape[l+1]+j] *= node_activation_derivs[l+1][j];
            }
        }
        symmetrix_blas_gemm<Precision>(
            CblasRowMajor,            // const CBLAS_LAYOUT Layout
            CblasNoTrans,             // const CBLAS_TRANSPOSE transa
            CblasNoTrans,             // const CBLAS_TRANSPOSE transb
            shape.back(),             // const MKL_INT m
            shape[l],                 // const MKL_INT n
            shape[l+1],               // const MKL_INT k
            1.0,                      // const Precision alpha
            node_derivs[l+1].data(),  // const Precision *a
            shape[l+1],               // const MKL_INT lda
            weights[l].data(),        // const Precision *b
            shape[l],                 // const MKL_INT ldb
            0.0,                      // const Precision beta
            node_derivs[l].data(),    // Precision *c
            shape[l]);                // const MKL_INT ldc
    }
    return {node_values.back(), node_derivs[0]};
}

template <typename Precision>
auto MultilayerPerceptronT<Precision>::evaluate_gradient_directional(
    std::vector<Precision> input,
    std::vector<Precision> input_dot)
    -> std::tuple<std::vector<Precision>,std::vector<Precision>,std::vector<Precision>>
{
    auto values = std::vector<std::vector<Precision>>(shape.size());
    auto value_dots = std::vector<std::vector<Precision>>(shape.size());
    auto pre_activation_dots = std::vector<std::vector<Precision>>(shape.size());
    auto activation_derivs = std::vector<std::vector<Precision>>(shape.size());
    auto activation_second_derivs = std::vector<std::vector<Precision>>(shape.size());
    for (int l=0; l<shape.size(); ++l) {
        values[l].resize(shape[l], 0.0);
        value_dots[l].resize(shape[l], 0.0);
        pre_activation_dots[l].resize(shape[l], 0.0);
        activation_derivs[l].resize(shape[l], 0.0);
        activation_second_derivs[l].resize(shape[l], 0.0);
    }

    std::copy(input.begin(), input.end(), values[0].begin());
    std::copy(input_dot.begin(), input_dot.end(), value_dots[0].begin());

    for (int l=0; l<shape.size()-2; ++l) {
        for (int j=0; j<shape[l+1]; ++j) {
            Precision z = 0.0;
            Precision z_dot = 0.0;
            for (int i=0; i<shape[l]; ++i) {
                const Precision weight = weights[l][j*shape[l]+i];
                z += weight*values[l][i];
                z_dot += weight*value_dots[l][i];
            }
            const Precision sigmoid = 1.0/(1.0+std::exp(-z));
            const Precision sigmoid_deriv = sigmoid*(1.0-sigmoid);
            const Precision sigmoid_second_deriv = sigmoid_deriv*(1.0-2.0*sigmoid);
            values[l+1][j] = activation_scale_factor*z*sigmoid;
            value_dots[l+1][j] =
                activation_scale_factor*(sigmoid + z*sigmoid_deriv)*z_dot;
            pre_activation_dots[l+1][j] = z_dot;
            activation_derivs[l+1][j] =
                activation_scale_factor*(sigmoid + z*sigmoid_deriv);
            activation_second_derivs[l+1][j] =
                activation_scale_factor*(2.0*sigmoid_deriv + z*sigmoid_second_deriv);
        }
    }

    const int output_size = shape.back();
    const int final_input_size = shape.end()[-2];
    auto output = std::vector<Precision>(output_size, 0.0);
    for (int o=0; o<output_size; ++o) {
        for (int i=0; i<final_input_size; ++i)
            output[o] += weights.back()[o*final_input_size+i] * values.end()[-2][i];
    }

    auto jac_next = weights.back();
    auto jac_next_dot = std::vector<Precision>(jac_next.size(), 0.0);
    for (int l=shape.size()-3; l>=0; --l) {
        auto jac_z = std::vector<Precision>(output_size*shape[l+1], 0.0);
        auto jac_z_dot = std::vector<Precision>(output_size*shape[l+1], 0.0);
        for (int o=0; o<output_size; ++o) {
            for (int j=0; j<shape[l+1]; ++j) {
                const int index = o*shape[l+1]+j;
                jac_z[index] = jac_next[index]*activation_derivs[l+1][j];
                jac_z_dot[index] =
                    jac_next_dot[index]*activation_derivs[l+1][j]
                    + jac_next[index]
                    * activation_second_derivs[l+1][j]
                    * pre_activation_dots[l+1][j];
            }
        }

        auto jac_prev = std::vector<Precision>(output_size*shape[l], 0.0);
        auto jac_prev_dot = std::vector<Precision>(output_size*shape[l], 0.0);
        for (int o=0; o<output_size; ++o) {
            for (int i=0; i<shape[l]; ++i) {
                for (int j=0; j<shape[l+1]; ++j) {
                    const Precision weight = weights[l][j*shape[l]+i];
                    jac_prev[o*shape[l]+i] += jac_z[o*shape[l+1]+j] * weight;
                    jac_prev_dot[o*shape[l]+i] += jac_z_dot[o*shape[l+1]+j] * weight;
                }
            }
        }
        jac_next = std::move(jac_prev);
        jac_next_dot = std::move(jac_prev_dot);
    }

    return {output, jac_next, jac_next_dot};
}

template <typename Precision>
auto MultilayerPerceptronT<Precision>::evaluate_batch(
    std::vector<Precision> input,
    const int batch_size
    )-> std::vector<Precision>
{
    const auto penultimate = evaluate_penultimate_batch(std::move(input), batch_size);
    node_values.back().resize(batch_size*shape.back());
    // Evaluate final layer (no nonlinearity)
    symmetrix_blas_gemm<Precision>(
        CblasRowMajor,                 // const CBLAS_LAYOUT Layout
        CblasNoTrans,                  // const CBLAS_TRANSPOSE transa
        CblasTrans,                    // const CBLAS_TRANSPOSE transb
        batch_size,                    // const MKL_INT m
        shape.end()[-1],               // const MKL_INT n
        shape.end()[-2],               // const MKL_INT k
        1.0,                           // const Precision alpha
        penultimate.data(),            // const Precision *a
        shape.end()[-2],               // const MKL_INT lda
        weights.end()[-1].data(),      // const Precision *b
        shape.end()[-2],               // const MKL_INT ldb
        0.0,                           // const Precision beta
        node_values.end()[-1].data(),  // Precision *c
        shape.end()[-1]);              // const MKL_INT ldc
    return node_values.end()[-1];
}

template <typename Precision>
auto MultilayerPerceptronT<Precision>::evaluate_penultimate_batch(
    std::vector<Precision> input,
    const int batch_size)
    -> std::vector<Precision>
{
    if (shape.size() < 2 || weights.size()+1 != shape.size())
        throw std::invalid_argument("MLP has an invalid shape.");
    if (batch_size < 0
        || input.size() != static_cast<std::size_t>(batch_size)*shape.front())
        throw std::invalid_argument("MLP batch input has an invalid size.");

    for (int layer=0; layer<shape.size()-1; ++layer)
        node_values[layer].resize(static_cast<std::size_t>(batch_size)*shape[layer]);
    std::copy(input.begin(), input.end(), node_values.front().begin());
    for (int layer=0; layer<shape.size()-2; ++layer) {
        symmetrix_blas_gemm<Precision>(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            batch_size,
            shape[layer+1],
            shape[layer],
            1.0,
            node_values[layer].data(),
            shape[layer],
            weights[layer].data(),
            shape[layer],
            0.0,
            node_values[layer+1].data(),
            shape[layer+1]);
        for (auto& value : node_values[layer+1])
            value = activation_scale_factor*value/(1.0+std::exp(-value));
    }
    return node_values.end()[-2];
}

template <typename Precision>
auto MultilayerPerceptronT<Precision>::evaluate_gradient_batch(
    std::vector<Precision> input,
    const int batch_size
    )-> std::tuple<std::vector<Precision>,std::vector<Precision>>
{
    // TODO: Check/sanitize input
    // Reshape node arrays and send input to nodes
    for (int i=0; i<shape.size(); ++i) {
        node_values[i].resize(batch_size*shape[i]);
        node_activation_derivs[i].resize(batch_size*shape[i]);
        node_derivs[i].resize(batch_size*shape.back()*shape[i]);
    }
    std::copy(input.begin(), input.end(), node_values[0].begin());
    // Evaluate layers
    for (int l=0; l<shape.size()-2; ++l) {
        symmetrix_blas_gemm<Precision>(
            CblasRowMajor,            // const CBLAS_LAYOUT Layout
            CblasNoTrans,             // const CBLAS_TRANSPOSE transa
            CblasTrans,               // const CBLAS_TRANSPOSE transb
            batch_size,               // const MKL_INT m
            shape[l+1],               // const MKL_INT n
            shape[l],                 // const MKL_INT k
            1.0,                      // const Precision alpha
            node_values[l].data(),    // const Precision *a
            shape[l],                 // const MKL_INT lda
            weights[l].data(),        // const Precision *b
            shape[l],                 // const MKL_INT ldb
            0.0,                      // const Precision beta
            node_values[l+1].data(),  // Precision *c
            shape[l+1]);              // const MKL_INT ldc
        for (int i=0; i<node_values[l+1].size(); ++i) {
            const Precision x = node_values[l+1][i];
            const Precision sigmoid = 1.0/(1.0+std::exp(-x));
            node_values[l+1][i] = activation_scale_factor*x*sigmoid;
            node_activation_derivs[l+1][i] = activation_scale_factor*sigmoid
                + activation_scale_factor*x*sigmoid*(1-sigmoid);
        }
    }
    // Evaluate final layer (no nonlinearity)
    symmetrix_blas_gemm<Precision>(
        CblasRowMajor,                 // const CBLAS_LAYOUT Layout
        CblasNoTrans,                  // const CBLAS_TRANSPOSE transa
        CblasTrans,                    // const CBLAS_TRANSPOSE transb
        batch_size,                    // const MKL_INT m
        shape.end()[-1],               // const MKL_INT n
        shape.end()[-2],               // const MKL_INT k
        1.0,                           // const Precision alpha
        node_values.end()[-2].data(),  // const Precision *a
        shape.end()[-2],               // const MKL_INT lda
        weights.end()[-1].data(),      // const Precision *b
        shape.end()[-2],               // const MKL_INT ldb
        0.0,                           // const Precision beta
        node_values.end()[-1].data(),  // Precision *c
        shape.end()[-1]);              // const MKL_INT ldc
    // Differentiate backwards
    for (int i=0; i<batch_size; ++i) {
        std::copy(weights.back().begin(), weights.back().end(), node_derivs.end()[-2].begin()+i*weights.back().size());
    }
    for (int l=shape.size()-3; l>=0; --l) {
        for (int i=0; i<batch_size; ++i) {
            for (int j=0; j<shape.back(); ++j) {
                for (int p=0; p<shape[l+1]; ++p) {
                    node_derivs[l+1][i*shape.back()*shape[l+1]+j*shape[l+1]+p] *= node_activation_derivs[l+1][i*shape[l+1]+p];
                }
            }
        }
        symmetrix_blas_gemm<Precision>(
            CblasRowMajor,            // const CBLAS_LAYOUT Layout
            CblasNoTrans,             // const CBLAS_TRANSPOSE transa
            CblasNoTrans,             // const CBLAS_TRANSPOSE transb
            batch_size*shape.back(),  // const MKL_INT m
            shape[l],                 // const MKL_INT n
            shape[l+1],               // const MKL_INT k
            1.0,                      // const Precision alpha
            node_derivs[l+1].data(),  // const Precision *a
            shape[l+1],               // const MKL_INT lda
            weights[l].data(),        // const Precision *b
            shape[l],                 // const MKL_INT ldb
            0.0,                      // const Precision beta
            node_derivs[l].data(),    // Precision *c
            shape[l]);                // const MKL_INT ldc
    }
    return {node_values.back(), node_derivs[0]};
}

template class MultilayerPerceptronT<float>;
template class MultilayerPerceptronT<double>;
