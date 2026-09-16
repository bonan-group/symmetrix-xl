import numpy as np
import pytest

import symmetrix

kokkos = True
if kokkos:
    MultilayerPerceptron = symmetrix.MultilayerPerceptronKokkos
    if not symmetrix._kokkos_is_initialized():
        symmetrix._init_kokkos()
else:
    MultilayerPerceptron = symmetrix.MultilayerPerceptron


def test_evaluate():
    ### Batch size: 11, input dimension: 8, hidden layers: 1
    x = np.random.random([11, 8])
    shape = [8, 32, 1]
    w0 = np.random.random([8, 32]).T
    w1 = np.random.random([32, 1]).T
    weights = [w0.flatten(), w1.flatten()]
    scale = 0.9
    # compute result
    mlp = MultilayerPerceptron(shape, weights, scale)
    f1 = np.zeros(x.shape[0])
    mlp.evaluate(x, f1)

    # compute reference result
    def act(x):
        return scale * x / (1 + np.exp(-x))

    def MLP(x):
        return w1.dot(act(w0.dot(x)))

    f2 = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        f2[i] = MLP(x[i, :]).item()
    assert f1 == pytest.approx(f2)

    ### Batch size: 32, input dimension: 5, hidden layers: 3
    x = np.random.random([32, 5])
    shape = [5, 8, 8, 4, 1]
    w0 = np.random.random([5, 8]).T
    w1 = np.random.random([8, 8]).T
    w2 = np.random.random([8, 4]).T
    w3 = np.random.random([4, 1]).T
    weights = [w0.flatten(), w1.flatten(), w2.flatten(), w3.flatten()]
    scale = 1.3
    # compute result
    mlp = MultilayerPerceptron(shape, weights, scale)
    f1 = np.zeros(x.shape[0])
    mlp.evaluate(x, f1)

    # compute reference result
    def act(x):
        return scale * x / (1 + np.exp(-x))

    def MLP(x):
        return w3.dot(act(w2.dot(act(w1.dot(act(w0.dot(x)))))))

    f2 = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        f2[i] = MLP(x[i, :]).item()
    assert f1 == pytest.approx(f2)


def test_evaluate_gradient_batch():
    ### Batch size: 11, input dimension: 8, hidden layers: 1
    x = np.random.random([11, 8])
    shape = [8, 32, 1]
    w0 = np.random.random([8, 32]).T
    w1 = np.random.random([32, 1]).T
    weights = [w0.flatten(), w1.flatten()]
    scale = 0.9
    # compute result with gradient
    mlp = MultilayerPerceptron(shape, weights, scale)
    f1 = np.empty(x.shape[0])
    g1 = np.empty(x.shape)
    mlp.evaluate_gradient(x, f1, g1)

    # compute reference result with gradient
    def act(x):
        return scale * x / (1 + np.exp(-x))

    def MLP(x):
        return w1.dot(act(w0.dot(x)))

    f2 = np.empty(x.shape[0])
    g2 = np.empty(x.shape)
    for i in range(x.shape[0]):
        f2[i] = MLP(x[i, :]).item()
        for j in range(x.shape[1]):
            x[i, j] += 1e-3
            fp = MLP(x[i, :]).item()
            x[i, j] -= 2e-3
            fm = MLP(x[i, :]).item()
            x[i, j] += 1e-3
            g2[i, j] = (fp - fm) / 2e-3
    assert f1 == pytest.approx(f2)
    assert g1 == pytest.approx(g2)

    ### Batch size: 32, input dimension: 5, hidden layers: 3
    x = np.random.random([32, 5])
    shape = [5, 8, 8, 4, 1]
    w0 = np.random.random([5, 8]).T
    w1 = np.random.random([8, 8]).T
    w2 = np.random.random([8, 4]).T
    w3 = np.random.random([4, 1]).T
    weights = [w0.flatten(), w1.flatten(), w2.flatten(), w3.flatten()]
    scale = 1.3
    # compute result with gradient
    mlp = MultilayerPerceptron(shape, weights, scale)
    f1 = np.empty(x.shape[0])
    g1 = np.empty(x.shape)
    mlp.evaluate_gradient(x, f1, g1)

    # compute reference result with gradient
    def act(x):
        return scale * x / (1 + np.exp(-x))

    def MLP(x):
        return w3.dot(act(w2.dot(act(w1.dot(act(w0.dot(x)))))))

    f2 = np.empty(x.shape[0])
    g2 = np.empty(x.shape)
    for i in range(x.shape[0]):
        f2[i] = MLP(x[i, :]).item()
        for j in range(x.shape[1]):
            x[i, j] += 1e-3
            fp = MLP(x[i, :]).item()
            x[i, j] -= 2e-3
            fm = MLP(x[i, :]).item()
            x[i, j] += 1e-3
            g2[i, j] = (fp - fm) / 2e-3
    assert f1 == pytest.approx(f2)
    assert g1 == pytest.approx(g2)


def test_direct_linear_gradient_and_empty_batch():
    shape = [3, 1]
    weights = [np.asarray([0.25, -0.5, 1.75])]
    mlp = MultilayerPerceptron(shape, weights, 0.7)
    x = np.asarray([[1.0, 2.0, -1.0], [-0.25, 0.5, 2.0]])
    output = np.empty(len(x))
    gradient = np.empty_like(x)

    mlp.evaluate_gradient(x, output, gradient)

    np.testing.assert_allclose(output, x @ weights[0], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        gradient,
        np.broadcast_to(weights[0], x.shape),
        rtol=0.0,
        atol=1e-14,
    )

    empty_x = np.empty((0, shape[0]))
    mlp.evaluate(empty_x, np.empty(0))
    mlp.evaluate_gradient(empty_x, np.empty(0), np.empty_like(empty_x))


def test_workspace_reuse_across_grow_shrink_grow_batches():
    rng = np.random.default_rng(20260729)
    shape = [4, 7, 3, 1]
    matrices = [
        rng.normal(size=(shape[layer + 1], shape[layer]))
        for layer in range(len(shape) - 1)
    ]
    scale = 1.1
    mlp = MultilayerPerceptron(
        shape,
        [matrix.flatten() for matrix in matrices],
        scale,
    )

    def reference(values):
        result = values
        for layer, matrix in enumerate(matrices):
            result = result @ matrix.T
            if layer + 1 < len(matrices):
                result = scale * result / (1.0 + np.exp(-result))
        return result[:, 0]

    for batch_size in (3, 19, 5, 27, 2):
        x = rng.normal(size=(batch_size, shape[0]))
        forward = np.empty(batch_size)
        value = np.empty(batch_size)
        gradient = np.empty_like(x)
        mlp.evaluate(x, forward)
        mlp.evaluate_gradient(x, value, gradient)
        np.testing.assert_allclose(forward, reference(x), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(value, forward, rtol=0.0, atol=1e-13)

        epsilon = 1e-6
        for column in range(shape[0]):
            plus = x.copy()
            minus = x.copy()
            plus[:, column] += epsilon
            minus[:, column] -= epsilon
            finite_difference = (reference(plus) - reference(minus)) / (2 * epsilon)
            np.testing.assert_allclose(
                gradient[:, column],
                finite_difference,
                rtol=2e-7,
                atol=2e-8,
            )


@pytest.mark.parametrize(
    "shape,weights,match",
    [
        ([3], [], "scalar output"),
        ([3, 2], [np.zeros(6)], "scalar output"),
        ([3, 1], [], "weight counts"),
        ([3, 0, 1], [np.zeros(0), np.zeros(0)], "positive"),
        ([3, 2, 1], [np.zeros(5), np.zeros(2)], "weight extent"),
    ],
)
def test_invalid_shapes_are_rejected(shape, weights, match):
    with pytest.raises(ValueError, match=match):
        MultilayerPerceptron(shape, weights, 1.0)
