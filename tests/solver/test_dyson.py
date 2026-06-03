import numpy as np

from sigml.solver.dyson import is_causal, positive_matsubara_mask, sigma_from_g


def test_dyson_atomic_limit_high_freq():
    n = np.arange(64)
    iw = 1j * np.pi / 70.0 * (2 * n + 1)
    U, mu, eps_d = 4.0, 2.0, 0.0
    g = 0.5 / (iw + mu - eps_d) + 0.5 / (iw + mu - eps_d - U)
    sigma = sigma_from_g(g, np.zeros_like(iw), mu=mu, eps_d=eps_d, iw=iw)
    assert abs(sigma[-1].real - U / 2) < 0.1
    assert is_causal(sigma, iw)


def test_positive_mask():
    iw = 1j * np.array([-3.0, -1.0, 1.0, 3.0])
    assert positive_matsubara_mask(iw).tolist() == [False, False, True, True]


def test_matrix_dyson_diagonal_reduces_to_scalar_result():
    iw = 1j * np.pi / 70.0 * (2 * np.arange(8) + 1)
    mu = 1.7
    eps = np.diag([0.1, -0.2, 0.4])
    delta_diag = np.stack(
        [
            -0.2j / (iw.imag + 1.0),
            -0.1j / (iw.imag + 1.5),
            -0.3j / (iw.imag + 2.0),
        ],
        axis=-1,
    )
    sigma_diag = np.stack(
        [
            0.5 - 0.03j / (iw.imag + 1.0),
            0.7 - 0.02j / (iw.imag + 1.0),
            0.9 - 0.01j / (iw.imag + 1.0),
        ],
        axis=-1,
    )
    delta = np.zeros((iw.size, 3, 3), dtype=complex)
    sigma_expected = np.zeros_like(delta)
    g = np.zeros_like(delta)
    for f in range(iw.size):
        delta[f] = np.diag(delta_diag[f])
        sigma_expected[f] = np.diag(sigma_diag[f])
        g_inv = iw[f] * np.eye(3) + mu * np.eye(3) - eps - delta[f] - sigma_expected[f]
        g[f] = np.linalg.inv(g_inv)

    sigma = sigma_from_g(g, delta, mu=mu, eps_d=eps, iw=iw)

    np.testing.assert_allclose(sigma, sigma_expected, atol=1e-12)
    for orbital in range(3):
        scalar = sigma_from_g(
            g[:, orbital, orbital],
            delta[:, orbital, orbital],
            mu=mu,
            eps_d=eps[orbital, orbital],
            iw=iw,
        )
        np.testing.assert_allclose(sigma[:, orbital, orbital], scalar, atol=1e-12)


def test_matrix_dyson_known_coupled_orbital_case():
    iw = 1j * np.pi / 70.0 * (2 * np.arange(6) + 1)
    mu = 0.3
    eps = np.array(
        [
            [0.1, 0.02, -0.01j],
            [0.02, -0.1, 0.03],
            [0.01j, 0.03, 0.2],
        ],
        dtype=complex,
    )
    sigma_expected = np.array(
        [
            [0.4 - 0.2j, 0.03 + 0.01j, 0.0],
            [0.03 + 0.01j, 0.5 - 0.25j, -0.02j],
            [0.0, -0.02j, 0.6 - 0.3j],
        ],
        dtype=complex,
    )
    delta = np.array([(-0.1j / (w.imag + 1.0)) * np.eye(3) for w in iw])
    g = np.array(
        [
            np.linalg.inv(w * np.eye(3) + mu * np.eye(3) - eps - d - sigma_expected)
            for w, d in zip(iw, delta, strict=True)
        ]
    )

    sigma = sigma_from_g(g, delta, mu=mu, eps_d=eps, iw=iw)

    np.testing.assert_allclose(sigma, np.broadcast_to(sigma_expected, sigma.shape), atol=1e-12)


def test_matrix_causality_uses_imaginary_parts_of_eigenvalues():
    iw = 1j * np.array([-3.0, -1.0, 1.0, 3.0])
    good = np.array(
        [
            np.diag([1.0j, 0.5j, 0.2j]),
            np.diag([0.5j, 0.4j, 0.3j]),
            np.diag([-0.1j, -0.2j, -0.3j]),
            np.diag([-0.2j, -0.3j, -0.4j]),
        ],
        dtype=complex,
    )
    bad = good.copy()
    bad[2] = np.diag([-0.1j, 0.01j, -0.3j])

    assert is_causal(good, iw)
    assert not is_causal(bad, iw)
