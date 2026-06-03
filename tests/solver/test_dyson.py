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
