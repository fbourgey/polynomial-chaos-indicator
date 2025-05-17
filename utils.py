import numpy as np
from numpy.polynomial.hermite_e import hermeval
from numpy.polynomial.legendre import legval
from scipy import special, stats


def gamma_hermite(n, x):
    "PCE coefficient for the Hermite polynomials."
    # x = np.asarray(x)
    if n == 0:
        return stats.norm.cdf(-x)
    return stats.norm.pdf(x) * special.hermitenorm(n - 1)(x) / special.factorial(n)


def gamma_laguerre(n, x, alpha):
    "PCE coefficient for the Laguerre polynomials."
    x = np.asarray(x)
    if n == 0:
        return special.gammaincc(alpha + 1, x)
    return (
        -(special.factorial(n - 1) / special.gamma(n + alpha + 1))
        * x ** (alpha + 1)
        * np.exp(-x)
        * special.eval_genlaguerre(n - 1, alpha + 1, x)
    )


def gamma_jacobi(n, x, alpha, beta):
    "PCE coefficient for the Jacobi polynomials."
    if n == 0:
        return special.betainc(alpha + 1, beta + 1, 0.5 * (1 - x))
    tmp1 = (
        2 ** (alpha + beta + 1)
        * special.gamma(n + alpha + 1)
        * special.gamma(n + beta + 1)
    )
    tmp2 = (
        (2 * n + alpha + beta + 1)
        * special.gamma(n + alpha + beta + 1)
        * special.factorial(n)
    )
    return (
        (1 / (2 * n * (tmp1 / tmp2)))
        * (1 - x) ** (alpha + 1)
        * (1 + x) ** (beta + 1)
        * special.eval_jacobi(n - 1, alpha + 1, beta + 1, x)
    )


def gamma_legendre(n, x):
    "PCE coefficient for the Legendre polynomials."
    return gamma_jacobi(n, x, 0, 0)


def l2_error(c, n_pce, n_mc, alpha=0.5, beta=0.5, poly="legendre", seed=None):
    """
    Compute the L2 error between an indicator function and its Legendre polynomial chaos
    expansion.

    This function estimates the root mean squared error (RMSE) between an indicator
    function (1 if c <= x, 0 otherwise)
    and its polynomial chaos expansion using Legendre polynomials, over a Monte
    Carlo sample of points in [-1, 1].
    It also computes a 95% confidence interval for the error estimate.

    Parameters
    ----------
    c : float
        The threshold value for the indicator function.
    n_pce : int
        The order of the polynomial chaos expansion.
    n_mc : int
        The number of Monte Carlo samples to use for the error estimation.

    Returns
    -------
    rmse : float
        The estimated root mean squared error between the indicator and its chaos
        expansion.
    ci : float
        The 95% confidence interval for the RMSE estimate.
    """
    if poly not in ["legendre", "hermite", "laguerre", "jacobi"]:
        raise ValueError(
            "poly must be one of 'legendre', 'hermite', 'laguerre', or 'jacobi'."
        )

    # c = np.atleast_1d(c)
    # if c.shape[0] != 1:
    #     raise ValueError("c must be a scalar.")

    if not (isinstance(n_pce, int | np.integer) and n_pce >= 0):
        raise ValueError("n_pce must be a non-negative integer.")

    if not (isinstance(n_mc, int) and n_mc > 0):
        raise ValueError("n_mc must be a positive integer.")

    if seed is not None:
        np.random.seed(seed)

    if poly == "legendre":
        x = np.random.uniform(-1, 1, n_mc)
        gam = np.array([gamma_legendre(n, c) for n in range(n_pce + 1)])
        indicator_pce = legval(x, c=gam, tensor=False)
    if poly == "hermite":
        x = np.random.randn(n_mc)
        gam = np.array([gamma_hermite(n, c) for n in range(n_pce + 1)])
        indicator_pce = hermeval(x, c=gam, tensor=False)
    if poly == "laguerre":
        x = np.random.gamma(shape=alpha + 1, scale=1, size=n_mc)
        indicator_pce = gamma_laguerre(0, c, alpha) * 1.0
        for n in range(1, n_pce + 1):
            indicator_pce += gamma_laguerre(n, c, alpha) * special.eval_genlaguerre(
                n, alpha, x
            )
    if poly == "jacobi":
        x = 1.0 - 2.0 * np.random.beta(a=alpha + 1.0, b=beta + 1.0, size=n_mc)
        indicator_pce = gamma_jacobi(0, c, alpha, beta) * 1.0
        for n in range(1, n_pce + 1):
            indicator_pce += gamma_jacobi(n, c, alpha, beta) * special.eval_jacobi(
                n, alpha, beta, x
            )
    indicator = 1.0 * (c <= x)
    rmse = np.mean((indicator - indicator_pce) ** 2) ** 0.5
    confidence_interval = (
        stats.norm.ppf(0.975) * np.std((indicator - indicator_pce) ** 2) / n_mc**0.5
    )
    return rmse, confidence_interval
