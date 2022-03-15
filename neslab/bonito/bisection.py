def crit_abs(val, tol):
    """Absolute tolerance criterion."""
    return abs(val) < tol


def crit_gt(val, tol):
    """Greater than tolerance criterion."""
    return val > 0 and val < tol


def crit_lt(val, tol):
    """Less than criterion."""
    return val < 0 and -val < tol


def bisection(fn, a: float, b: float, tol: float = 1e-3, max_iter: int = 100, criterion=None):
    """Finds the root of a function in the given interval using the bisection method.

    Args:
        fn: target function
        a: lower interval bracket
        b: upper interval bracket
        tol: required tolerance for convergence
        max_iter: maximum number of bisection iterations
        criterion: convergence criterion function

    Returns:
        Number of iterations and solution
    """
    if criterion is None:
        criterion = crit_abs

    fa = fn(a)
    fb = fn(b)

    if abs(fb) < tol:
        return 0, b
    if abs(fa) < tol:
        return 0, a

    if a > b:
        raise ValueError("a must be less than b")

    if fa * fb >= 0:
        raise ValueError("f(a) and f(b) must have different signs")

    for it in range(1, max_iter):
        c = (a + b) / 2
        fc = fn(c)

        if criterion(fc, tol):
            return it, c
        if fc * fa >= 0:
            a = c
        else:
            b = c
    raise RuntimeError("bisection did not converge")
