
#!/usr/bin/env python3
"""
BLG202E - HW2 Q4
Least squares fitting for the given dataset:
  - Linear model:      l(x) = a + b x
  - Quadratic model:   p(x) = c x^2 + d x + e
  - Logarithmic model: f(x) = k + m ln(x)

Prints the coefficients and saves a single plot showing data points and all three fits.
"""
import math
import numpy as np
import matplotlib.pyplot as plt

# Given dataset (20 points)
xs = np.array([0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,
               5.5,6,6.5,7,7.5,8,8.5,9,9.5,10], dtype=float)
ys = np.array([0.72,1.63,1.88,3.39,4.02,3.89,4.25,3.99,4.68,5.03,
               5.27,4.82,5.67,5.95,5.72,6.01,5.5,6.41,5.83,6.33], dtype=float)

def fit_linear(x, y):
    """Return (a, b) minimizing ||a + b x - y||_2 via normal equations."""
    A = np.column_stack([np.ones_like(x), x])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeffs
    return float(a), float(b)

def fit_quadratic(x, y):
    """Return (c, d, e) minimizing ||c x^2 + d x + e - y||_2 via normal equations."""
    A = np.column_stack([x**2, x, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    c, d, e = coeffs
    return float(c), float(d), float(e)

def fit_log(x, y):
    """Return (k, m) minimizing ||k + m ln(x) - y||_2. Requires x > 0."""
    if np.any(x <= 0):
        raise ValueError("Log model requires all x > 0.")
    z = np.log(x)
    A = np.column_stack([np.ones_like(z), z])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    k, m = coeffs
    return float(k), float(m)

def evaluate_models(xgrid, lin, quad, logm):
    a, b = lin
    c, d, e = quad
    k, m = logm
    y_lin = a + b * xgrid
    y_quad = c * xgrid**2 + d * xgrid + e
    y_log = k + m * np.log(xgrid)
    return y_lin, y_quad, y_log

def main():
    # Fit
    lin = fit_linear(xs, ys)
    quad = fit_quadratic(xs, ys)
    logm = fit_log(xs, ys)
    print("Linear    l(x) = a + b x      -> a = %.6f, b = %.6f" % lin)
    print("Quadratic p(x) = c x^2 + d x + e -> c = %.6f, d = %.6f, e = %.6f" % quad)
    print("Logarithm f(x) = k + m ln(x) -> k = %.6f, m = %.6f" % logm)

    # Plot on a dense grid
    xgrid = np.linspace(xs.min(), xs.max(), 400)
    y_lin, y_quad, y_log = evaluate_models(xgrid, lin, quad, logm)

    # Single chart containing data and all fits (no seaborn, no specified colors/styles)
    plt.figure()
    plt.scatter(xs, ys, label="data")
    plt.plot(xgrid, y_lin, label="linear fit")
    plt.plot(xgrid, y_quad, label="quadratic fit")
    plt.plot(xgrid, y_log, label="log fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("HW2 Q4: Least Squares Fits (data + models)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/mnt/data/hw2_q4_fit.png", dpi=160)
    # Do not call plt.show() in this environment

if __name__ == "__main__":
    main()
