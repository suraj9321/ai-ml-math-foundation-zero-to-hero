# %% [markdown]
# # Module 00: Exercises (Math with Python and Notation)
# - 💻 **Code** exercises: replace `raise NotImplementedError` with your code, then run the check cell.
# - ✍️ **By hand** exercises: work on paper first, then type your answer in place of `None`. Don't use Python
#   to compute these.
#
# Solutions are in `solutions.ipynb`. Try first!

# %%
#! SETUP

# %% [markdown]
# ### Exercise 1 ✍️: Read the notation
# Let $\mathbf{x} = (3, -1, 4, 2)$ (math indexing starts at 1). Compute by hand:
# 1. $x_3$
# 2. $\sum_{i=1}^{4} x_i$
# 3. $\sum_{i=2}^{3} x_i^2$
# 4. $\prod_{i=1}^{4} x_i$
# 5. $\sum_{i=1}^{4} \mathbb{1}[x_i > 0]$

# %%
### BEGIN SOLUTION
#> a1 = None  # TODO
#> a2 = None
#> a3 = None
#> a4 = None
#> a5 = None
a1 = 4            # third component (1-based!)
a2 = 8            # 3 - 1 + 4 + 2
a3 = 17           # (-1)^2 + 4^2
a4 = -24          # 3 * -1 * 4 * 2
a5 = 3            # three positive entries
### END SOLUTION

# %%
xv = np.array([3, -1, 4, 2])
check(a1, xv[2], name="x_3")
check(a2, xv.sum(), name="Σ x_i")
check(a3, (xv[1:3] ** 2).sum(), name="Σ_{i=2..3} x_i²")
check(a4, xv.prod(), name="Π x_i")
check(a5, (xv > 0).sum(), name="Σ 1[x_i > 0]")

# %% [markdown]
# ### Exercise 2 ✍️: Data-matrix indexing
# $X = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$ holds 2 examples (rows) and 3 features (columns).
# 1. What is $x_2^{(1)}$ (feature 2 of example 1, both 1-based)?
# 2. What is $\sum_{i=1}^{2} x_3^{(i)}$?
# 3. Write $\mathbf{x}^{(2)}$ as a list.

# %%
### BEGIN SOLUTION
#> b1 = None  # TODO
#> b2 = None
#> b3 = None
b1 = 2
b2 = 9            # 3 + 6
b3 = [4, 5, 6]
### END SOLUTION

# %%
Xd = np.array([[1, 2, 3], [4, 5, 6]])
check(b1, Xd[0, 1], name="x_2^(1)")
check(b2, Xd[:, 2].sum(), name="Σ_i x_3^(i)")
check(b3, Xd[1], name="x^(2)")

# %% [markdown]
# ### Exercise 3 💻: Formulas to NumPy
# Implement each without Python loops (`x`, `w`, `y`, `y_hat` are 1-D arrays):
# - `variance(x)` $= \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$
# - `weighted_sum(w, x, b)` $= \sum_{j=1}^d w_j x_j + b$
# - `mae(y, y_hat)` $= \frac{1}{n}\sum_{i=1}^n \lvert y_i - \hat{y}_i \rvert$
# - `rmse(y, y_hat)` $= \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$

# %%
def variance(x):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    return np.mean((x - x.mean()) ** 2)
    ### END SOLUTION


def weighted_sum(w, x, b):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    return np.sum(w * x) + b     # same as w @ x + b
    ### END SOLUTION


def mae(y, y_hat):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    return np.mean(np.abs(y - y_hat))
    ### END SOLUTION


def rmse(y, y_hat):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    return np.sqrt(np.mean((y - y_hat) ** 2))
    ### END SOLUTION


# %%
data = rng.normal(5, 2, size=1000)
check(variance(data), np.var(data), name="variance")
check(weighted_sum(np.array([1.0, -2, 0.5]), np.array([4.0, 1, 2]), 3.0), 6.0, name="weighted_sum")
check(mae(np.array([1.0, 2, 3]), np.array([2.0, 2, 1])), 1.0, name="mae")
check(rmse(np.array([1.0, 2, 3]), np.array([2.0, 2, 1])), np.sqrt(5 / 3), name="rmse")

# %% [markdown]
# ### Exercise 4 💻: Sums over one axis
# `X` has shape (m, n): m examples, n features. Return:
# 1. `feature_means`: $\mu_j = \frac{1}{m}\sum_{i=1}^m x_j^{(i)}$, shape (n,)
# 2. `row_norms`: $r^{(i)} = \sqrt{\sum_{j=1}^n (x_j^{(i)})^2}$, shape (m,)
# 3. `total`: $\sum_{i=1}^m\sum_{j=1}^n x_j^{(i)}$, a scalar

# %%
def axis_sums(X):
    ### BEGIN SOLUTION
    #> raise NotImplementedError  # return feature_means, row_norms, total
    feature_means = X.mean(axis=0)                 # collapse the example axis
    row_norms = np.sqrt((X ** 2).sum(axis=1))      # collapse the feature axis
    total = X.sum()
    return feature_means, row_norms, total
    ### END SOLUTION


# %%
Xr = rng.normal(size=(50, 4))
fm, rn, tot = axis_sums(Xr)
check(fm, Xr.mean(axis=0), name="feature_means")
check(rn, np.linalg.norm(Xr, axis=1), name="row_norms")
check(tot, Xr.sum(), name="total")

# %% [markdown]
# ### Exercise 5 💻: argmin on a grid
# Find $x^* = \arg\min_{x \in [-5, 5]} f(x)$ for $f(x) = (x - 2)^2 + \sin(3x)$ by evaluating f on a grid of
# 10,001 points. Return `(x_star, f_min)`.

# %%
def grid_argmin():
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    xs = np.linspace(-5, 5, 10_001)
    fx = (xs - 2) ** 2 + np.sin(3 * xs)
    k = np.argmin(fx)
    return xs[k], fx[k]
    ### END SOLUTION


# %%
x_star, f_min = grid_argmin()
from scipy.optimize import minimize_scalar
ref = minimize_scalar(lambda t: (t - 2) ** 2 + np.sin(3 * t), bounds=(1, 3), method="bounded")
check(x_star, ref.x, tol=1e-3, name="x*")
check(f_min, ref.fun, tol=1e-3, name="f(x*)")

# %% [markdown]
# ### Exercise 6 ✍️ + SymPy: Closed-form sums
# 1. By hand, use $\sum_{i=1}^n i = \frac{n(n+1)}{2}$ to find $\sum_{i=1}^{100} i$.
# 2. By hand, use the sum rules to find $\sum_{i=1}^{10} (3i + 2)$. (Hint: split it, pull the constant out,
#    and remember $\sum_{i=1}^{10} 2 = 20$.)
# 3. Use SymPy to get a closed form for $\sum_{i=1}^n i^3$ as a SymPy expression in `n`.

# %%
i, n = sp.symbols("i n", integer=True, positive=True)
### BEGIN SOLUTION
#> c1 = None  # TODO (by hand)
#> c2 = None  # TODO (by hand)
#> c3 = None  # TODO (SymPy)
c1 = 5050
c2 = 185          # 3 * 55 + 20
c3 = sp.summation(i**3, (i, 1, n))
### END SOLUTION

# %%
check(c1, 5050, name="Σ_{1..100} i")
check(c2, sum(3 * k + 2 for k in range(1, 11)), name="Σ (3i + 2)")
check(c3, (n * (n + 1) / 2) ** 2, name="Σ i³")   # fun fact: it's the square of Σ i

# %% [markdown]
# ### Exercise 7 💻: Safe float comparison and a NaN hunt
# 1. `close_enough(a, b)` returns True when |a − b| ≤ atol + rtol·|b|, with `rtol=1e-9`, `atol=1e-12`
#    (this is the rule `np.isclose` uses). Don't call `np.isclose`.
# 2. `first_bad_index(arr)` returns the index of the first NaN **or** ±inf in `arr`, or -1 if there is none.

# %%
def close_enough(a, b, rtol=1e-9, atol=1e-12):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    return bool(abs(a - b) <= atol + rtol * abs(b))
    ### END SOLUTION


def first_bad_index(arr):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    bad = np.flatnonzero(~np.isfinite(arr))   # isfinite is False for nan, +inf and -inf
    return int(bad[0]) if bad.size else -1
    ### END SOLUTION


# %%
check(close_enough(0.1 + 0.2, 0.3), True, name="0.1+0.2 ≈ 0.3")
check(close_enough(1.0, 1.001), False, name="1.0 vs 1.001")
check(first_bad_index(np.array([1.0, 2.0, np.inf, np.nan])), 2, name="first bad index")
check(first_bad_index(np.array([1.0, 2.0])), -1, name="no bad values")

# %% [markdown]
# ### Exercise 8 ✍️: Spot the bug
# A colleague computes the probability of a dataset as `np.prod(p)` where `p` holds 5,000 per-example
# probabilities, all around 0.9. They get `0.0`.
# 1. Which trap is this? Answer with one of `"overflow"`, `"underflow"`, `"cancellation"`, `"nan"`.
# 2. Without running code, roughly what is $\log \prod p_i = \sum \log p_i$? Use $\ln 0.9 \approx -0.105$.

# %%
### BEGIN SOLUTION
#> d1 = None  # TODO
#> d2 = None  # TODO
d1 = "underflow"
d2 = -525         # 5000 * -0.105: a perfectly representable number
### END SOLUTION

# %%
check(d1, "underflow", name="trap")
check(d2, 5000 * np.log(0.9), tol=0.01, name="Σ log p")
