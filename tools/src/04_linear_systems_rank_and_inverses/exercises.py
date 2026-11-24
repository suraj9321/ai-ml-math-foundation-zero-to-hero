# %% [markdown]
# # Module 04: Exercises (Linear Systems, Rank and Inverses)
# 💻 = code, ✍️ = by hand first, then type the answer. Solutions are in `solutions.ipynb`.

# %%
#! SETUP

# %% [markdown]
# ### Exercise 1 ✍️: Elimination by hand
# Solve by elimination and back-substitution:
# $$x + 2y = 5, \qquad 3x + 4y = 11.$$
# Also give the multiplier you used: $R_2 \leftarrow R_2 - m R_1$. What is m?

# %%
### BEGIN SOLUTION
#> sol1 = None  # TODO: [x, y]
#> mult = None  # TODO: m
sol1 = [1, 2]
mult = 3          # R2 - 3 R1 gives -2y = -4
### END SOLUTION

# %%
check(sol1, np.linalg.solve([[1, 2], [3, 4]], [5, 11]), name="[x, y]")
check(mult, 3, name="multiplier")

# %% [markdown]
# ### Exercise 2 ✍️: Determinants
# 1. $\det\begin{pmatrix}3 & 1\\2 & 4\end{pmatrix}$
# 2. $\det\begin{pmatrix}2 & 4\\1 & 2\end{pmatrix}$
# 3. If $\det A = 2$ and $\det B = -3$, what is $\det(AB)$?
# 4. A is 2×2 with $\det A = 5$. What is $\det(2A)$? (Hint: each of the 2 dimensions gets stretched by 2.)
# 5. A rotation by 37°: what is its determinant?

# %%
### BEGIN SOLUTION
#> d1 = d2 = d3 = d4 = d5 = None  # TODO
d1 = 10
d2 = 0
d3 = -6
d4 = 20         # 2^2 * 5
d5 = 1
### END SOLUTION

# %%
check(d1, np.linalg.det([[3, 1], [2, 4]]), name="det 1")
check(d2, np.linalg.det([[2, 4], [1, 2]]), tol=1e-9, name="det 2")
check(d3, -6, name="det(AB)")
Aq = rng.normal(size=(2, 2)); Aq *= np.sqrt(5 / abs(np.linalg.det(Aq))); Aq[0] *= np.sign(np.linalg.det(Aq))
check(d4, np.linalg.det(2 * Aq), name="det(2A)")
th = np.radians(37)
check(d5, np.linalg.det([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]), name="det(rotation)")

# %% [markdown]
# ### Exercise 3 ✍️: A 2×2 inverse
# Use the 2×2 formula to invert $\begin{pmatrix}2 & 1\\5 & 3\end{pmatrix}$, then multiply your answer by the
# original matrix on paper to confirm you get I.

# %%
### BEGIN SOLUTION
#> inv1 = None  # TODO
inv1 = [[3, -1], [-5, 2]]     # det = 6 - 5 = 1
### END SOLUTION

# %%
check(inv1, np.linalg.inv([[2, 1], [5, 3]]), name="inverse")

# %% [markdown]
# ### Exercise 4 ✍️: Rank and null space
# 1. The rank of $\begin{pmatrix}1 & 2\\2 & 4\end{pmatrix}$.
# 2. The rank of $C = \begin{pmatrix}1 & 0 & 1\\0 & 1 & 1\\1 & 1 & 2\end{pmatrix}$. (Look for a column that is a
#    combination of the others.)
# 3. A non-zero vector $\mathbf{v}$ with $C\mathbf{v} = \mathbf{0}$.
# 4. The dimension of C's null space (use rank–nullity).

# %%
### BEGIN SOLUTION
#> k1 = k2 = k3 = k4 = None  # TODO
k1 = 1
k2 = 2          # col3 = col1 + col2
k3 = [1, 1, -1]
k4 = 1          # 3 columns - rank 2
### END SOLUTION

# %%
C = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 2]])
check(k1, np.linalg.matrix_rank([[1, 2], [2, 4]]), name="rank 1")
check(k2, np.linalg.matrix_rank(C), name="rank of C")
assert k3 is not None and np.linalg.norm(k3) > 0, "the null vector must be non-zero"
check(C @ np.array(k3), np.zeros(3), name="C v")
check(k4, 3 - np.linalg.matrix_rank(C), name="nullity")

# %% [markdown]
# ### Exercise 5 💻: Back substitution
# `U` is upper-triangular (n×n) with a non-zero diagonal. Solve $U\mathbf{x} = \mathbf{b}$, starting from the last row.

# %%
def back_substitute(U, b):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x
    ### END SOLUTION


# %%
U5 = np.triu(rng.normal(size=(6, 6))) + 3 * np.eye(6)
b5 = rng.normal(size=6)
check(back_substitute(U5, b5), np.linalg.solve(U5, b5), name="back substitution")

# %% [markdown]
# ### Exercise 6 💻: 2×2 inverse with a singularity check
# `inverse_2x2(M)` uses the ad − bc formula. If |det| < 1e-12, raise `ValueError("singular")`.

# %%
def inverse_2x2(M):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    (a, b_), (c, d) = M
    det = a * d - b_ * c
    if abs(det) < 1e-12:
        raise ValueError("singular")
    return np.array([[d, -b_], [-c, a]]) / det
    ### END SOLUTION


# %%
M6 = rng.normal(size=(2, 2))
check(inverse_2x2(M6) @ M6, np.eye(2), name="M⁻¹ M = I")
try:
    inverse_2x2(np.array([[1.0, 2.0], [2.0, 4.0]]))
    raise AssertionError("❌ should have raised ValueError for a singular matrix")
except ValueError:
    print("✅ singular matrix detected")

# %% [markdown]
# ### Exercise 7 💻: Find redundant features
# Walk through the columns of X from left to right. A column is **redundant** if adding it doesn't increase the rank of
# the columns kept so far. Return the list of redundant column indices. (Use `np.linalg.matrix_rank`.)

# %%
def redundant_columns(X):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    kept, redundant = [], []
    for j in range(X.shape[1]):
        if np.linalg.matrix_rank(X[:, kept + [j]]) > len(kept):
            kept.append(j)
        else:
            redundant.append(j)
    return redundant
    ### END SOLUTION


# %%
base = rng.normal(size=(200, 3))
X7 = np.column_stack([base[:, 0], base[:, 1], base[:, 0] - 2 * base[:, 1], base[:, 2], 3 * base[:, 2]])
check(redundant_columns(X7), [2, 4], name="redundant columns")

# %% [markdown]
# ### Exercise 8 💻: Taming a near-singular regression
# `X8` has two almost identical features. Solving the normal equations $X^\top X\mathbf{w} = X^\top\mathbf{y}$ is
# unstable.
# 1. `cond_xtx(X, lam)` returns the condition number of $X^\top X + \lambda I$.
# 2. `ridge_solve(X, y, lam)` solves $(X^\top X + \lambda I)\mathbf{w} = X^\top\mathbf{y}$ with `np.linalg.solve`.
#
# Then look at how much the condition number drops as λ grows.

# %%
def cond_xtx(X, lam):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    return np.linalg.cond(X.T @ X + lam * np.eye(X.shape[1]))
    ### END SOLUTION


def ridge_solve(X, y, lam):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    ### END SOLUTION


# %%
from sklearn.linear_model import Ridge
z8 = rng.normal(size=100)
X8 = np.column_stack([z8, z8 + 1e-4 * rng.normal(size=100)])
y8 = 3 * z8 + rng.normal(scale=0.1, size=100)
for lam in [0, 0.01, 1.0]:
    print(f"λ = {lam:<5} cond = {cond_xtx(X8, lam):.2e}   w = {ridge_solve(X8, y8, lam).round(3) if lam else '(unstable)'}")
check(ridge_solve(X8, y8, 1.0), Ridge(alpha=1.0, fit_intercept=False).fit(X8, y8).coef_, name="ridge weights")
assert cond_xtx(X8, 1.0) < cond_xtx(X8, 0) / 1e3, "regularization should cut the condition number a lot"
print("✅ conditioning improved")
