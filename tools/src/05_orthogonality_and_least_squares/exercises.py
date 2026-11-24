# %% [markdown]
# # Module 05: Exercises (Orthogonality and Least Squares)
# 💻 = code, ✍️ = by hand first, then type the answer. Solutions are in `solutions.ipynb`.

# %%
#! SETUP

# %% [markdown]
# ### Exercise 1 ✍️: A projection matrix
# 1. Write $P = \frac{\mathbf{a}\mathbf{a}^\top}{\mathbf{a}^\top\mathbf{a}}$ for $\mathbf{a} = (1, 1)$.
# 2. Compute $P\,(3, 1)$.
# 3. Compute $P\,(1, -1)$. Why is that the answer?

# %%
### BEGIN SOLUTION
#> p1 = p2 = p3 = None  # TODO
p1 = [[0.5, 0.5], [0.5, 0.5]]
p2 = [2, 2]
p3 = [0, 0]       # (1, -1) is orthogonal to (1, 1), so its shadow is the zero vector
### END SOLUTION

# %%
a1 = np.array([1.0, 1.0]); P1 = np.outer(a1, a1) / (a1 @ a1)
check(p1, P1, name="P"); check(p2, P1 @ [3, 1], name="P (3,1)"); check(p3, P1 @ [1, -1], name="P (1,−1)")

# %% [markdown]
# ### Exercise 2 ✍️: Least squares by hand
# Fit $y = w_0 + w_1 x$ to the three points $(0, 1), (1, 3), (2, 4)$.
# 1. Write the design matrix X (3×2, with a column of ones) and compute $X^\top X$.
# 2. Compute $X^\top\mathbf{y}$.
# 3. Solve the 2×2 normal equations for $\mathbf{w} = (w_0, w_1)$. Fractions are fine, e.g. `7/6`.
# 4. Compute the residuals $\mathbf{y} - X\mathbf{w}$ and check on paper that they sum to 0.

# %%
### BEGIN SOLUTION
#> xtx = xty = w_ls = resid = None  # TODO
xtx = [[3, 3], [3, 5]]
xty = [8, 11]
w_ls = [7 / 6, 3 / 2]           # det = 6, inverse = (1/6)[[5, -3], [-3, 3]]
resid = [-1 / 6, 1 / 3, -1 / 6]
### END SOLUTION

# %%
X2 = np.array([[1, 0], [1, 1], [1, 2]], dtype=float); y2 = np.array([1.0, 3, 4])
check(xtx, X2.T @ X2, name="XᵀX"); check(xty, X2.T @ y2, name="Xᵀy")
w_ref = np.linalg.lstsq(X2, y2, rcond=None)[0]
check(w_ls, w_ref, name="w"); check(resid, y2 - X2 @ w_ref, name="residuals")

# %% [markdown]
# ### Exercise 3 ✍️: Orthogonal matrices
# $Q = \frac{1}{\sqrt2}\begin{pmatrix}1 & 1\\1 & -1\end{pmatrix}$.
# 1. Is Q orthogonal? (`True` / `False`)
# 2. What is $Q^{-1}$? (Hint: $Q^{-1} = Q^\top$. Then look closely at $Q^\top$.)
# 3. If $\lVert\mathbf{x}\rVert = 7$, what is $\lVert Q\mathbf{x}\rVert$?

# %%
### BEGIN SOLUTION
#> o1 = o2 = o3 = None  # TODO
o1 = True
o2 = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])    # Q is symmetric, so Q⁻¹ = Qᵀ = Q
o3 = 7
### END SOLUTION

# %%
Q3 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
check(o1, bool(np.allclose(Q3.T @ Q3, np.eye(2))), name="orthogonal?")
check(o2, np.linalg.inv(Q3), name="Q⁻¹"); check(o3, 7, name="‖Qx‖")

# %% [markdown]
# ### Exercise 4 💻: Projection onto a column space
# `projection_matrix(A)` returns $P = A(A^\top A)^{-1}A^\top$. Use `np.linalg.solve` rather than `inv`: solve
# $(A^\top A)Z = A^\top$ for Z, then $P = AZ$.

# %%
def projection_matrix(A):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    return A @ np.linalg.solve(A.T @ A, A.T)
    ### END SOLUTION


# %%
A4 = rng.normal(size=(8, 3)); P4 = projection_matrix(A4)
check(P4 @ P4, P4, name="P² = P"); check(P4, P4.T, name="P = Pᵀ")
check(P4 @ A4, A4, name="P leaves the columns of A unchanged")
check(np.trace(P4), 3, name="trace(P) = rank = 3")

# %% [markdown]
# ### Exercise 5 💻: QR via Gram–Schmidt
# Write `qr_gram_schmidt(A)` that returns `(Q, R)`, where Q (m×n) has orthonormal columns, R (n×n) is
# upper-triangular, and A = QR. Hint: run Gram–Schmidt to get Q, then $R = Q^\top A$.

# %%
def qr_gram_schmidt(A):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    m, n = A.shape
    Q = np.zeros((m, n))
    for j in range(n):
        v = A[:, j].astype(float)
        for i in range(j):
            v = v - (Q[:, i] @ v) * Q[:, i]     # "modified" Gram-Schmidt: project the running v (more stable)
        Q[:, j] = v / np.linalg.norm(v)
    return Q, Q.T @ A
    ### END SOLUTION


# %%
A5 = rng.normal(size=(7, 4)); Q5, R5 = qr_gram_schmidt(A5)
check(Q5.T @ Q5, np.eye(4), name="QᵀQ = I"); check(Q5 @ R5, A5, name="A = QR")
check(np.tril(R5, -1), np.zeros((4, 4)), tol=1e-9, name="R upper-triangular")

# %% [markdown]
# ### Exercise 6 💻: A least-squares solver built on QR
# `lstsq_qr(X, y)` uses `np.linalg.qr(X)`, then solves $R\mathbf{w} = Q^\top\mathbf{y}$ with
# `scipy.linalg.solve_triangular`. Never form $X^\top X$.

# %%
from scipy.linalg import solve_triangular


def lstsq_qr(X, y):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    Q, R = np.linalg.qr(X)
    return solve_triangular(R, Q.T @ y)
    ### END SOLUTION


# %%
X6 = rng.normal(size=(100, 5)); y6 = X6 @ np.arange(1.0, 6) + rng.normal(scale=0.1, size=100)
check(lstsq_qr(X6, y6), np.linalg.lstsq(X6, y6, rcond=None)[0], name="QR least squares")

# %% [markdown]
# ### Exercise 7 💻: Polynomial regression from scratch
# 1. `design_matrix(x, deg)` returns the columns $1, x, x^2, \dots, x^{\text{deg}}$, with shape (n, deg + 1).
# 2. `fit_poly(x, y, deg)` returns the weights (lowest power first) using your `lstsq_qr`.
#
# The check compares with `np.polyfit`. Note that it returns the **highest** power first.

# %%
def design_matrix(x, deg):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    return x[:, None] ** np.arange(deg + 1)[None, :]     # broadcasting: (n, 1) ** (1, deg + 1)
    ### END SOLUTION


def fit_poly(x, y, deg):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    return lstsq_qr(design_matrix(x, deg), y)
    ### END SOLUTION


# %%
x7 = np.linspace(-1, 1, 60); y7 = 1 - 2 * x7 + 3 * x7 ** 3 + rng.normal(scale=0.05, size=60)
check(design_matrix(np.array([2.0, 3.0]), 2), [[1, 2, 4], [1, 3, 9]], name="design matrix")
check(fit_poly(x7, y7, 3), np.polyfit(x7, y7, 3)[::-1], name="polynomial weights")
plt.scatter(x7, y7, s=10); plt.plot(x7, design_matrix(x7, 3) @ fit_poly(x7, y7, 3), "r"); plt.show()

# %% [markdown]
# ### Exercise 8 ✍️: Think geometrically
# 1. If b is already in the column space of A, what is the least-squares error $\lVert\mathbf{b} - A\hat{\mathbf{x}}\rVert$?
# 2. A is 3×3 and invertible. What is the projection matrix onto C(A)? (Give the 3×3 matrix.)
# 3. A least-squares fit **with** an intercept column: what do the residuals sum to?
# 4. $\kappa(X) = 10^5$. Roughly what is $\kappa(X^\top X)$?

# %%
### BEGIN SOLUTION
#> g1 = g2 = g3 = g4 = None  # TODO
g1 = 0
g2 = np.eye(3).tolist()   # C(A) is all of R³, so projecting does nothing
g3 = 0
g4 = 1e10
### END SOLUTION

# %%
check(g1, 0, name="error"); check(g2, projection_matrix(rng.normal(size=(3, 3))), tol=1e-6, name="P for invertible A")
check(g3, 0, name="sum of residuals"); check(g4, 1e10, tol=1e-3, name="κ(XᵀX)")
