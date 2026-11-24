# %% [markdown]
# # Module 03: Exercises (Matrices as Transformations)
# 💻 = code, ✍️ = by hand first, then type the answer. Solutions are in `solutions.ipynb`.

# %%
#! SETUP

# %% [markdown]
# ### Exercise 1 ✍️: Multiply by hand
# $A = \begin{pmatrix}1 & 2\\3 & 4\end{pmatrix}$, $B = \begin{pmatrix}0 & 1\\1 & 0\end{pmatrix}$, $\mathbf{x} = (1, -1)$.
# 1. $A\mathbf{x}$
# 2. $AB$ (what did multiplying by B on the right do to A's columns?)
# 3. $BA$ (what did multiplying by B on the left do to A's rows?)
# 4. $A^\top$

# %%
### BEGIN SOLUTION
#> m1 = m2 = m3 = m4 = None  # TODO (use nested lists for matrices)
m1 = [-1, -1]
m2 = [[2, 1], [4, 3]]       # columns swapped
m3 = [[3, 4], [1, 2]]       # rows swapped
m4 = [[1, 3], [2, 4]]
### END SOLUTION

# %%
A1, B1 = np.array([[1, 2], [3, 4]]), np.array([[0, 1], [1, 0]])
check(m1, A1 @ np.array([1, -1]), name="A x")
check(m2, A1 @ B1, name="A B")
check(m3, B1 @ A1, name="B A")
check(m4, A1.T, name="Aᵀ")

# %% [markdown]
# ### Exercise 2 ✍️: Build the matrix from where e₁ and e₂ land
# Write the 2×2 matrix for each transformation. Think about where $\mathbf{e}_1$ and $\mathbf{e}_2$ go, and make those
# the columns.
# 1. Rotate 90° counter-clockwise.
# 2. Stretch x by 3 and leave y unchanged.
# 3. Reflect across the x-axis.
# 4. First stretch x by 2, **then** rotate 90° counter-clockwise (one combined matrix).

# %%
### BEGIN SOLUTION
#> r1 = r2 = r3 = r4 = None  # TODO
r1 = [[0, -1], [1, 0]]      # e1 -> (0, 1), e2 -> (-1, 0)
r2 = [[3, 0], [0, 1]]
r3 = [[1, 0], [0, -1]]
r4 = [[0, -1], [2, 0]]      # R @ S: e1 -> S -> (2, 0) -> R -> (0, 2); e2 -> (0, 1) -> (-1, 0)
### END SOLUTION

# %%
R90 = np.array([[0, -1], [1, 0]])
check(r1, R90, name="rotate 90°")
check(r2, np.diag([3, 1]), name="stretch x")
check(r3, np.diag([1, -1]), name="reflect")
check(r4, R90 @ np.diag([2, 1]), name="stretch, then rotate")

# %% [markdown]
# ### Exercise 3 ✍️: Shape detective
# $A$ is 3×5, $B$ is 5×2, $\mathbf{x}$ has shape (2,), $X$ is 100×5.
# 1. The shape of $AB\mathbf{x}$ (as a tuple).
# 2. Is $BA$ defined? (`True` / `False`)
# 3. The shape of $A^\top A$.
# 4. The shape of $XA^\top$.
# 5. $A$ maps $\mathbb{R}^n \to \mathbb{R}^m$. What are (n, m)?

# %%
### BEGIN SOLUTION
#> sh1 = sh2 = sh3 = sh4 = sh5 = None  # TODO
sh1 = (3,)
sh2 = False         # (5×2)(3×5): the inner dims 2 and 3 don't match
sh3 = (5, 5)
sh4 = (100, 3)
sh5 = (5, 3)
### END SOLUTION

# %%
A3, B3, x3, X3 = np.ones((3, 5)), np.ones((5, 2)), np.ones(2), np.ones((100, 5))
check(list(sh1), list((A3 @ B3 @ x3).shape), name="shape of ABx")
check(sh2, False, name="BA defined?")
check(list(sh3), list((A3.T @ A3).shape), name="shape of AᵀA")
check(list(sh4), list((X3 @ A3.T).shape), name="shape of XAᵀ")
check(list(sh5), [5, 3], name="(n, m)")

# %% [markdown]
# ### Exercise 4 💻: Rotation matrices
# 1. `rotation(theta)` returns the 2×2 counter-clockwise rotation matrix for angle `theta` (radians).
# 2. Then confirm numerically (it's in the check cell) that rotations keep lengths, and that rotating by a then by b
#    equals rotating by a + b.

# %%
def rotation(theta):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])
    ### END SOLUTION


# %%
check(rotation(np.pi / 2) @ np.array([1.0, 0]), [0, 1], name="R(90°) e1")
vv = rng.normal(size=2)
check(np.linalg.norm(rotation(1.234) @ vv), np.linalg.norm(vv), name="length preserved")
check(rotation(0.3) @ rotation(0.5), rotation(0.8), name="R(a) R(b) = R(a+b)")

# %% [markdown]
# ### Exercise 5 💻: Matrix products from scratch, two views
# Don't use `@`, `np.dot` or `np.matmul` here.
# 1. `matvec_rows(A, x)`: the row view. Each output entry is `np.sum(A[i] * x)`.
# 2. `matvec_cols(A, x)`: the column view. The output is Σⱼ x[j] · A[:, j].
# 3. `matmul(A, B)`: build AB **column by column**, where column k of AB is A times column k of B (use `matvec_cols`).

# %%
def matvec_rows(A, x):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    return np.array([np.sum(A[i] * x) for i in range(A.shape[0])])
    ### END SOLUTION


def matvec_cols(A, x):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    out = np.zeros(A.shape[0])
    for j in range(A.shape[1]):
        out = out + x[j] * A[:, j]
    return out
    ### END SOLUTION


def matmul(A, B):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    return np.stack([matvec_cols(A, B[:, k]) for k in range(B.shape[1])], axis=1)
    ### END SOLUTION


# %%
A5, B5, x5 = rng.normal(size=(4, 3)), rng.normal(size=(3, 5)), rng.normal(size=3)
check(matvec_rows(A5, x5), A5 @ x5, name="row view")
check(matvec_cols(A5, x5), A5 @ x5, name="column view")
check(matmul(A5, B5), A5 @ B5, name="matmul")

# %% [markdown]
# ### Exercise 6 💻: A dense layer on a batch
# `X` is (batch, d_in), `W` is (d_out, d_in) (the PyTorch `nn.Linear` convention), and `b` is (d_out,).
# Return `ReLU(X Wᵀ + b)`, with shape (batch, d_out), without loops.

# %%
def dense_relu(X, W, b):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    return np.maximum(X @ W.T + b, 0)
    ### END SOLUTION


# %%
X6, W6, b6 = rng.normal(size=(32, 10)), rng.normal(size=(4, 10)), rng.normal(size=4)
out6 = dense_relu(X6, W6, b6)
check(list(out6.shape), [32, 4], name="output shape")
check(out6[7], np.maximum(W6 @ X6[7] + b6, 0), name="row 7 = ReLU(W x + b)")

# %% [markdown]
# ### Exercise 7 💻: Collapse a stack of linear layers
# Without activations, layers `W1` (applied first), `W2`, ..., `Wk` (applied last) equal one matrix. Write
# `collapse(weights)` that returns it. `weights` is a list in **application order**. Mind the multiplication order!

# %%
def collapse(weights):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    M = np.eye(weights[0].shape[1])
    for W_ in weights:
        M = W_ @ M          # each new layer multiplies on the LEFT
    return M
    ### END SOLUTION


# %%
Ws = [rng.normal(size=(6, 3)), rng.normal(size=(4, 6)), rng.normal(size=(2, 4))]
x7 = rng.normal(size=3)
check(collapse(Ws) @ x7, Ws[2] @ (Ws[1] @ (Ws[0] @ x7)), name="collapsed network")
check(list(collapse(Ws).shape), [2, 3], name="collapsed shape")

# %% [markdown]
# ### Exercise 8 ✍️: Which transformations lose information?
# For each matrix answer `True` if it squashes the plane onto a line or a point (so it can't be undone):
# 1. $\begin{pmatrix}2 & 4\\1 & 2\end{pmatrix}$  2. $\begin{pmatrix}1 & 1\\0 & 1\end{pmatrix}$
# 3. $\begin{pmatrix}0 & 0\\0 & 0\end{pmatrix}$  4. $\begin{pmatrix}3 & 0\\0 & 0.01\end{pmatrix}$
#
# Hint: are the two columns on the same line?

# %%
### BEGIN SOLUTION
#> q1 = q2 = q3 = q4 = None  # TODO
q1 = True       # second column = 2 × first
q2 = False      # a shear can be undone
q3 = True       # everything goes to the origin
q4 = False      # squashed a lot, but still 2-D
### END SOLUTION

# %%
for k, (ans, M) in enumerate(zip([q1, q2, q3, q4], [[[2, 4], [1, 2]], [[1, 1], [0, 1]], [[0, 0], [0, 0]],
                                                     [[3, 0], [0, 0.01]]]), 1):
    check(ans, bool(np.linalg.matrix_rank(np.array(M)) < 2), name=f"matrix {k}")
