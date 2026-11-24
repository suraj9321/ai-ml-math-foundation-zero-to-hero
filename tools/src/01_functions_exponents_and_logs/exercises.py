# %% [markdown]
# # Module 01: Exercises (Functions, Exponents and Logs)
# 💻 = code, ✍️ = by hand first, then type the answer. Solutions are in `solutions.ipynb`.

# %%
#! SETUP

# %% [markdown]
# ### Exercise 1 ✍️: Exponents and logs
# Compute by hand (ln is the natural log):
# 1. $\log_2 32$
# 2. $\log_{10} 0.001$
# 3. $\ln(e^4)$
# 4. $8^{2/3}$
# 5. $\log_{10} 2 + \log_{10} 5$
# 6. $\ln 1$

# %%
### BEGIN SOLUTION
#> e1 = e2 = e3 = e4 = e5 = e6 = None  # TODO: one value each
e1 = 5          # 2^5 = 32
e2 = -3         # 10^-3 = 0.001
e3 = 4
e4 = 4          # (8^(1/3))^2 = 2^2
e5 = 1          # log10(2 * 5) = log10(10)
e6 = 0
### END SOLUTION

# %%
check(e1, np.log2(32), name="log2 32")
check(e2, np.log10(0.001), name="log10 0.001")
check(e3, 4.0, name="ln e^4")
check(e4, 8 ** (2 / 3), name="8^(2/3)")
check(e5, np.log10(2) + np.log10(5), name="log 2 + log 5")
check(e6, np.log(1), name="ln 1")

# %% [markdown]
# ### Exercise 2 ✍️: Expand a log
# Using the log rules, expand $\ln\left(\dfrac{x^3 y}{z^2}\right)$ into a sum of simple logs.
# Type it as a SymPy expression, e.g. `2*sp.log(x) - sp.log(y)`.

# %%
x, y, z = sp.symbols("x y z", positive=True)
### BEGIN SOLUTION
#> expanded = None  # TODO
expanded = 3 * sp.log(x) + sp.log(y) - 2 * sp.log(z)
### END SOLUTION

# %%
check(expanded, sp.expand_log(sp.log(x**3 * y / z**2)), name="expanded log")
assert all(isinstance(t, (sp.log, sp.Mul)) for t in sp.Add.make_args(expanded)), "write it as a sum of simple logs"

# %% [markdown]
# ### Exercise 3 ✍️: Sigmoid facts
# 1. $\sigma(0)$
# 2. Given $\sigma(2) \approx 0.8808$, what is $\sigma(-2)$? (Use the symmetry $\sigma(-z) = 1 - \sigma(z)$.)
# 3. $\text{logit}(0.75) = \ln\frac{0.75}{0.25}$. Write it as `np.log(<a whole number>)`.
# 4. Which logit $z$ gives probability $p = 0.5$?

# %%
### BEGIN SOLUTION
#> s1 = s2 = s3 = s4 = None  # TODO
s1 = 0.5
s2 = 1 - 0.8808
s3 = np.log(3)
s4 = 0
### END SOLUTION

# %%
sig = lambda t: 1 / (1 + np.exp(-t))
check(s1, sig(0.0), name="σ(0)")
check(s2, sig(-2.0), tol=1e-3, name="σ(−2)")
check(s3, np.log(0.75 / 0.25), name="logit(0.75)")
check(s4, 0.0, name="z for p = 0.5")

# %% [markdown]
# ### Exercise 4 💻: Bernoulli log-likelihood
# For binary labels $y_i \in \{0, 1\}$ and a single probability p, the log-likelihood is
# $$\ell(p) = \sum_{i=1}^n \left[ y_i \log p + (1 - y_i)\log(1 - p) \right].$$
# 1. Implement `bernoulli_loglik(y, p)` (vectorized, no loops).
# 2. Implement `mle_on_grid(y)`: evaluate ℓ on `np.linspace(0.001, 0.999, 999)` and return the argmax p.
#
# The check compares your MLE with the sample mean of y. Module 18 proves they are equal.

# %%
def bernoulli_loglik(y, p):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    ### END SOLUTION


def mle_on_grid(y):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    ps = np.linspace(0.001, 0.999, 999)
    ll = np.array([bernoulli_loglik(y, p) for p in ps])
    return ps[np.argmax(ll)]
    ### END SOLUTION


# %%
y_obs = (rng.random(5000) < 0.37).astype(int)
check(bernoulli_loglik(np.array([1, 0, 1]), 0.5), 3 * np.log(0.5), name="loglik")
check(mle_on_grid(y_obs), y_obs.mean(), tol=2e-3, name="MLE p̂")

# %% [markdown]
# ### Exercise 5 💻: Stable batched softmax
# `Z` has shape (batch, K): one row of logits per example. Return the softmax of **each row**, shape (batch, K).
# It must not overflow for logits like 1000. Hint: `max(axis=1, keepdims=True)` keeps the shapes broadcastable.

# %%
def softmax_rows(Z):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)
    ### END SOLUTION


# %%
from scipy.special import softmax as sp_softmax
Z = rng.normal(size=(6, 4)) * 5
Z[0] += 1000                                   # one row of huge logits
P = softmax_rows(Z)
check(P, sp_softmax(Z, axis=1), name="softmax rows")
check(P.sum(axis=1), np.ones(6), name="rows sum to 1")

# %% [markdown]
# ### Exercise 6 💻: Cross-entropy from logits
# For logits `Z` (batch, K) and integer labels `y` (batch,), the mean cross-entropy is
# $$\text{CE} = -\frac{1}{n}\sum_{i=1}^n \log \text{softmax}(\mathbf{z}^{(i)})_{y_i}.$$
# 1. `log_softmax_rows(Z)`: $z_k - \text{logsumexp}(\mathbf{z})$ for each row, computed stably (don't take the log of
#    softmax).
# 2. `cross_entropy(Z, y)`: use your log-softmax and fancy indexing `L[np.arange(n), y]`.

# %%
def log_softmax_rows(Z):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    c = Z.max(axis=1, keepdims=True)
    lse = c + np.log(np.exp(Z - c).sum(axis=1, keepdims=True))
    return Z - lse
    ### END SOLUTION


def cross_entropy(Z, y):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    L = log_softmax_rows(Z)
    return -L[np.arange(len(y)), y].mean()
    ### END SOLUTION


# %%
from scipy.special import log_softmax as sp_log_softmax
yl = rng.integers(0, 4, size=6)
check(log_softmax_rows(Z), sp_log_softmax(Z, axis=1), name="log-softmax")
check(cross_entropy(Z, yl), -sp_log_softmax(Z, axis=1)[np.arange(6), yl].mean(), name="cross-entropy")
check(cross_entropy(np.zeros((3, 4)), np.array([0, 1, 2])), np.log(4), name="CE of uniform guess = log K")

# %% [markdown]
# ### Exercise 7 💻: Recover an exponential law
# The data follows $y = C e^{kx}$ with multiplicative noise. Taking logs makes it a straight line,
# $\ln y = \ln C + kx$. Return `(k, C)` using `np.polyfit` on the log-transformed data.

# %%
x_data = np.linspace(0, 4, 60)
y_data = 3.0 * np.exp(0.8 * x_data) * np.exp(rng.normal(0, 0.05, size=60))


def fit_exponential(x, y):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    k, log_c = np.polyfit(x, np.log(y), 1)
    return k, np.exp(log_c)
    ### END SOLUTION


# %%
k_hat, C_hat = fit_exponential(x_data, y_data)
check(k_hat, 0.8, tol=0.03, name="k")
check(C_hat, 3.0, tol=0.05, name="C")

# %% [markdown]
# ### Exercise 8 ✍️: Temperature limits
# For logits $\mathbf{z} = (1, 3, 2)$:
# 1. What does $\text{softmax}(\mathbf{z}/T)$ approach as $T \to 0^+$? (Give a list of 3 numbers.)
# 2. What does it approach as $T \to \infty$?
# 3. Does adding 100 to every logit change the softmax output? Answer `True` (changes) or `False`.

# %%
### BEGIN SOLUTION
#> t1 = t2 = t3 = None  # TODO
t1 = [0, 1, 0]                 # all the mass goes to the largest logit
t2 = [1 / 3, 1 / 3, 1 / 3]     # every logit/T approaches 0, so the result is uniform
t3 = False                     # the shift-invariance used by the stable softmax
### END SOLUTION

# %%
zt = np.array([1.0, 3.0, 2.0])
check(t1, sp_softmax(zt / 1e-3), tol=1e-6, name="T → 0")
check(t2, sp_softmax(zt / 1e6), tol=1e-5, name="T → ∞")
check(t3, bool(not np.allclose(sp_softmax(zt), sp_softmax(zt + 100))), name="shift changes softmax?")
