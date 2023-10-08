### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ f9e8cdb0-5239-11ee-0bdd-cf0684181c0b
md"""
Error control in scientific modelling (MATH 500, Herbst)
"""

# ╔═╡ 1a4ed4ac-e68a-485f-8ce1-8cd8210fc04a
md"""
# Sheet 3: Matrix eigenproblems (10 P)
*To be handed in via moodle by 12.10.2023*
"""

# ╔═╡ 64dbc4f4-c01c-4f84-b07a-d3c383f88fc7
md"""
## Exercise 1 (1 + 1.5 + 0.5 P)

We saw in the lecture that if $A \in \mathbb{C}^{n\times n}$ is Hermitian (i.e. $A = A^H$, then the Rayleigh quotient
```math
R_A(x) = \frac{\langle x , A x\rangle}{\langle x, x \rangle}
```
is real for all $x \in \mathbb{C}^n$. The point of this exercise is to prove the converse, i.e.

> If Rayleigh quotient $R_A(x)$ is real for all $x \in \mathbb{C}^n$,
> then $A$ is a Hermitian matrix.

To show this we proceed as follows:

(a) Given an arbitrary matrix $S$, show that if $\langle x, S x \rangle = 0$ for all $x \in \mathbb{C}^n$, then we must have
```math
\langle y, Sz \rangle + \langle z, Sy \rangle = 0  \qquad \forall y, z \in \mathbb{C}^n
```
*Hint:* Expand $\langle (y+z), S(y+z) \rangle$.

(b) Use the result of (a) to show if $\langle x, Ax\rangle$
is real for all $x\in \mathbb{C}^n$, then $A$ must be Hermitian.
*Hint:* First relate $\langle x, A x\rangle$ to $\langle x, A^H x\rangle$, then use (a)

(c) Prove the above statement, i.e. if $R_A(x)$ is real for all $x \in \mathbb{C}^n$, then $A$ is Hermitian.
"""

# ╔═╡ 048e550d-11a5-49e9-bf86-400005ba5dbe
md"""
## Exercise 2 (1.5+1+0.5 P)

Recall that the Frobenius norm of a matrix $A \in \mathbb{C}^{n \times n}$ is given by
```math
\|A\|_F = \sqrt{\text{tr}(A^H A)} = \sqrt{\sum_{i=1}^n \sum_{j=1}^n |A_{ij}|^2}
```

(a) What is the Frobenius norm of a diagonal matrix? What is the $p$-norm of a diagonal matrix? Conclude whether the Frobenius norm can be associated to any vector $p$-norm?

(b) Show that
```math
\|A\|_2 \leq \|A\|_F
```
and use this to conclude
```math
R_A(x) \leq \|A\|_F. \qquad\qquad\qquad\text{($\ast$)}
```

(c) Keeping in mind our corrollary of Courant-Fisher, namely that
```math
\lambda_n = \max_{0\neq x\in\mathbb{C}} R_A(x)
```
as well as your result from (a) on diagonal matrices, argue why $(\ast)$ is a crude bound, in particular for large matrices.
"""

# ╔═╡ 94330639-53d6-4082-9b8f-3b2cee1c08a1
md"""
## Exercise 3 (2+1+1 P)

In the lectures we saw that minimising the Rayleigh quotient provides a numerical tool for computing eigenvalues via optimisation problems. But even beyond that setting the Rayleigh quotient can be interpreted as a tool to obtain an approximation for the eigenvalue corresponding to an approximate eigenvector. We will explore this in this exercise.

We consider the setting where we want to compute the eigenpair $(\lambda, v) \in \mathbb{R} \times \mathbb{C}^n$ of the Hermitian matrix $A \in \mathbb{C}^{n\times n}$. Employing some numerical scheme we do not get the exact eigenvector $v$ but only the approximation $u = v + \delta$, where $\delta\in\mathbb{C}^n$ is the error compared to $v$. As usual we take $v$ to be normalised. Moreover $\delta$ is always orthogonal to $v$ (Why?).

(a) To simplify our calculations in this exercise, we introduce
```math
t = \| \delta \| \in \mathbb{R} \quad d = \delta / t \in \mathbb{C}^n,
```
such that we can write $u = v + t d$. Note that both $v$ and $d$ are unit vectors. The Rayleigh quotient $R_A(u)$ provides an approximation to $\lambda$. Show that
```math
R_A(u) = \lambda + t^2 \left( \langle d, A d \rangle - \lambda \right) + O(t^4).
```

(b) Assume now a numerical scheme (e.g. power iteration) yields an approximation $u$ to the exact eigenvector $v$, which is accurate to a tolerance $\varepsilon$, i.e. $\|δ\| = \varepsilon$. We want to estimate the corresponding eigenvalue using $R_A(u)$. How does the error between this estimate and the true eigenvalue $\lambda$ scale with $\varepsilon$?

(c) Reconsider your power iteration implementation from Sheet 1. Extend it, such that it employs the iterated eigenvector $x^{(i)}$ as well as the Rayleigh quotient to estimate the eigenvalue at each step. For the procedure computing the largest eigenvalue of the matrix
```math
A = \left(
\begin{array}{ccc}
30\,000 & -10\,000 & 10\,000 \\
-10\,000 & -30\,000 & 0 \\
10\,000 & 0 & 1
\end{array}
\right)
```
record both the approximate eigenvalue as well as the approximate eigenvector in each iteration in two separate arrays. Use this data to plot the error in the approximate eigenvalue as well as the error in the eigenvector as the iteration proceeds. You should find numerical confirmation to your analysis of (a) and (b).

*Some hints:*
- For computing the eigenvalue error just take the modulus of the absolute error, for the eigenvector error take the $l_2$-norm (`norm` function in Julia).
- You can compute the exact eigenvalue and eigenvector using the `eigen` routine of Julia. Note, however, that (real) eigenvectors are only determined up to the sign, so you have to ensure that the same sign convention is used in `eigen` as well as your own algorithm. The easiest is to determine the sign of the first element of the vector returned by your routine as well as `eigen` and multiply one of the vectors by `-1` in case these differ.
- Usually it is best to employ a `log`-scale on the $y$-axis for such error plots.
"""

# ╔═╡ d03ece6d-10cc-46c7-a11b-13f7ec50daad
# Your code goes here

# ╔═╡ Cell order:
# ╟─f9e8cdb0-5239-11ee-0bdd-cf0684181c0b
# ╟─1a4ed4ac-e68a-485f-8ce1-8cd8210fc04a
# ╟─64dbc4f4-c01c-4f84-b07a-d3c383f88fc7
# ╟─048e550d-11a5-49e9-bf86-400005ba5dbe
# ╟─94330639-53d6-4082-9b8f-3b2cee1c08a1
# ╠═d03ece6d-10cc-46c7-a11b-13f7ec50daad
