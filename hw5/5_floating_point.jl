### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 9c772d3c-b394-494c-a908-58533618113a
# ╠═╡ disabled = true
#=╠═╡
begin 
	import Pkg
	Pkg.add("Decimals")
	
	Pkg.add("DecFP")

end
  ╠═╡ =#

# ╔═╡ 2cb083cf-a8fc-40ab-b68e-7865c7bafb63
begin
	using LinearAlgebra
	using BFloat16s
	using IntervalArithmetic
end

# ╔═╡ 7eda14a6-02c2-4eb2-b774-dccea23f5632
md"""
Error control in scientific modelling (MATH 500, Herbst)
"""

# ╔═╡ 73097e41-590f-4975-9781-19351dfbfed5
md"""
# Sheet 5: Floating-point numbers (10 P)
*To be handed in via moodle by 27.10.2023*
"""

# ╔═╡ 64892b7a-5ae5-11ee-3683-214c5333532e
md"""
## Exercise 1 (2.5 P)

Rewrite the following expressions in order to avoid cancellation for the indicated relations of the arguments or between the arguments. Assume that the input arguments $x$, $y$ and $\theta$ are known exactly. For each improved expression explain shortly (e.g. in 1 bullet point) why it is improved.

1. For $x ≈ 0\ $ : $\ \sqrt{x + 1} - 1$
1. For $x ≈ y\ $ : $\ \sin(x) - \sin(y)$
1. For $x ≈ y\ $ : $\ x^2 - y^2$
1. For $x ≈ 0\ $ : $\ \frac{1 - \cos(x)}{\sin(x)}$
1. For $a ≈ b$ and $|θ| \ll 1\ $ : $\ c = \sqrt{a^2 + b^2 - 2ab \cos θ}$

*Hint:* Find mathematically equivalent expressions in which cancellation is harmless.

*Tip: You might try yourself or use a tool like Herbie (https://herbie.uwplse.org/demo/) to automatically search over expressions. In any case, include a short explanation additionally.*
"""

# ╔═╡ 1c5d1cc9-13e2-46ec-ad2c-8cca52011bc4
md"""

**Solution**


Using Herbie, one can find the most computationally efficient & precise way of computing these calculations:


"""

# ╔═╡ feaad6a4-3d7a-403a-8a3b-6135285d364b
md"""
## Exercise 2 (2.5 P)

Show that
```math
0.1 = \sum_{i=1}^\infty 2^{-4i} + 2^{-4i-1}
```
and deduce that $x = 0.1$ has the base $2$ representation $0.000\overline{1100}$ (i.e. the last $4$ bits periodically repeated). Let $\hat{x} = fl(0.1)$ the IEEE single-precision (`Float32`) version. Show that $\frac{x - \hat{x}}{x} = - \frac14 u$ where the single-precision unit roundoff is $u = 2^{-24}$.
"""

# ╔═╡ 081476f4-351f-4611-a653-67a46b0c2624
md"""
**Solution:**


This resembles the case of a geometric series:

```math

\sum_{k=0}^\infty r^k = \frac{1}{1-r} \qquad \text{if } |r|<1
```
With $r= \frac12$, except that here the first index is $1$ and $2$ indices of $k$ are skipped every $4$ iterations.

By taking the 2 terms of the series one by one: 


```math
\sum_{i=1}^\infty r^{4i}= \sum_{i=1}^\infty (\frac12^4)^i= \sum_{i=1}^\infty (\frac{1}{16})^i= -\tilde{r}^0+\sum_{i=0}^\infty \tilde{r}^i

```

With $\tilde{r}=\frac{1}{16}$, one can rewrite it using the geometric series' result:

```math
\sum_{i=1}^\infty r^{4i}= -1 + \frac{1}{1-\tilde{r}}=1-\frac{1}{1-\frac{1}{16}}=\frac{1}{15}
```

Doing the same thing with the other term $\sum_{i=1}^\infty r^{4i-1}$ one gets:

```math
\sum_{i=1}^\infty r^{4i-1}=\frac{1}{r}\sum_{i=1}^\infty r^{4i}

```
And as it has been shown that just above that $\sum_{i=1}^\infty r^{4i}=\frac{1}{15}$, one can rewrite it as:

```math 

\sum_{i=1}^\infty r^{4i-1}=\frac12 \frac{1}{15}=\frac{1}{30}

```
Summing both sums, the result obtained for the truncated series is:

```math
\sum_{i=1}^\infty 2^{-4i} + 2^{-4i-1}=\frac{1}{15}+\frac{1}{30}=\frac{3}{30}=0.1 
```

$\qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \square$

"""

# ╔═╡ 44a13521-29bd-43a3-95b7-f8bacb16f5aa
md"""

First thing one can do is to rewrite $\hat{x}$ as the sum demonstrated above and applying the float$32$, it can be decomposed into :

$\hat{x}=(-1)^{b_{31}} 2^{(E-127)}(1+\sum_{i=1}^{23} b_{23-i} 2^{-i})$

with $b_i$ reprensenting one of 32 bits. 

```math
\hat{x} = fl(0.1)=fl(\sum_{i=1}^\infty 2^{-4i} + 2^{-4i-1})=\sum_{i=1}^{31} 2^{-4i} + 2^{-4i-1}
```


To demonstrate the relative error, one of the ways is to show that  

```math
\frac{x - \hat{x}}{x} =\frac{\sum_{i=1}^\infty 2^{-4i} + 2^{-4i-1}}{}
```

"""

# ╔═╡ f29c6589-0707-4264-b7f5-eb6c97010e8d
begin
	println(BigFloat(0.1,precision=32))
	println(bitstring(Float32(0.1)))

	
	#println((BigFloat(0.1)-Float32(0.1))/BigFloat(0.1))

end

# ╔═╡ 354b70cf-2bda-4986-9db3-b5c5741ccbd0
md"""

We have the binary representation of float32(0.1), which can be calculated in decimal:

- the sign bit is $0$

- the exponent is represented by $01111011 = 1 +2 + 8 + 16 +32 + 64 =104+11 =123$

- the mantiss is $10011001100110011001101 = 0.7999997735$ in decimal

From there one can calculate the value in base 10 via the formula above:

$\hat{x}=(-1)^0 \times 2^{123-127} (1+0.7999997735)= 2^{-4} \times 1.7999997735 = 0.11249998584$ 



"""



# ╔═╡ e8e1fd4b-7776-4533-af2e-279be0319a85
md"""
## Exercise 3 (2.5 P)

Providing function implementations, that produce expected answers for all IEEE numbers is sometimes surprisingly tricky. Extra care is in particular needed to ensure that special cases (`±Inf`, `NaN`, `±0` etc.) behave as expected.

Consider the following naive implementation of the `max` function. Does this always produce the expected answer for IEEE arithmetic? If yes, why? If no, suggest an improved version.

```julia
function max(a, b)
	if a > b
		return a
	else
		return b
	end
end
```
"""

# ╔═╡ f8d063ad-b90f-436b-9bbd-85ff2f8dd1d6
md"""
**Solution:**


In this exercise the goal is to replicate the max() function from Julia, there are a couple of edge cases that will pose problems to our naive max function above that should be checked and corrected,

Let's begin by defining the naive Max(a,b) function: 

"""

# ╔═╡ 7465768c-2f08-41a7-9158-ef3e2b8df505
function Max(a, b)
	if a > b
		return a
	else
		return b
	end
end

# ╔═╡ 4388de8f-4425-49ce-83cb-d827785fa5e9
md"""
We also define a function for finding the minimum to compare the results afterwards.
"""

# ╔═╡ 63eca55f-f98f-4ad7-a0b6-21138a6c7df4
function Min(a, b)
	if a < b
		return a
	else
		return b
	end
end

# ╔═╡ 9364f526-5149-4730-b0c7-fa619cfa598c
md"""
Let's now go over the problematic cases and try to find a way to overcome the issues.

##### Not a Number (NaN)
"""

# ╔═╡ 00296b33-9bbe-4723-a3fd-355ce99a9484
md"""
Let's check what happens to our `Max()` function if at least one of the inputs is NaN.
"""

# ╔═╡ 80a7925d-6c1c-48c1-8763-76f29aa59047
Max(NaN, -1.)

# ╔═╡ 1abc04e5-423b-45f6-bae8-b5e19580cd34
Max(NaN, Inf)

# ╔═╡ 2da8d976-ed45-477d-a4e2-768376290de1
Max(Inf,NaN)

# ╔═╡ 72696b2d-3bda-411c-a9f6-28610f01bff7
Max(NaN,NaN)

# ╔═╡ 51a876a5-fb65-4a9c-a5f1-83c278a62487
md"""
The problem here is that NaN should never be a max nor a min, since it is not comparable to a number in $\mathbb{R}$. In this case, the convention chosen by Julia is to return NaN.
"""

# ╔═╡ 73e898e0-560f-4833-8dc8-093a70064c42
md"""
To circumvent the NaN case, we should add an additional check if there is NaN given as an input:

```julia
	if isnan(a) || isnan(b)
		return NaN
	end
```
"""

# ╔═╡ 0468e498-fdde-41f8-b202-6c152a3416dc
md"""
##### Max(a, b $\pm0$)
"""

# ╔═╡ a54db2e7-2088-4ed4-8263-8f278fcac55a
md"""
When dealing with limits, the sign from which one is approaching $0$ matters, therefore one should check that if the sign is mentioned it should be taken into account.
"""

# ╔═╡ bbf1a1ee-889d-43f2-8c82-5c5d20f3fedf

begin
	#the naive functions do not take into consideration the sign of the 0 
	println(Max(-0.0,0.0))
	println(Min(-0.0,0.0),"\n")
end

# ╔═╡ 55c4903b-63af-413e-8cfb-7653abe0da1d
md"""
For the case of approximating $0$ from both sides, it should return the unsigned $0.0$ as it is bigger than $-0.0$

In order to do that, one should check if both sides are considered as "equal" and then check if they differ by a sign, and if so the term without the minus sign should get returned:

```julia
if a == b
	if a == 0.0 || b == 0.0
			return 0.0
	end
end
```

Now that both cases have been resolved, one can write the Max_corrected() function, that will reproduce the implemented max() function from Julia:
"""

# ╔═╡ 606c3200-95e2-4d77-b34b-b6b47daedd8d
#here is the corrected function to circumvent the issues 

function Max_corrected(a,b)
	#NaN part 
	
	if isnan(a) || isnan(b)
		return NaN
	end
	

	# 0.0 ≠ -0.0 part
	if a == b
		if a == 0.0 || b == 0.0
				return 0.0
		end
		return a
	end

	#rest of the cases
	
	if a > b
		return a 
	else 
		return b 
	end
end

# ╔═╡ 98f09752-c95b-45c6-801b-954ebe2f7e92
#check that the function follows the Julia convention

println(Max_corrected(NaN,Inf) ," ",max(NaN,Inf)," ","OK")

# ╔═╡ 626a224c-8812-4e6b-b27d-6f0f96714e3f
#check for the case of the sign of 0

if Max_corrected(-0.0,0.0)==max(-0.0,0.0)
	println("OK")
end

# ╔═╡ 11298834-e667-4cb0-ae57-2d8004b05893
md"""
## Exercise 4 (2.5 + 0 P)

In this exercise we want to perform an iterative diagonalisation with $16$-bit floating point numbers. The underlying idea is to simulate the scenario when we want to perform the diagonalisation of our matrix on a specialised hardware, which can only perform 16-bit operations. 

We consider the usual power method implementation
"""

# ╔═╡ 38e4320c-0fe4-42da-82ce-51e20b129bba
function power_method(A, u=randn(eltype(A), size(A, 2));
                      tol=1e-6, maxiter=100, verbose=true)
	norm_Δu = NaN
	for i in 1:maxiter
		u_prev = u
		u = A * u
		u /= norm(u)
		norm_Δu = min(norm(u - u_prev), norm(-u - u_prev))
		norm_Δu < tol && break
		verbose && println("$i   $norm_Δu")
	end
	μ = dot(u, A, u)
	norm_Δu ≥ tol && verbose && @warn "Power not converged $norm_Δu"
	(; μ, u)
end

# ╔═╡ a2d92cf5-e999-4af2-9b00-80285006f7bd
md"""
And the example matrix
"""

# ╔═╡ 11a2fb22-6d0c-49a1-bd9a-8b27998c21f2
A = diagm([-10, 30, 0.2]) + 1e-3 * randn(3, 3)

# ╔═╡ cce824e3-a693-4a4d-9e51-eabdf27da8d7
md"""
(a) Run the power method using `Float16` and converge the procedure using `tol=1e-6`. Compute the error in the eigenvalue and eigenvector against a diagonalisation with `eigen`, which employs `Float64` precision. Compute the residual norm of the obtained eigenpair in `Float16` and `Float64` precision. What do you observe? Repeat the computation in `BFloat16`.
"""

# ╔═╡ f95f989b-25b7-400c-a321-0f4ff4c90bb7
# Example for casting an array to Float16 and BFloat.
let x = [1.1, 2.2, 3.3]
	Float16.(x), BFloat16.(x)
end

# ╔═╡ feecb363-1d58-4be6-96d1-19cc31335b70
# Your code and answer here

# ╔═╡ 026027ca-351e-48d8-bd1d-2e424cd664fa
md"""
(b) *(optional exercise)* We now move from $16$-bit to `Float32` as our working precision. Run the `power_method` on $A$ using `Float32` precision. By employing interval arithmetic in `Float32` precision and by tuning the `tol` parameter appropriately compute the eigenpair in a way that you can computationally prove that your eigenpair is exact to a residual norm below $10^{-5}$. What is the highest accuracy (smallest residual) you can provably achieve with `Float32`-only operations?

*Hint:* To cast a `Float32` array to an appropriate array of `Float32` intervals employ:
"""

# ╔═╡ 1fda2189-cfa4-4da8-8645-20bb22df0c54
let
	float32_vector = randn(Float32, 3)
	interval.(float32_vector)
end

# ╔═╡ Cell order:
# ╟─7eda14a6-02c2-4eb2-b774-dccea23f5632
# ╟─73097e41-590f-4975-9781-19351dfbfed5
# ╟─64892b7a-5ae5-11ee-3683-214c5333532e
# ╠═1c5d1cc9-13e2-46ec-ad2c-8cca52011bc4
# ╠═feaad6a4-3d7a-403a-8a3b-6135285d364b
# ╠═081476f4-351f-4611-a653-67a46b0c2624
# ╠═44a13521-29bd-43a3-95b7-f8bacb16f5aa
# ╠═9c772d3c-b394-494c-a908-58533618113a
# ╠═f29c6589-0707-4264-b7f5-eb6c97010e8d
# ╠═354b70cf-2bda-4986-9db3-b5c5741ccbd0
# ╟─1c5d1cc9-13e2-46ec-ad2c-8cca52011bc4
# ╟─feaad6a4-3d7a-403a-8a3b-6135285d364b
# ╟─081476f4-351f-4611-a653-67a46b0c2624
# ╟─44a13521-29bd-43a3-95b7-f8bacb16f5aa
# ╟─e8e1fd4b-7776-4533-af2e-279be0319a85
# ╟─f8d063ad-b90f-436b-9bbd-85ff2f8dd1d6
# ╠═7465768c-2f08-41a7-9158-ef3e2b8df505
# ╟─4388de8f-4425-49ce-83cb-d827785fa5e9
# ╠═63eca55f-f98f-4ad7-a0b6-21138a6c7df4
# ╟─9364f526-5149-4730-b0c7-fa619cfa598c
# ╟─00296b33-9bbe-4723-a3fd-355ce99a9484
# ╠═80a7925d-6c1c-48c1-8763-76f29aa59047
# ╠═1abc04e5-423b-45f6-bae8-b5e19580cd34
# ╠═2da8d976-ed45-477d-a4e2-768376290de1
# ╠═72696b2d-3bda-411c-a9f6-28610f01bff7
# ╟─51a876a5-fb65-4a9c-a5f1-83c278a62487
# ╟─73e898e0-560f-4833-8dc8-093a70064c42
# ╟─0468e498-fdde-41f8-b202-6c152a3416dc
# ╠═a54db2e7-2088-4ed4-8263-8f278fcac55a
# ╟─bbf1a1ee-889d-43f2-8c82-5c5d20f3fedf
# ╟─55c4903b-63af-413e-8cfb-7653abe0da1d
# ╠═606c3200-95e2-4d77-b34b-b6b47daedd8d
# ╠═98f09752-c95b-45c6-801b-954ebe2f7e92
# ╠═626a224c-8812-4e6b-b27d-6f0f96714e3f
# ╟─11298834-e667-4cb0-ae57-2d8004b05893
# ╠═2cb083cf-a8fc-40ab-b68e-7865c7bafb63
# ╠═38e4320c-0fe4-42da-82ce-51e20b129bba
# ╟─a2d92cf5-e999-4af2-9b00-80285006f7bd
# ╠═11a2fb22-6d0c-49a1-bd9a-8b27998c21f2
# ╟─cce824e3-a693-4a4d-9e51-eabdf27da8d7
# ╠═f95f989b-25b7-400c-a321-0f4ff4c90bb7
# ╠═feecb363-1d58-4be6-96d1-19cc31335b70
# ╟─026027ca-351e-48d8-bd1d-2e424cd664fa
# ╠═1fda2189-cfa4-4da8-8645-20bb22df0c54
