### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

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
Using Herbie, one can find the most computationally efficient & precise way of computing these calculations:

1) $ sqrt()

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
This resembles the case of a geometric series:

```math

\sum_{k=0}^\infty r^k = \frac{1}{1-r} \qquad \text{if } |r|<1
```
With $r= \frac12$, except that here the first index is $1$ and $2$ indices of $k$ are skipped every $4$ iterations.

```math
\begin{align}
\sum_{k=1}^\infty r^{4k-1} +r^{4k}= \lim_{k \to \infty} & r^3+r^4+r^7+r^8+...+r^{4k-1}+r^{4k} \\

(1-r^4)\sum_{k=1}^\infty r^{4k-1} +r^{4k}= &(1-r^4)  \lim_{k \to \infty} (r^3+r^4+r^7+r^8+...+r^{4k-1}+r^{4k}) \\
= &\lim_{k \to \infty}  r^3+r^4+r^7+r^8+...+r^{4k-1}+r^{4k}-r^7-r^8-...-r^{4k}-r^{4k+1} \\
= &\lim_{k \to \infty} r^3+r^4 -r^{4k}-r^{4k+1}

\end{align}
```
With the terms to the power $k$ vanishing as $r=\frac12$, the result obtained for this troncated geometric series is:

```math
\sum_{k=1}^\infty r^{4k-1} +r^{4k}=\frac{r^3+r^4}{1-r^4}=\frac{0.125+0.0625}{1-0.0625}=0.2
```

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

# ╔═╡ 9364f526-5149-4730-b0c7-fa619cfa598c
md"""
**Exercise 3 Answer**

In this exercise the goal is to replicate the max() function from Julia, there are a couple of edge cases that will pose problem to our naive Max(a,b) that should be checked and corrected:

1) if there is a NaN: "Not a Number" should never be a max nor a min, since it is not comparable to a number in $\mathbb{R}$, the convention chosen by Julia is to return "NaN".

2) if $a,b$ $\pm0$: when dealing with limits, the sign from which one is approaching $0$ matters, therefore we should check that if the sign is mentioned it should be taken into account.
"""

# ╔═╡ 549d5975-a3cc-44b1-95c4-cdbf148f21ba
begin 

	function Max(a, b)
		if a > b
			return a
		else
			return b
		end
	end
	#for completeness, a Min() function has been built to compare the results
	
	function Min(a, b)
		if a < b
			return a
		else
			return b
		end
	end
	
end

# ╔═╡ bbf1a1ee-889d-43f2-8c82-5c5d20f3fedf

begin
	#the naive functions do not take into consideration the sign of the 0 
	println(Max(-0.0,0.0))
	println(Min(-0.0,0.0),"\n")

	#they also only return NaN if both a and b are NaN's
	println(Max(NaN,Inf))
	println(Min(NaN,Inf))
	println(Max(NaN,NaN))
end

# ╔═╡ 606c3200-95e2-4d77-b34b-b6b47daedd8d
#here is the corrected function to circumvent the issues 

function Max_corrected(a,b)
	if a==b
		if a==-0.0 || b==-0.0
				return 0.0
		end
	return a
	end
	if isnan(a) || isnan(b)
		return NaN
	end
	if a>b
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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
IntervalArithmetic = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[compat]
BFloat16s = "~0.4.2"
IntervalArithmetic = "~0.21.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "de202fd6de5ac930b4e5aae858c31bb15c836feb"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "dbf84058d0a8cbbadee18d25cf606934b22d7c66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.4.2"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "32abd86e3c2025db5172aa182b982debed519834"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.1"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ErrorfreeArithmetic]]
git-tree-sha1 = "d6863c556f1142a061532e79f611aa46be201686"
uuid = "90fa49ef-747e-5e6f-a989-263ba693cf1a"
version = "0.5.2"

[[deps.FastRounding]]
deps = ["ErrorfreeArithmetic", "LinearAlgebra"]
git-tree-sha1 = "6344aa18f654196be82e62816935225b3b9abe44"
uuid = "fa42c844-2597-5d31-933b-ebd51ab2693f"
version = "0.3.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "EnumX", "FastRounding", "LinearAlgebra", "Markdown", "Random", "RecipesBase", "RoundingEmulator", "SetRounding", "StaticArrays"]
git-tree-sha1 = "d70eb5999afad9c180b6aa7947260c1b66163f8a"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.21.1"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SetRounding]]
git-tree-sha1 = "d7a25e439d07a17b7cdf97eecee504c50fedf5f6"
uuid = "3cc68bcd-71a2-5612-b932-767ffbe40ab0"
version = "0.2.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "51621cca8651d9e334a659443a74ce50a3b6dfab"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.3"

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─7eda14a6-02c2-4eb2-b774-dccea23f5632
# ╟─73097e41-590f-4975-9781-19351dfbfed5
# ╟─64892b7a-5ae5-11ee-3683-214c5333532e
# ╠═1c5d1cc9-13e2-46ec-ad2c-8cca52011bc4
# ╠═feaad6a4-3d7a-403a-8a3b-6135285d364b
# ╠═081476f4-351f-4611-a653-67a46b0c2624
# ╟─e8e1fd4b-7776-4533-af2e-279be0319a85
# ╠═9364f526-5149-4730-b0c7-fa619cfa598c
# ╠═549d5975-a3cc-44b1-95c4-cdbf148f21ba
# ╠═bbf1a1ee-889d-43f2-8c82-5c5d20f3fedf
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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
