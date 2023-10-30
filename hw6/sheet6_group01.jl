### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ b6cf13cd-451e-49f3-b339-6e1ed55659f9
begin
	using MatrixDepot
	using LinearAlgebra
	using PlutoUI
	using Printf
	using LinearMaps


end

# ╔═╡ f669e56e-9c27-4025-9bce-79ee0b5f5981
md"Error control in scientific modelling (MATH 500, Herbst)"

# ╔═╡ 767f0942-e9a8-4260-b3f9-273ea35a0844
md"""# Sheet 6: Diagonalisation algorithms (10 P)
*To be handed in via moodle by 02.11.2023*
"""

# ╔═╡ ee7657bb-923e-4b74-a70f-e230da92e4c0
md"""
## Exercise 1 (1 + 1.5 + 1 + 0.5 + 1 P)

In this exercise we want to extend the `projected_subspace_iteration` approach from the lecture in order to numerically compute the eigenpairs closest to the eigenvalue $1$ of the matrix
"""

# ╔═╡ 8fd23d56-ec08-4446-87ec-4fffa206cf20
md"Size of the test matrix: $n =$ $(@bind n PlutoUI.Slider(5:5:100, default=10, show_value=true))" 

# ╔═╡ 3ed9cf5a-cda2-4e96-b6f0-5e09f0ee87b1
begin
	A = matrixdepot("poisson", n)
end

# ╔═╡ f8fedbaf-ec6d-4d00-bf40-3b2759cbf45f
#X=randn(eltype(A), size(A))
X=randn(eltype(A), size(A, 2), 2)

# ╔═╡ 9b80b038-16a5-418a-8dd2-ee6a3a2f1f00
AX=A*X

# ╔═╡ ba3c816e-b0d0-46f0-bcd4-e284af89e3ba
md"""
which is a sparse matrix resulting from solving a Poisson equation in 2 dimensions.
You can assume that the eigenvalue $1$ is twice degenerate, i.e. that two eigenvectors are associated with this eigenvalue.

**(a)** Employ the `projected_subspace_iteration` algorithm of the lectures to numerically compute the the eigenpairs closest to eigenvalue $1$.
*Hints:* Use spectral transformations; for sparse matrices the `\` operator works as expected.

**(b)** Modify your algorithm, such that you can use varying subspace sizes, but only test convergence in the eigenpairs closest to $1$. For example use more than two initial guess vectors in `X`, but only check the residual norms for those two eigenpairs corresponding to the eigenvalue $1$. Experiment with different subspace sizes between $2$ and $5$ and plot the observed residual norms wrt. iteration number. Which variant converges in the least number of iterations?

**(c)** Given that the computational time per iteration scales roughly linearly in the number of subspace vectors, what is the most economical configuration for this setting? Measure the runtime of your algorithm in this setting using Julia's `@time` macro. Take the average of multiple measurements to reduce the influence due to the interference of other processes on your computer. 
*Optional:* If you want automise this, take a look at the [BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl) Julia package.

**(d)** Modify the original `projected_subspace_iteration` a second time, but in a different way: Extend it, such that it employs in each iteration the optimal *dynamical* shift, just like in Rayleigh quotient interation (RQI). As an initial guess take random vectors and test your algorithm by running it a few times using `A`. Ensure that the resulting eigenpairs are indeed approximate eigenpairs of $A$.

**(e)** Similar to RQI the procedure of (d) converges quickly to an eigenpair, but as you probably saw it is hard to predict which. If we want to employ it for approximating the eigenvalues around $1$ we therefore need to already use an initial guess, which is very close to the corresponding eigenvectors we care about. The solution is to chain the algorithms of (a) and (d), i.e. to employ one step of your algorithm in (a) on a random initial guess (e.g. set `maxiter=1`) as the starting point for your procedure in (d). Code up this chained algorithm and time it a few times using the same settings as in (c), i.e. the same subspace size in particular. Is employing the dynamical shift worth it? If you increase $n$ using the Slider, does this change your assesment?
"""

# ╔═╡ add79a1d-1c0e-44aa-a73d-84c5bddd2ecc
md"""
** Solutions **


"""

# ╔═╡ 6cb1101b-9a4c-4a1d-bf74-e84ea4aa36d3
begin
	#ortho_qr(A) = Matrix(qr(A).Q)
	#println(typeof(A))
	#println(typeof(ortho_qr(A)))
	ortho_qr(A) = Matrix(qr(A).Q)
	σ = 0.4  # Our approximation to the eigenvalue of interest
	shifted  = A - σ * I        # Shift the matrix
	A_factorised = factorize(shifted)  # Factorise to obtain fast \ operation
	(; λ, eigenvalues) = projected_subspace_iteration(A=A_factorised)
end


# ╔═╡ d720c05f-2999-4b0f-8319-bbe68e8b0945
begin
	function projected_subspace_iteration(A; tol=1e-6, maxiter=100, verbose=true,
	                                      X=randn(eltype(A), size(A, 2), 2),
	                                      ortho=ortho_qr)
		T = real(eltype(A))
	
		eigenvalues    = Vector{T}[]
		residual_norms = Vector{T}[]
		λ = T[]
		for i in 1:maxiter
			X = ortho(X)
			println(Dims(X))
			println(Dims(A))
			AX = A * X
			λ, Y = eigen(X' * AX)  # Notice the change to subspace_iteration
			                       # This is the Rayleigh-Ritz step
			push!(eigenvalues, λ)
			
			residuals = AX * Y - X * Y * Diagonal(λ)
			norm_r = norm.(eachcol(residuals))
			push!(residual_norms, norm_r)
	
			verbose && @printf "%3i %8.4g %8.4g\n" i λ[end] norm_r[end]
			maximum(norm_r) < tol && break
			
			X = AX
		end
		
		(; λ, X, eigenvalues, residual_norms)
	end
end

# ╔═╡ 5a00f6cc-e684-4db2-8376-e69d73cee8a8
md"""
## Exercise 2 (1 + 1 + 1 + 1 + 1 P)

In this exercise we will discuss some error estimation strategies for orthogonal projection methods.

We follow the Rayleigh-Ritz procedure discussed in the lectures to estimate the eigenpairs of the Hermitian matrix $B \in \mathbb{C}^{N\times N}$ using an $m$-subspace $\mathcal{S}$. Solving the eigenproblem projected into this subspace yields the Ritz pairs $(\tilde{λ}_i, \tilde{y}_i) \in \mathbb{R} \times \mathbb{C}^\textcolor{red}{m}$, from which we can in turn compute the approximate eigenvectors $(\tilde{λ}_i, \tilde{x}_i) \in \mathbb{R} \times \mathbb{C}^\textcolor{red}{N}$.

For the computational part of this exercise we will employ the matrix
"""

# ╔═╡ d51a787b-0ac8-48de-849e-1b475572dc11
B = matrixdepot("poisson", 10)

# ╔═╡ d4df3a19-a04f-4b3c-b3ef-3bca92f5a8a7
md"""
which is a sparse matrix resulting from solving a Poisson equation in 2 dimensions as well as the subspace $\mathcal{S}$ spanned by the three vectors
"""

# ╔═╡ a8802b27-864d-4bf0-b2d0-4812f7f54f78
begin
	V = zeros(size(B, 2), 3)
	V[1:2:end, 2] .= 1.0
	V[1:3:end, 3] .= 1.0
	V[1:4:end, 1] .= 1.0

	V = Matrix(qr(V).Q)
end

# ╔═╡ 44974d4c-498f-4a93-8d33-0e78374b074f
md"""
**(a)** Show that the Ritz eigenvectors $\tilde{x}_i$ and $\tilde{x}_j$ corresponding to different approximate eigenvalues $\tilde{λ}_i \neq \tilde{λ}_j$ are orthogonal.

**(b)** Use the Rayleigh-Ritz procedure with the given subspace to obtain three approximate eigenvectors $\tilde{x}_1$, $\tilde{x}_2$, $\tilde{x}_3$ and corresponding eigenvalues $\tilde{λ}_1$, $\tilde{λ}_2$, $\tilde{λ}_3$, respectively.

**(c)** Use the Bauer-Fike theorem to obtain an *a posteriori* bound for the error in the computed eigenvalues. Verify your computed value is indeed an upper bound by computing the exact eigenvalues of the densified matrix (`Matrix(B)`).

**(d)** Starting from the subspace $\mathcal{S}$, respectively the vectors `V` run different iterative diagonalisation algorithms, e.g. the `projected_subspace_iteration` and the `lobpcg` routines from the lecture or your *dynamical* shift algorithm from Exercise 1(e). Use `tol=1e-6` and be not afraid to increase `maxiter` for this part of the exercise. You should observe that different eigenpairs are found in each case. Try to explain why each of the algorithms finds the respective eigenpairs. Keeping your orbservations in mind, can one in general rely on an algorithm to find *all* eigenvalues with correct multiplicity within the part of the spectrum spanned by the smallest and largest eigenvalue the algorithm returns?

**(e)** Run the LOBPCG algorithm on `B` starting from a random guess aiming for $4$ eigenvectors. Use the Bauer-Fike and Kato-Temple theorems to estimate the error in the first eigenvalue. Use the tightest estimate that is available to you. You may assume that the LOBPCG algorithm did not miss any eigenvalue, i.e. that you have indeed approximations for the first and second eigenpair at your disposal. Vary the tolerance between `1e-4` and `1e-10` and plot the relationships between tolerance, estimated error and true error.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LinearMaps = "7a12625a-238d-50fd-b39a-03d52299707e"
MatrixDepot = "b51810bb-c9f3-55da-ae3c-350fc1fbce05"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[compat]
LinearMaps = "~3.11.1"
MatrixDepot = "~1.0.10"
PlutoUI = "~0.7.52"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "9139bb6e1a53a21039e2e00336290884a7ea635b"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BufferedStreams]]
git-tree-sha1 = "4ae47f9a4b1dc19897d3743ff13685925c5202ec"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.2.1"

[[deps.ChannelBuffers]]
deps = ["CodecZlib", "Downloads", "Serialization", "Tar", "TranscodingStreams"]
git-tree-sha1 = "c522f957325aab2c5457328d711b7dfef3603cb1"
uuid = "79a69506-cdd1-4876-b8e5-7af85e53af4f"
version = "0.3.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "02aa26a4cf76381be7f66e020a3eddeb27b0a092"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.2"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "e460f044ca8b99be31d35fe54fc33a5c33dd8ed7"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.9.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "Mmap", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "114e20044677badbc631ee6fdc80a67920561a29"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.16.16"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "38c8874692d48d5440d5752d6c74b0c6b0b60739"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.2+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

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

[[deps.LinearMaps]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9df2ab050ffefe870a09c7b6afdb0cde381703f2"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.11.1"

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"
    LinearMapsSparseArraysExt = "SparseArrays"
    LinearMapsStatisticsExt = "Statistics"

    [deps.LinearMaps.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "79fd0b5ee384caf8ebba6c8fb3f365ca3e2c5493"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.5"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "8a5b4d2220377d1ece13f49438d71ad20cf1ba83"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.1.2+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "781916a2ebf2841467cda03b6f1af43e23839d85"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.9"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "6979eccb6a9edbbb62681e158443e79ecc0d056a"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.3.1+0"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MatrixDepot]]
deps = ["ChannelBuffers", "DataFrames", "LinearAlgebra", "MAT", "Markdown", "Mmap", "Scratch", "Serialization", "SparseArrays"]
git-tree-sha1 = "fb9ab8df44551c1cc26cffcce0d6ab93a8f303b4"
uuid = "b51810bb-c9f3-55da-ae3c-350fc1fbce05"
version = "1.0.10"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "a8027af3d1743b3bfae34e54872359fdebb31422"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.3+4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

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

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "f3080f4212a8ba2ceb10a34b938601b862094314"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "4.1.5+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e78db7bd5c26fc5a6911b50a47ee302219157ea8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.10+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "ee094908d720185ddbdc58dbe0c1cbe35453ec7a"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "b7a5e99f24892b6824a954199a45e9ffcc1c70f0"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eddd19a8dea6b139ea97bdc8a0e2667d4b661720"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.0.6+1"

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
# ╟─f669e56e-9c27-4025-9bce-79ee0b5f5981
# ╟─767f0942-e9a8-4260-b3f9-273ea35a0844
# ╠═b6cf13cd-451e-49f3-b339-6e1ed55659f9
# ╟─ee7657bb-923e-4b74-a70f-e230da92e4c0
# ╠═3ed9cf5a-cda2-4e96-b6f0-5e09f0ee87b1
# ╠═f8fedbaf-ec6d-4d00-bf40-3b2759cbf45f
# ╠═9b80b038-16a5-418a-8dd2-ee6a3a2f1f00
# ╟─8fd23d56-ec08-4446-87ec-4fffa206cf20
# ╟─ba3c816e-b0d0-46f0-bcd4-e284af89e3ba
# ╠═add79a1d-1c0e-44aa-a73d-84c5bddd2ecc
# ╠═d720c05f-2999-4b0f-8319-bbe68e8b0945
# ╠═6cb1101b-9a4c-4a1d-bf74-e84ea4aa36d3
# ╟─5a00f6cc-e684-4db2-8376-e69d73cee8a8
# ╠═d51a787b-0ac8-48de-849e-1b475572dc11
# ╟─d4df3a19-a04f-4b3c-b3ef-3bca92f5a8a7
# ╠═a8802b27-864d-4bf0-b2d0-4812f7f54f78
# ╟─44974d4c-498f-4a93-8d33-0e78374b074f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
