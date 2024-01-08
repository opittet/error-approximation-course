### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ a9548adc-cc71-4f26-96d2-7aefc39a554f
begin
	using Brillouin
	using DFTK
	using LinearAlgebra
	using Plots
	using PlutoTeachingTools
	using PlutoUI
	using Printf
	using LaTeXStrings
	using Measurements
end

# ╔═╡ 4b7ade30-8c59-11ee-188d-c3d4588f7106
md"Error control in scientific modelling (MATH 500, Herbst)"

# ╔═╡ 35066ab2-d52d-4a7f-9170-abbae6216996
md"""
# Project 2: Band structures with guaranteed error bars
*To be handed in via moodle by 12.01.2024*
"""

# ╔═╡ f09f563c-e458-480f-92be-355ab4582fb5
TableOfContents()

# ╔═╡ 363b4496-5728-4eea-a3cc-4a090952155c
md"""
## Introduction

In this project we will compute approximations of
parts of the eigenspectra of a specific class of Hamiltonians
```math
H = -\frac12 \Delta + V.
```
Precisely we will consider simple potentials $V$,
which are periodic with respect to some lattice $\mathbb{L}$.
For this setting we will further develop mathematically guaranteed
estimates for the total numerical error of our eigenvalue computations,
such that the exact answer is provably within our error bound.
Following the lecture we will use Bloch-Floquet theory
to reformulate this problem in terms of Bloch fibers
```math
H_k = \frac{1}{2}(-i \nabla_x + k)^2
```
where the $k$-points are taken from the Brillouin zone $\Omega^\ast$.
Of main interest and the focus of our work is the so-called band structure,
that is the dependency of the few lowest eigenvalues of the $H_k$ on $k$.
At the end of this project you will have coded up a routine to
compute the band structure of simple periodic potentials $V$ in a way
that the error is fully tracked and mathematically guaranteed
error bars can be annotated to the computed band structure.

For our numerical computation in this project we will mostly employ the density-functional toolkit (DFTK), which has already been installed alongside this notebook. You can find the source code on [dftk.org](https://dftk.org) and documentation as well as plenty of usage examples on [docs.dftk.org](https://docs.dftk.org). In particular an alternative introduction to periodic problems, which focuses more on the numerics than the mathematics is given on [https://docs.dftk.org/stable/guide/periodic_problems/](https://docs.dftk.org/stable/guide/periodic_problems/).
"""

# ╔═╡ 11ca4a8e-2f55-40ea-b9cc-37ba7806bb5c
md"""
As example systems in this work we will only consider face-centred cubic crystals in the diamond structure. As the name suggests this family of crystals includes diamond, but also materials such as bulk silicon, germanium or tin. For these the lattice
```math
\mathbb{L} = \mathbb{Z} a_1 + \mathbb{Z} a_2 + \mathbb{Z} a_3
```
can be constructed from the unit cell vectors
```math
a_1 = \frac{a}{2} \left(\begin{array}{c} 0\\1\\1 \end{array}\right)\qquad
a_2 = \frac{a}{2} \left(\begin{array}{c} 1\\0\\1 \end{array}\right)\qquad
a_3 = \frac{a}{2}\left(\begin{array}{c} 1\\1\\0 \end{array}\right)
```
where $a$ is the lattice constant, which differs between the materials. Keeping the notation from the lecture we will use $\Omega$ to refer to the unit cell
```math
\Omega = \left\{x \in \mathbb{R}^3 \,\middle|\, |x - R| > |x| \quad \forall R \in \mathbb{L}\setminus\{0\}\right\},
```
and employ $\mathbb{L}^\ast$ to refer to the reciprocal lattice with vectors $b_1$, $b_2$, $b_3$ with $a_i \cdot b_j = 2π δ_{ij}$. $\Omega^\ast$ denotes the first Brillouin zone (unit cell of  $\mathbb{L}^\ast$).

We introduce the normalised plane waves
```math
e_G(x) = \frac{e^{iG \cdot x}}{\sqrt{|\Omega|}}\quad \forall G \in \mathbb{L}^\ast,
```
which allow to express the Fourier series of the potential as
```math
V(x) = \frac{1}{\sqrt{|\Omega|}} \sum_{G \in \mathbb{L}^\ast} \hat{V}(G) e^{i G \cdot x}
= \sum_{G \in \mathbb{L}^\ast} \hat{V}(G) e_G(x).
```
To simplify our notation we will assume that $G$-vectors always run over the countably infinite reciprocal lattice $\mathbb{L}^\ast$ without making explicit reference to this set.

An early model to describe these materials theoretically is the Cohen-Bergstresser model[^CB1966]. The special and simplifying property of this model is that only a relatively small number of Fourier coefficients is non-zero. In particular we can define a cutoff $\mathcal{E}_V$ such that
```math
V(x) = \sum_{|G| < \sqrt{2\mathcal{E}_V}} \hat{V}(G) e_G(x),
```
where each $\hat{V}(G)$ is finite.

[^CB1966]: M. L. Cohen and T. K. Bergstresser Phys. Rev. **141**, 789 (1966) DOI [10.1103/PhysRev.141.789](https://doi.org/10.1103/PhysRev.141.789)

"""

# ╔═╡ 354b8072-dcf0-4182-9897-c3e9534bef5a
md"""
### Task 1: Cohen-Bergstresser Hamiltonians

**(a)** Using the results of the lecture show that $H$ is self-adjoint on $L^2(\mathbb{R}^3)$ with the domain $D(H) = H^2(\mathbb{R}^3)$ if $V$ is the Cohen-Bergstresser potential. *Hint:* Show and use that the Cohen-Bergstresser potential is bounded and $\mathbb{L}$-periodic. 
"""

# ╔═╡ 0c00fca4-41b4-4c1d-b4b1-d6668c42fe65
md"""
**(a) Solution:**

A function is called bounded if there exists a constant $C$ such that:

```math
\sup_{0 \neq x \in H^2} | V(x)| \leq C.
```

Considering the norm of $V(x)$ we have:
```math
\begin{align}
| V(x)| &\leq  \frac{1}{\sqrt{|\Omega|}}  \sum_{|G| < \sqrt{2\mathcal{E}_V}} | \hat{V}(G)| |e^{iG \cdot x}| = \\
	&= \sum_{|G| < \sqrt{2\mathcal{E}_V}} \frac{1}{\sqrt{|\Omega|}} | \hat{V}(G)| = C,
\end{align}
```

where $C$ is a constant since $\hat{V}(G)$ is finite and only a small number of Fourier coefficients is non-zero: ...because $G$ is from the lattice and inside the circle. The constant $| \Omega |$ is the volume of the unit cell which can be calculated as follows: 

```math
```

Therefore, the Cohen-Bergstresser potential is bounded.

On the other hand, the periodicity of the complex exponential $e^{iG \cdot x}$ on the lattice $\mathbb{L}$, from which follows the $\mathbb{L}$-periodicity of $e_G(x)$. Since $V(x)$ is a finite sum of $\mathbb{L}$-periodic functions, it follows that $V(x)$ is itself $\mathbb{L}$-periodic.

Therefore, since $V(x) \in L^{3/2}_{per}(\Omega)$ by applying theorem 10.1 the operator $H$ is self adjoint.
"""

# ╔═╡ ef2796f3-b6d4-4403-aaa6-ed3d9d541a3a
md"""
A consequence of *(a)* is that the Bloch-Floquet transformation yields Bloch fibers
```math
H_k = T_k + V \quad \text{where} \quad T_k = \frac{1}{2}(-i \nabla_x + k)^2
```
which are self-adjoint on $L^2_\text{per}(\Omega)$ with $D(H_k) = H^2_\text{per}(\Omega)$. The natural inner product for our problem is thus the $\langle \, \cdot\, | \, \cdot \, \rangle_{L^2_\text{per}(\Omega)}$ inner product, which is the one we will employ unless a different one is specified.

For a particular $k$-point we can thus discretise the problem using the plane-wave basis
```math
\mathbb{B}_k^{\mathcal{E}} = \left\{ e_G \, \middle| \, G \in \mathbb{L}^\ast \ \text{ and } \frac12 |G+k|^2 < \mathcal{E} \right\}
```
where $\mathcal{E}$ is a chosen cutoff.
"""

# ╔═╡ 40f62be2-8394-4886-84e8-d595b6ff7cab
md"""
**(b)** Show that 
```math
\langle e_G | e_{G'} \rangle = \delta_{GG'}
```
and that
```math
\langle e_G | T_k e_{G'} \rangle = \delta_{GG'} \frac12 |G+k|^2.
```
From this conclude that for any $e_G \in \mathbb{B}_k^{\mathcal{E}}$ we have
```math
|\Delta G| > \sqrt{2\mathcal{E}_V} \quad \Longrightarrow \quad
\langle e_{G + \Delta G} | H e_{G} \rangle = 0, \qquad (1)
```
i.e. that each plane wave only couples via the Hamiltonian with these plane waves, that differ in the wave vector by no more than $\sqrt{2\mathcal{E}_V}$.

-------
"""

# ╔═╡ e06ad336-9ff8-48bb-9eaf-4a606d69d51c
md"""
**Solution (b):**

We have

```math
\begin{align}
\langle e_G | e_{G'} \rangle &= \int_{\Omega} e_G^*(x) e_{G'}(x) \,dx = \frac{1}{|\Omega|} \int_{\Omega} e^{i(G' - G) \cdot x} \,dx = \frac{1}{|\Omega|} \delta_{G, G'} \cdot | \Omega |
\end{align}
```

why Fuirer coefficients are orthogonal
"""

# ╔═╡ 05d40b5e-fd83-4e73-8c78-dfd986a42fc0
md"""
## Familiarisation and Bauer-Fike estimates

Work through [https://docs.dftk.org/stable/guide/periodic_problems/](https://docs.dftk.org/stable/guide/periodic_problems/) as well as [https://docs.dftk.org/stable/guide/tutorial/](https://docs.dftk.org/stable/guide/tutorial/) to get some basic familiarity with DFTK. 

Following loosely [another documented example](https://docs.dftk.org/stable/examples/cohen_bergstresser/), this code sets up a calculation of silicon using the Cohen-Bergstresser potential and $\mathcal{E} = 10$.
"""

# ╔═╡ c4393902-7c57-4126-80af-8765bea42ebd
begin
	Si = ElementCohenBergstresser(:Si)
	atoms = [Si, Si]
	positions = [ones(3)/8, -ones(3)/8]
	lattice = Si.lattice_constant / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
end;

# ╔═╡ 78cc8d4a-cb63-48d0-a2f9-b8ec8c2950e5
begin
	model = Model(lattice, atoms, positions; terms=[Kinetic(), AtomicLocal()])
	basis_one = PlaneWaveBasis(model; Ecut=10.0, kgrid=(1, 1, 1));
end

# ╔═╡ 0aaeb747-ce4f-4d53-a3b7-f8019c3d688b
md"""
The `kgrid` parameter selects the $k$-point mesh $\mathbb{K} \subset \overline{\Omega^\ast}$, which is employed by determining the number of $k$-points in each space dimension. In this case only a single $k$-point is used, namely the origin of $\overline{\Omega^\ast}$:
"""

# ╔═╡ ffd75a52-9741-4c12-ae2f-303db8d70332
basis_one.kpoints

# ╔═╡ 0f6a001d-ce82-4d4b-a914-37bf0b62d604
md"""
The origin of the Brillouin zone has a special name and is usually called $\Gamma$-point. Before treating the case of band structure computations (where $k$ is varied), we stick to using only a single $k$-point.

Each `KPoint` in DFTK automatically stores a representation of the basis $\mathbb{B}_k^{\mathcal{E}}$. If you are curious, the $G$-vectors of the respective plane waves $e_G$ can be looked at using the functions
"""

# ╔═╡ 66b93963-272f-4e1b-827c-0f3b551ee52b
G_vectors_cart(basis_one, basis_one.kpoints[1])  # G-vectors of 1st k-point

# ╔═╡ dfe6be2f-fb6b-458b-8016-8d89dfa66ed9
md"""
Similarly a discretised representation of $H$ restricted to the fibres matching the chosen $k$-point mesh can be obtained using
"""

# ╔═╡ 252e505d-a43d-472a-9fff-d529c3a2eeb7
ham_one = Hamiltonian(basis_one)

# ╔═╡ 0f5d7c6c-28d7-4cf9-a279-8117ddf7d9d5
md"""
Let us denote by $P_k^\mathcal{E}$ the projection
into the plane-wave basis $\mathbb{B}_k^\mathcal{E}$
and further by
$H_k^{\mathcal{E}\mathcal{E}} \equiv P_k^\mathcal{E} H_k P_k^\mathcal{E}$
the discretised fibers with elements
```math
\left(H_k^{\mathcal{E}\mathcal{E}}\right)_{GG'}
= \left(P_k^\mathcal{E} H_k P_k^\mathcal{E}\right)_{GG'}
= \langle e_G,\, H_k\, e_{G'} \rangle
\qquad \text{for $G, G' \in \mathbb{B}_k^\mathcal{E}$}.
```
Representations of $H_k^{\mathcal{E}\mathcal{E}}$ in DFTK
are available via indexing of `Hamiltonian` objects.
E.g. `ham[1]` corresponds to the fiber of `basis.kpoints[1]`,
`ham[2]` to `basis.kpoints[2]` and so on.
This can also be seen from comparing their sizes with the
size of $\mathbb{B}_k^\mathcal{E}$ (available via the `G_vectors_cart`):
"""

# ╔═╡ b4ffe212-6f71-4d61-989a-570c5448877a
begin
	@show length(G_vectors_cart(basis_one, basis_one.kpoints[1]))
	@show size(ham_one[1])
	@show typeof(ham_one[1])
end

# ╔═╡ b9de31a7-1e75-47ca-9901-f2a5ee3a5630
md"""
These `DftHamiltonianBlock` objects very much behave like matrices,
e.g. they can be multiplied with vectors, but also used in iterative solvers.

The `diagonalize_all_kblocks` function does some optimisations and book-keeping to solve for the `n_bands` lowest eigenpairs at each $H_k$. Here we solve for $6$ eigenpairs:
"""

# ╔═╡ c3a6aec7-ab22-4017-8e2f-bfe5696021dd
begin
	n_bands = 6
	eigres_one = diagonalize_all_kblocks(DFTK.lobpcg_hyper, ham_one, n_bands)
end

# ╔═╡ 8f3634ca-9c13-4d40-af3c-0c80588fd8c8
md"""
The returned object `eigres_one` now contains the eigenvalues in `band_data.λ` and the Bloch waves in `band_data.X`. Again both are a vector along the $k$-points, further `band_data.X[ik]` is of size `length(G_vectors_cart(basis, kpoint))` $\times$ `n_bands`:
"""

# ╔═╡ ac0cffd2-effb-489c-a118-60864798d55e
begin
	@show size(eigres_one.X[1])
	@show (length(G_vectors_cart(basis_one, basis_one.kpoints[1])), n_bands)
end

# ╔═╡ c50b9d2e-dcf5-42be-b401-3da8c514165b
function vector_norm(A,d=1)
	return sqrt.(sum(abs2,A,d))
end

# ╔═╡ 142ac96e-bd9d-45a7-ad31-e187f28e884a
md"""
As a result if we wanted to compute at each $k$-point the residual
```math
H_k^{\mathcal{E}\mathcal{E}} \widetilde{X}_{kn} - \widetilde{λ}_{kn} \widetilde{X}_{kn}
\qquad (2)
```
after we have been able to diagonalise $H_k^{\mathcal{E}\mathcal{E}}$,
we can simply iterate as such:
"""

# ╔═╡ 601c837c-1615-4826-938c-b39bb35f46d1
for (ik, kpt) in enumerate(basis_one.kpoints)
	hamk = ham_one[ik]
	λk   = eigres_one.λ[ik]
	Xk   = eigres_one.X[ik]

	residual_k = hamk * Xk - Xk * Diagonal(λk)
	println(ik, "  ", norm(residual_k))
end

# ╔═╡ e0a07aca-f81a-436b-b11e-8446120e0235
md"""
### Task 2: $\Gamma$-point calculations

**(a)** For the case of employing only a single $k$-point (`kgrid = (1, 1, 1)`) vary $\mathcal{E}$ (i.e. `Ecut`) between $5$ and $30$. Taking $\mathcal{E} = 80$ as a reference, plot the convergence of the first two eigenpairs of $H_k$ at the $\Gamma$ point.
"""

# ╔═╡ 2bf7d1a6-cd49-41ee-a670-4352febd02b6
begin
	Ecut_2a=collect(5:30)
	push!(Ecut_2a,80)
	λ_conv=[]
	X_conv=[]
	res_norm=[]
	ρ_bauer_fike=[]
	n_bands_2a = 2
	δ_kato_temple=[]

	for Ecut_iter in Ecut_2a
		
		basis_2a = PlaneWaveBasis(model; Ecut=Ecut_iter, kgrid=(1, 1, 1))
		ham_2a = Hamiltonian(basis_2a)
		eigres_2a = diagonalize_all_kblocks(DFTK.lobpcg_hyper, ham_2a, n_bands_2a)

		
		for (ik, kpt) in enumerate(basis_2a.kpoints)

			hamk = ham_2a[ik]
			λk   = eigres_2a.λ[ik]
			Xk   = eigres_2a.X[ik]
			#Xk_reordered=Vector{Vector{Float64}}()
			#for i in 1:n_bands_2a
					
			#	Xk_i= getindex.(Xk,i)
			#	println("xk",size(Xk_i))
			#	push!(Xk_reordered,Xk_i)
			#end	
				
				
			# not possible to do norm of vectors for a matrix https://discourse.julialang.org/t/vecnorm-column-row-wise-norm/12354


			
			residual_k = hamk * Xk - Xk * Diagonal(λk)

			column_norms = [norm(residual_k[:,i]) for i in 1:n_bands_2a] #bauer_fike
			
			δk=[]
			for i in 1:n_bands_2a #Kato-Temple
				if i==n_bands_2a
					δi= abs(λk[i] - λk[i-1]) - column_norms[i]
				else
					δi= abs(λk[i] - λk[i+1]) - column_norms[i]
				end
			push!(δk,Float64(δi))
			println(δk)
			end
			push!(δ_kato_temple,δk)
			
			push!(ρ_bauer_fike,column_norms)
			push!(λ_conv,λk)
			push!(X_conv,Xk)
			push!(res_norm,norm(residual_k))
		end		
	end
	println(δ_kato_temple)
	ρ_kato_temple= [getindex.(ρ_bauer_fike,i) .^2 ./ getindex.(δ_kato_temple,i) for i in 1:n_bands_2a]
	println(size(ρ_bauer_fike))
	println(ρ_bauer_fike)
	#println(res_norm)
	
	#plot in for loop with n_bands doesn't work for some reason
	
	for i in 1:n_bands_2a
	    #println(getindex.(λ_conv, i))
	    plot!(Ecut_2a, [λ[i] for λ in λ_conv], label="Curve $i")
	end
end


# ╔═╡ 3856b78c-cadf-4e6c-868b-8026ab0bb670
begin
	println(δ_kato_temple)
	println(ρ_kato_temple)
end

# ╔═╡ 0f70ff13-ffb0-4b9a-8570-7e324ff4afd7
begin
	p= plot(layout=(n_bands_2a, 1),xlabel=L"$\mathcal{E}$ values")
	λ1=[λ[1] for λ in λ_conv]
	λ2=[λ[2] for λ in λ_conv]


	plot!(p,Ecut_2a[1:end-1],λ1[1:end-1],subplot=1,shape=:cross,label="first eigenvalue")

    hline!([λ1[end]], line=:dash, color=:red, subplot=1, label="Ecut = 80")
	plot!(p,Ecut_2a[1:end-1],λ2[1:end-1],subplot=2,shape=:cross,label="second eigenvalue")
    hline!([λ2[end]], line=:dash, color=:red, subplot=2, label="Ecut = 80")
	title!(p,subplot=1,"convergence of the first 2 eigenvalues")




end

# ╔═╡ f41b96ae-51a6-4aae-b9d2-ef0c546f2999
begin
	p_r= plot(layout=(2, 1),xlabel=L"$\mathcal{E}$ values")

	println(getindex.(ρ_bauer_fike,1))
	println(getindex.(ρ_kato_temple,1))
	
	plot!(p_r,Ecut_2a[1:end-1],getindex.(ρ_bauer_fike,1)[1:end-1],subplot=1,shape=:cross,label="the residual norm of first eigenvalue")
	plot!(p_r,Ecut_2a[1:end-1],getindex.(ρ_bauer_fike,2)[1:end-1],subplot=1,ylabel="Bauer-Fike bounds",shape=:cross,label="the residual norm of 2nd eigenvalue")
	
	plot!(p_r,Ecut_2a[1:end-1],ρ_kato_temple[1][1:end-1],subplot=2,shape=:cross,label="the residual norm of first eigenvalue")
	plot!(p_r,Ecut_2a[1:end-1],ρ_kato_temple[2][1:end-1],ylabel="Kato-temple bounds",subplot=2,shape=:cross,label="the residual norm of 2nd eigenvalue")


	title!(p_r,subplot=1,"Error bounds with Bauer-Fike & Kato-Temple")

	




end

# ╔═╡ 189d4007-8e6f-4f54-9828-005febe2b89c
begin
	p_x= plot(layout=(n_bands_2a, 1),xlabel=L"$\mathcal{E}$ values",yscale=:log10,minorgrid=true)
	X1=[X[1] for X in X_conv]
	X2=[X[2] for X in X_conv]
	println(norm.(X1))

	plot!(p_x,Ecut_2a[1:end-1],norm.(X1[1:end-1]),subplot=1,shape=:cross,label="first eigenvector norm")
    hline!([norm.(X1[end])], line=:dash, color=:red, subplot=1, label="Ecut = 80")
	
	plot!(p_x,Ecut_2a[1:end-1],norm.(X2[1:end-1]),subplot=2,shape=:cross,label="second eigenvector norm")
    hline!([norm.(X2[end])], line=:dash, color=:red, subplot=2, label="Ecut = 80")
	
	title!(p_x,subplot=1,"convergence of the first 2 eigenvector norms")




end

# ╔═╡ 6bfa0a42-a0e6-4bef-9dc0-0023796f087b
begin
	p_diff= plot(layout=(n_bands_2a, 1),xlabel=L"$\mathcal{E}$ values",yscale=:log10,minorgrid=true)

	λ1_dif=[abs(λ-λ1[end]) for λ in λ1]
	λ2_dif=[abs(λ-λ2[end]) for λ in λ2]


	plot!(p_diff,Ecut_2a[1:end-1],λ1_dif[1:end-1],subplot=1,shape=:cross,label=L"|\lambda_1-\lambda_{1,\mathcal{E}=80}|")

	plot!(p_diff,Ecut_2a[1:end-1],λ2_dif[1:end-1],subplot=2,shape=:cross,label=L"|\lambda_2-\lambda_{2,\mathcal{E}=80}|")
	title!(p_diff,subplot=1,"absolute difference with converged λ")



end

# ╔═╡ 682f52e7-f7a9-4146-b789-268672049fa9


# ╔═╡ 58856ccd-3a1c-4a5f-9bd2-159be331f07c
md"""
We want to compare this convergence behaviour with a first estimate of the eigenvalue discretisation error based on the Bauer-Fike bound. Given an approximate eigenpair $(\widetilde{λ}_{kn}, \widetilde{X}_{kn})$ of the fiber $H_k$ we thus need access to the residual
```math
r_{kn} = H_k \widetilde{X}_{kn} - \widetilde{λ}_{kn} \widetilde{X}_{kn}.
```
If we performed our computation
using the plane-wave basis $\mathbb{B}_k^\mathcal{E}$,
then by construction
```math
P_k^\mathcal{E} \widetilde{X}_{kn} = \widetilde{X}_{kn},
```
such that
```math
\begin{align*}
P_k^\mathcal{E} r_{kn} &= P_k^\mathcal{E} H_k P_k^\mathcal{E}\, P_k^\mathcal{E} \widetilde{X}_{kn} - \widetilde{λ}_{kn} P_k^\mathcal{E} \widetilde{X}_{kn} \\
&= H_k^{\mathcal{E}\mathcal{E}} \widetilde{X}_{kn} - \widetilde{λ}_{kn} \widetilde{X}_{kn},
\end{align*}
```
which is exactly the quantity we computed in $(2)$ above.
In other words $P_k^\mathcal{E} r_{kn}$ is the residual corresponding to the
iterative diagonalisation, thus leading to the **algorithm error**.

However, we are also interested in the **discretisation error**, which we obtain from the missing residual term in $(2)$, namely 
```math
Q_k^\mathcal{E} r_{kn} = (1 - P_k^\mathcal{E}) r_{kn} = Q_k^\mathcal{E} H_k P_k^\mathcal{E} \widetilde{X}_{kn} = H_k^{\mathcal{E}^\perp\mathcal{E}} \widetilde{X}_{kn}
```
where $Q_k^\mathcal{E} = (1-P_k^\mathcal{E})$ is the orthogonal projector to $P_k^\mathcal{E}$ and we introduced the notation
```math
H_k^{\mathcal{E}^\perp\mathcal{E}} = Q_k^\mathcal{E} H_k P_k^\mathcal{E}.
```


While computing $P_k^\mathcal{E} r_{kn}$ is generally easy (we just did it above in 5 lines), obtaining $Q_k^\mathcal{E} r_{kn}$ is impossible for general potentials $V$: since $H_k P_k^\mathcal{E} \widetilde{X}_{kn}$ can have support anywhere on $L^2_\text{per}(\Omega)$, estimating $Q_k^\mathcal{E} r_{kn}$ usually requires making a mathematical statement about the behaviour of $H_k$ on all of $L^2_\text{per}(\Omega)$. Estimating discretisation errors is thus substantially more challenging than estimating algorithm errors.
"""

# ╔═╡ e04087db-9973-4fad-a964-20d109fff335
md"""
**(b)** In our case of Cohen-Bergstresser Hamiltonians we are more lucky. Using $(1)$ show that there exists a cutoff $\mathcal{F} > \mathcal{E}$, such that
```math
r_{kn} = P_k^\mathcal{F} r_{kn} = H_k^{\mathcal{F}\mathcal{F}} \widetilde{X}_{kn} - \widetilde{λ}_{kn} \widetilde{X}_{kn}.
```
Having diagonalised the Hamiltonian using cutoff $\mathcal{E}$ we can thus obtain the *full residual* by a computation of the Hamiltonian-vector product using an elevated cutoff $\mathcal{F}$.
"""

# ╔═╡ abdfcc43-245d-425a-809e-d668f03e9b45
md"""
**(c)** In DFTK nobody stops you from using multiple bases of different size. Moreover, having obtained a set of bloch waves `X_small` using `basis_small` you can obtain its representation on the bigger `basis_large` using the function
```julia
X_large = transfer_blochwave(X_small, basis_small, basis_large)
```
Using this technique you can compute the application of the Hamiltonian using a bigger basis (just as `Hamltonian(basis_large) * X_large`). Use this setup to vary $\mathcal{F}$ and use this to estimate $\mathcal{E}_V$ numerically. Check your estimate for various values of $\mathcal{E}$ to ensure it is consistent. Rationalise your results by taking a look at the [Cohen Bergstresser implementation in DFTK](https://github.com/JuliaMolSim/DFTK.jl/blob/0b61e06db832ce94f6b66c0ffb1d215dfa4822d4/src/elements.jl#L142).
"""

# ╔═╡ d26fec73-4416-4a20-bdf3-3a4c8ea533d1
md"""
**(d)** Based on the Bauer-Fike bound estimate the algorithm and arithmetic error for the first two eigenpairs at the $\Gamma$ point and for using cutoffs between $\mathcal{E} = 5$ and $\mathcal{E} = 30$. Add these estimates to your plot in $(a)$. What do you observe regarding the tightness of the bound ?
"""

# ╔═╡ e42ff75d-f401-4b15-9421-81b24277e01b
begin
	p_bf= plot(layout=(n_bands_2a, 1),xlabel=L"$\mathcal{E}$ values")
	print(ρ_bauer_fike)


	plot!(p_bf,Ecut_2a[1:end-1],λ1[1:end-1].±getindex.(ρ_bauer_fike,1)[1:end-1],subplot=1,shape=:cross,label="first eigenvalue")
    hline!([λ1[end]], line=:dash, color=:red, subplot=1, label="Ecut = 80")
	
	plot!(p_bf,Ecut_2a[10:end-1],λ2[10:end-1].±getindex.(ρ_bauer_fike,2)[10:end-1],subplot=2,shape=:cross,label="second eigenvalue")
    hline!([λ2[end]], line=:dash, color=:red, subplot=2, label="Ecut = 80")
	
	title!(p_bf,subplot=1,"convergence of the first 2 eigenvalues with Bauer-Fike bounds")
	




end

# ╔═╡ 983b1a76-ecb9-42cd-a109-d8847d78ca79
md"""
The bounds do not get tighter as the eigenvalues converge.

"""

# ╔═╡ faa100bb-4012-49ab-929d-303a26fa0634
begin
	p_kt= plot(layout=(n_bands_2a, 1),xlabel=L"$\mathcal{E}$ values")
	print(ρ_bauer_fike)


	plot!(p_kt,Ecut_2a[15:end-1],λ1[15:end-1].±ρ_kato_temple[1][15:end-1],subplot=1,shape=:cross,label="first eigenvalue")
    hline!([λ1[end]], line=:dash, color=:red, subplot=1, label="Ecut = 80")
	
	plot!(p_kt,Ecut_2a[20:end-1],λ2[20:end-1].±ρ_kato_temple[2][20:end-1],subplot=2,shape=:cross,label="second eigenvalue")
    hline!([λ2[end]], line=:dash, color=:red, subplot=2, label="Ecut = 80")
	
	title!(p_kt,subplot=1,"convergence of the first 2 eigenvalues with Kato-Temple bounds")
	




end

# ╔═╡ 9a953b4e-2eab-4cb6-8b12-afe754640e22
md"""----"""

# ╔═╡ 0616cc6b-c5f8-4d83-a247-849a3d8c5de8
md"""
## Band structure computations

Next we turn our attention to band structure computations. Recall that the $n$-th band is the mapping from $k$ to the $n$-th eigenvalue $λ_{kn}$ of the Bloch fiber $H_k$. For standard lattices --- like the diamond-FCC lattices we consider --- there are tabulated standard 1D paths through the Brillouin zone $\Omega^\ast$, that should be used to maximise the physical insight one may gain from the computation.

In DFTK this path can be automatically determined. The following function provides a corresponding list of $k$-points to consider (the warnings can be ignored in this function call):
"""

# ╔═╡ d363fa0a-d48b-4904-9337-098bcef015bb
kpath = interpolate(irrfbz_path(model); density=15)

# ╔═╡ 9e7d9b0f-f29d-468f-8f7c-e01f271d57c6
md"""
The `density` parameter in the above call determines how many $k$-points are considered along the path. A larger density leads to more points.

Based on this data we can use the `compute_bands` function to diagonalise all $H_k$ along the path using a defined `density` and cutoff `Ecut`. Internally this function uses `diagonalize_all_kblocks`, so the returned data structure is alike.
"""

# ╔═╡ d34fa8e5-903f-45f4-97f8-32abd8883e9f
function compute_bands_auto(model; Ecut, kwargs...)
	basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))
	compute_bands(basis, kpath; show_progress=false, kwargs...)
end

# ╔═╡ e3244165-9f04-4676-ae80-57667a6cb8d5
band_data = compute_bands_auto(model; n_bands=6, tol=1e-3, Ecut=10)

# ╔═╡ 9bc1e1b6-7419-4b8b-aebc-a90b076e87da
md"""
For example we obtain the eigenvalues of the 3rd $k$-point
"""

# ╔═╡ 61448d06-c628-493b-8fa0-a8f0a41acd1d
band_data.basis.kpoints[3]

# ╔═╡ 722f34dd-e1cb-4a3c-9a13-7ff6117029fc
md"using the expression"

# ╔═╡ 06270fe2-df78-431a-8433-d69fc89ae5c3
band_data.λ[3]

# ╔═╡ e56b2164-5f66-4745-b1a5-711dcc1a324b
md"Once the data has been computed, a nice bandstructure plot can be produced using"

# ╔═╡ 66992c0d-262c-4211-b5d2-1ba247f9e8ba
DFTK.plot_band_data(kpath, band_data)

# ╔═╡ 86f70a52-13ce-42b5-9a28-2c01d92b022d
md"Error bars for indicating eigenvalue errors can also be easily added:"

# ╔═╡ 2dee9a6b-c5dd-4511-a9e3-1d5bc33db946
begin
	λerror = [0.02 * abs.(randn(size(λk))) for λk in band_data.λ]  # dummy data
	data_with_errors = merge(band_data, (; λerror))
	print(band_data)
	print(data_with_errors)
	DFTK.plot_band_data(kpath, data_with_errors)
end

# ╔═╡ da8c23eb-d473-4c2f-86f9-879016659f3e
md"""
### Task 3: Bands with Bauer-Fike estimates

For $\mathcal{E} = 7$ a $k$-point density of $15$ and the $6$ lowest eigenpairs plot a band structure in which you annotate the bands with error bars estimated from Bauer-Fike. To compute the Hamiltonian-vector product employing a larger basis set with cutoff `Ecut_large`, employ the following piece of code, which forwards the $k$-Point coordinates from one basis to another:
"""

# ╔═╡ 3006a991-d017-40b3-b52b-31438c05d088
function basis_change_Ecut(basis_small, Ecut_large)
	PlaneWaveBasis(basis_small.model, Ecut_large,
                   basis_small.kcoords_global,
                   basis_small.kweights_global)
end

# ╔═╡ 8ad78ab5-fb3b-4a0c-a21d-fbddbf3d3f8c
	band_data3 = compute_bands_auto(model; n_bands=6, tol=1e-3, Ecut=7)


# ╔═╡ 6bda48f6-a89a-4827-9ba9-b20d00e27761
begin
	λ_conv3=[]
	X_conv3=[]
	res_norm3=[]
	ham_list3=[]
	ρ_bauer_fike3 = Vector{Vector{Float64}}()
	println(ρ_bauer_fike3)

		
	ham_3 = Hamiltonian(band_data3.basis)
	eigres_3 = diagonalize_all_kblocks(DFTK.lobpcg_hyper, ham_3, 6)
	
	large_base3=basis_change_Ecut(band_data3.basis,20)
	ham_large3=Hamiltonian(large_base3)
	eigres_large_3 = diagonalize_all_kblocks(DFTK.lobpcg_hyper, ham_large3, 6)


	for (ik, kpt) in enumerate(band_data3.basis.kpoints)
		

		

		hamk = ham_3[ik]
		hamk_large= ham_large3[ik]
		
		λk   = eigres_3.λ[ik]
		Xk   = eigres_3.X[ik]
		
		λk_large   = eigres_large_3.λ[ik]
		Xk_large   = eigres_large_3.X[ik]
		
		residual_k = hamk_large * Xk_large - Xk_large * Diagonal(λk_large)
		column_norms = [norm(residual_k[:,i]) for i in 1:6]

		#print(size(residual_k))
		#println(norm.(residual_k))
		#println(typeof(norm.(residual_k)))
		#println(size(norm.(residual_k)))

		push!(ham_list3,hamk)
		push!(ρ_bauer_fike3,column_norms)
		push!(λ_conv3,λk)
		push!(X_conv3,Xk)
		push!(res_norm3,norm(residual_k))
	end		
	println(ρ_bauer_fike3)

end

# ╔═╡ e2274211-47d9-4751-92e4-fbd2839a4b2c

begin
	#println(ρ_bauer_fike3)
	band_data3_bf=merge(band_data3,(;ρ_bauer_fike3))
	DFTK.plot_band_data(kpath, band_data3_bf)
end

# ╔═╡ f3a8531e-cad3-4575-9944-52d2cade1f76
print(data_with_errors)

# ╔═╡ 2c62a65c-5b8c-44cc-a028-5860f5119e8c
print(band_data3_bf)

# ╔═╡ b83e56f8-a339-4f53-a765-3f1295145287
begin
	println(typeof(ρ_bauer_fike3))
	println(size(ρ_bauer_fike3))
	println(size(ρ_bauer_fike3[1]))
	
	println(typeof(λerror))
	println(size(λerror))
	println(size(λerror[1]))
end

# ╔═╡ 8aa3159d-4845-463a-a1d2-336ec8f6efa6
basis3=basis_change_Ecut(basis_one,Ecut_large=7)

# ╔═╡ 646242a2-8df6-4caf-81dc-933345490e6c
md"""
-------
"""

# ╔═╡ d7f3cadf-6161-41ce-8a7a-20fba5182cfb
md"""
## Kato-Temple estimates

Following up on Task 2 (d) we want to improve the bounds using Kato-Temple estimates. Similar to our previous developments in this regard the challenging ingredient is estimating a lower bound to the gap. If $λ_{kn}$ denotes the eigenvalue closest to the approximate eigenvalue $\widetilde{λ}_{kn}$, we thus need a lower bound to
```math
δ_{kn} = \min_{s \in \sigma(H_k) \backslash \{λ_{kn}\}} |s - \widetilde{λ}_{kn}|.
```

The usual approach we employed so far (see e.g. Sheet 6) was to assume that our obtained approximations $\widetilde{λ}_{kn}$ for $n = 1,\ldots,N$ did not miss any of the exact eigenvalues. With this assumption we would employ a combination of Kato-Temple and Bauer-Fike bound to obtain a lower bound to $δ_{kn}$.

Already for estimating the eigenvalue error of iterative eigensolvers, this can be a far-fetched assumptions as we saw in previous exercises. For estimating the discretisation error where the discretisation by nature of projecting into a finite-dimensional space inevitably changes the properties of the spectrum, this is even more far fetched. In fact in practice it is not rare that an unsuited basis changes the order of eigenpairs, completely neglects certain eigenpairs etc. Since the goal of *a posteriori* estimation is exactly to detect such cases, we will in Task 5 also develop techniques based on weaker assumptions.

Before doing so, we first extend the usual technique to infinite dimensions.
"""

# ╔═╡ f6efea9b-9656-4337-82a1-5b894e078338
md"""
### Task 4: Simple and non-guaranteed Kato-Temple estimates

**(a)** Notice that the Kato-Temple theorem requires the targeted eigenvalue $λ_{kn}$ to be isolated, see the conditions of Theorem 9.9. and 9.8. in the notes. Considering our setting of periodic Schrödinger operators in $L^2$. Looking at the spectral properties of $H$ and $H_k$ argue why there is even hope that Kato-Temple can be employed in our setting?

**(b)** Assume that our numerical procedure (discretisation + diagonalisation) for all considered cutoffs is sufficiently good, such that no eigenpairs are missed. Employ the prevously discussed (e.g. Sheet 6) technique of combining Kato-Temple and Bauer-Fike estimates to obtain an improved estimate for the error in the eigenvalue. Add your estimate for the first eigenvalue at the $Γ$-point to your plot in Task 2 (d). What do you observe ?

**(c)** Perform a band structure computation at cutoff $\mathcal{E} = 7$ annotated with the error estimated using the estimate developed in *(b)*. For approximate eigenvalues where you cannot obtain a Kato-Temple estimate (e.g. degeneracies), fall back to a Bauer-Fike estimate. Play with the cutoff. What do you observe ? Do the error bars correspond to the expectations of a variational convergence ? Apart from the probably unjustified assumption in *(b)*, what is the biggest drawback of the Kato-Temple estimate ?

-----
"""

# ╔═╡ 7121fcd7-c9ec-4d59-ad26-e8b3887ba65e
begin
	p_kt_again= plot(layout=(n_bands_2a, 1),xlabel=L"$\mathcal{E}$ values")

	plot!(p_kt_again,Ecut_2a[15:end-1],λ1[15:end-1].±ρ_kato_temple[1][15:end-1],subplot=1,shape=:cross,label="first eigenvalue")
    hline!([λ1[end]], line=:dash, color=:red, subplot=1, label="Ecut = 80")
	
	plot!(p_kt_again,Ecut_2a[20:end-1],λ2[20:end-1].±ρ_kato_temple[2][20:end-1],subplot=2,shape=:cross,label="second eigenvalue")
    hline!([λ2[end]], line=:dash, color=:red, subplot=2, label="Ecut = 80")
	
	title!(p_kt_again,subplot=1,"convergence of the first 2 eigenvalues with Kato-Temple bounds")
	




end

# ╔═╡ c7c3f1ab-def9-4de9-8fc7-112be4c3631d
begin
		p_r_again=plot(layout=(2, 1),xlabel=L"$\mathcal{E}$ values")
		plot!(p_r_again,Ecut_2a[1:end-1],getindex.(ρ_bauer_fike,1)[1:end-1],subplot=1,shape=:cross,label="the residual norm of first eigenvalue")
		plot!(p_r_again,Ecut_2a[1:end-1],getindex.(ρ_bauer_fike,2)[1:end-1],subplot=1,ylabel="Bauer-Fike bounds",shape=:cross,label="the residual norm of 2nd eigenvalue")
		
		plot!(p_r_again,Ecut_2a[1:end-1],ρ_kato_temple[1][1:end-1],subplot=2,shape=:cross,label="the residual norm of first eigenvalue")
		plot!(p_r_again,Ecut_2a[1:end-1],ρ_kato_temple[2][1:end-1],ylabel="Kato-temple bounds",subplot=2,shape=:cross,label="the residual norm of 2nd eigenvalue")
	
	
		title!(p_r_again,subplot=1,"Error bounds with Bauer-Fike & Kato-Temple")
end

# ╔═╡ cda35232-b0ef-490f-9be2-b5a2c84d4ad6


# ╔═╡ 1738ca7b-5d33-484c-a37f-eb2f70086a64
md"""
the Kato-Temple is simply a shift of $\approx 10^{-6}$ because the $\delta$ value is very stable, thus this bound performs strictly better than its Bauer-Fike counter-part.  
"""

# ╔═╡ 14df1c0f-1c4f-4da9-be49-3941b9c12fd3
md"""
### Task 5: A Schur-based estimate for the gap

Based on the definition of the basis cutoffs $\mathcal{E}$ and $\mathcal{F}$
and the respective projectors $P_k^\mathcal{E}$, $P_k^\mathcal{F}$ into these
bases as well as $Q_k^\mathcal{E}$ and $Q_k^\mathcal{F}$
as the projectors to the complements of the bases,
we can identify the following orthogonal decompositions of the Hilbert space
$L^2_\text{per}(\Omega)$:
```math
\begin{align}
id &= (P_k^{\mathcal{E}} + Q_k^{\mathcal{E}}) \\
   &= (P_k^{\mathcal{E}} + P_k^{\mathcal{F}} Q_k^{\mathcal{E}} + Q_k^{\mathcal{F}} Q_k^{\mathcal{E}})\\
   &= (P_k^{\mathcal{E}} + P_k^{\mathcal{R}} + Q_k^{\mathcal{F}})
\end{align}
```
where we defined $P_k^\mathcal{R} \equiv P_k^\mathcal{F} Q_k^\mathcal{E}$.
Using the notations
```math
\begin{aligned}
	H_k^{\mathcal{E}\mathcal{F}}       &= P_k^\mathcal{E} H_k P_k^\mathcal{F} &
	\quad
	H_k^{\mathcal{R}\mathcal{E}}       &= P_k^\mathcal{R} H_k P_k^\mathcal{E} \\
	H_k^{\mathcal{E}\mathcal{E}^\perp} &= P_k^\mathcal{E} H_k Q_k^\mathcal{E} &
	H_k^{\mathcal{E}\mathcal{F}^\perp} &= P_k^\mathcal{E} H_k Q_k^\mathcal{F}
\end{aligned}
```
and so on, we can decompose the Hamiltonian fiber into blocks:
```math
	H_k 
= \left(\begin{array}{ccc}
	H_k^{\mathcal{E}\mathcal{E}} & H_k^{\mathcal{E}\mathcal{E}^\perp} \\
	H_k^{\mathcal{E}^\perp\mathcal{E}} & 
	\textcolor{darkred}{H_k^{\mathcal{E}^\perp\mathcal{E}^\perp}} \\
\end{array}\right)
= 
\left(\begin{array}{ccc}
	H_k^{\mathcal{E}\mathcal{E}} & H_k^{\mathcal{E}\mathcal{R}} & H_k^{\mathcal{E}\mathcal{F}^\perp} \\
	H_k^{\mathcal{R}\mathcal{E}} & \textcolor{darkred}{H_k^{\mathcal{R}\mathcal{R}}} & \textcolor{darkred}{H_k^{\mathcal{R}\mathcal{F}^\perp}} \\
	H_k^{\mathcal{F}^\perp\mathcal{E}} & \textcolor{darkred}{H_k^{\mathcal{F}^\perp\mathcal{R}}} & \textcolor{darkred}{H_k^{\mathcal{F}^\perp\mathcal{F}^\perp}} \\
\end{array}\right).
```

We note that our iterative diagonalisation as part of the band structure calculation indeed gives us access to a few eigenpairs $(\widetilde{λ}_{kn}, \widetilde{X}_{kn})$ of $H_k^{\mathcal{E}\mathcal{E}}$. Moreover due to Courant-Fisher $λ_{kn} \leq \widetilde{λ}_{kn}$, i.e. an upper bound to $λ_{kn}$ is readily at hand. Suppose now we additionally had a lower bound $μ_{kn}$, i.e.
```math
\mu_{kn} \leq λ_{kn} \leq \widetilde{λ}_{kn}
```
then
```math
δ_{kn} \geq \min\left(μ_{kn} - \widetilde{λ}_{k,n-1}, \, μ_{k,n+1} - \widetilde{λ}_{k,n}\right) \qquad (3).
```
Finding such a suitable $μ_{kn}$ provably is the goal of this task.
"""

# ╔═╡ 3fc07beb-33c1-43b3-9d66-27693d78e46a
md"""
**(a)** Show that for Cohen-Bergstresser Hamiltonians and an appropriate choice of $\mathcal{F}$ depending on $\mathcal{E}$ (Task 2 (c)), the following structure of the Hamiltonian is obtained:
```math
	H_k =
\left(\begin{array}{ccc}
	H_k^{\mathcal{E}\mathcal{E}} & V_k^{\mathcal{E}\mathcal{R}} & 0\\
	V_k^{\mathcal{R}\mathcal{E}} & H_k^{\mathcal{R}\mathcal{R}} & V_k^{\mathcal{R}\mathcal{F}^\perp} \\
	0 & V_k^{\mathcal{F}^\perp\mathcal{R}} & H_k^{\mathcal{F}^\perp\mathcal{F}^\perp} \\
\end{array}\right),
```
where the use of $V$ instead of $H$ indicates that the kinetic term does not contribute to the respective block and $0$ indicates an all-zero block.
"""

# ╔═╡ 047d630b-e85e-45e9-9574-758955cb160e
md"""
We focus on the splitting of $H_k$ into four blocks. An important realisation for our purpose is now that if $H_k - \mu_{kn}$ has exactly $n-1$ negative eigenvalues, then we necessarily have $\mu_{kn} < λ_{kn}$. In the following we will thus develop conditions on a parameter $\mu$, such that $H_k - \mu$ has a provable number of negative eigenvalues.

A first step is the  Haynsworth inertia additivity formula[^Hay1968]. For a matrix
```math
	H_k - \mu
= \left(\begin{array}{ccc}
	H_k^{\mathcal{E}\mathcal{E}} - \mu & V_k^{\mathcal{E}\mathcal{E}^\perp} \\
	V_k^{\mathcal{E}^\perp\mathcal{E}} & 
	H_k^{\mathcal{E}^\perp\mathcal{E}^\perp} - \mu \\
\end{array}\right).
```
this theorem states that
```math
N(H_k - \mu) = N(H_k^{\mathcal{E}\mathcal{E}} - \mu) + N(S_\mu)
```
where $N(\mathcal{A})$ denotes the number of negative eigenvalues of the operator $\mathcal{A}$ and $S_\mu$ is the Schur complement
```math
S_\mu = \left(H_k^{\mathcal{E}^\perp \mathcal{E}^\perp} - \mu\right)
- V_k^{\mathcal{E}^\perp\mathcal{E}} \left( H_k^{\mathcal{E} \mathcal{E}} - \mu \right)^{-1} V_k^{\mathcal{E}\mathcal{E}^\perp}.
```

[^Hay1968]: E. V. Haynsworth. Linear Algebra Appl. **1**, 73 (1968)
"""

# ╔═╡ fc040eb6-872b-475d-a6cd-7d3ad1fae229
md"""
**(b)** Using this statement explain why a $\mu \in (\widetilde{λ}_{k,n-1}, \widetilde{λ}_{kn})$ such that $S_\mu \geq 0$ is a guaranteed lower bound to $λ_{kn}$.

**(c)** We proceed to obtain conditions that ensure $S_\mu \geq 0$. Let $X = L^2_\text{per}(\Omega) \backslash \text{span}\left(\mathbb{B}_k^\mathcal{E}\right)$. Recall that $S_\mu \geq 0$ exactly if
```math
\langle x,  S_\mu x \rangle \geq 0 \qquad \forall x \in X
```
Prove the following statements:
- First $\forall x \in X$
  ```math
  \left\langle x \,\middle|\, (H_k^{\mathcal{E}\mathcal{E}} - \mu) x \right\rangle
  \geq \mathcal{E} - \|V_k^{\mathcal{E}^\perp\mathcal{E}^\perp} \|_\text{op} - \mu,
  ```
  where $\|\,\cdot\,\|_\text{op}$ is the standard Hilbert operator norm
  ```math
  \|\mathcal{A}\|_\text{op} = \sup_{0\neq\varphi\in L^2_\text{per}(\Omega)} \frac{\langle\varphi, \mathcal{A} \varphi\rangle}{\langle\varphi, \varphi\rangle}
  ```
- Second, given a full eigendecomposition $H_k^{\mathcal{E} \mathcal{E}} = \widetilde{X}_k \widetilde{Λ}_k \widetilde{X}_k^H$ show that
  for $\mu \in I_n$ with the open interval
  ```math
  I_n = \left(\frac12 \left(\widetilde{λ}_{k,n-1} + \widetilde{λ}_{kn}\right), \, \widetilde{λ}_{kn}\right)
  ```
  we have $\forall x \in X$:
  ```math
  \begin{align}
  \left\langle x \, \middle|\,
  V_k^{\mathcal{E}^\perp\mathcal{E}} \left( H_k^{\mathcal{E} \mathcal{E}} - \mu   \right)^{-1} V_k^{\mathcal{E}\mathcal{E}^\perp} x \right\rangle
  &\leq \left\|\left(V_k^{\mathcal{R}\mathcal{E}} \widetilde{X}_k\right) \left(\widetilde{Λ}_k - \mu\right)^{-1} \left( V_k^{\mathcal{R}\mathcal{E}} \widetilde{X}_k \right)^H \right\|_\text{op}\\
  &\leq \frac{\|V_k\|_\text{op}^2}{\widetilde{λ}_{kn} - \mu}
  \end{align}
  ```
- Combining both show that
  ```math
  S_\mu \geq \mathcal{E} - \mu - l_1^V - \frac{(l_1^V)^2}{\widetilde{λ}_{kn} - \mu} 
  \quad \text{where} \quad l_1^V = \sum_{|G| < \sqrt{2\mathcal{E}_V}} |\hat{V}(G)|. \quad\quad (4)
  ```
  You may want to use Young's inequality, that is
  ```math
  \|V_k\|_\text{op} \leq \sum_{G \in \mathbb{L}^\ast} |\hat{V}(G)|.
  ```
"""

# ╔═╡ 74df4d8b-0345-449e-ad3a-ded44a94a40d
md"""
The strategy to determine $\mu$ is thus as follows:

1. Compute $l_1^V$. For this we need a Fourier representation of $V$. A real space representation is available as `Vreal = DFTK.total_local_potential(ham)` where `ham` is a `DFTK.Hamiltonian`, which can be transformed into Fourier space using a fast-fourier transform `fft(basis, kpoint, Vreal)` where `basis` is the `ham.basis` and `kpoint` is the `Kpoint` object matching the currently processed $k$. Note that this assumes that $\mathcal{E}_V < \mathcal{E}$.

2. Using the lower bound $(4)$ find the largest $\mu \in I_n$ for which it can be ensured that $S_\mu \geq 0$. Since we are not using the exact $S_\mu$ here, but only a lower bound, this step can fail (i.e. both possible solutions $\mu$ are outside of $I_n$). In this case we are unable to obtain a lower bound for $\widetilde{λ}_{kn}$ and thus unable to obtain a Kato-Temple estimate.

3. If 2. goes through we set $\mu_{kn} = \mu$ and thus have guaranteed bounds $\mu_{kn} \leq λ_{kn} \leq \widetilde{λ}_{kn}$.

**(d)** Focusing on the first two eigenpairs of the $\Gamma$-point of the Cohen-Bergstresser model, plot both the guaranteed lower bound $\mu_{kn}$ from above procedure as well as the non-guarandeed bound of Task 4 as you increase $\mathcal{E}$. Take the computation of $\widetilde{λ}_{nk}$ for $\mathcal{E} = 80$ as the reference. Which bound is sharper ? While it should be emphasised that better guaranteed bounds than our development are possible (see for example [^HLC2020]), can you comment on the advantages and disadvantages of guaranteed error bounds ?

--------

[^HLC2020]:  M. F. Herbst, A. Levitt and E. Cancès. Faraday Discuss., **224**, 227 (2020). DOI [10.1039/D0FD00048E](https://doi.org/10.1039/D0FD00048E)
"""

# ╔═╡ 48ffb85e-884d-46a4-8184-40126b603aac
md"""
### Task 6: Band structure with guaranteed bounds

Based on the results so far compute a band structure with guaranteed error bars for $\mathcal{E} = 7$ a $k$-point density of $15$ and the $6$ first bands (i.e. the $6$ lowest eigenpairs). For each eigenvalue:
- Estimate the **algorithm** and **arithmetic error** by re-computing the in-basis residual $P_k^\mathcal{E}r_{kn}$ using `Double64`. For this you need to build a basis and a Hamiltonian that employes `Double64` as the working precision. To change DFTK's internal working precision follow the [Arbitrary floating-point types](https://docs.dftk.org/stable/examples/arbitrary_floattype/) documentation page. From a `basis_double64` using `Double64` precision a corresponding Hamiltonian is obtained by `Hamiltonian(basis_double64)` as shown before. *Hint:* Before computing products `Hamiltonian(basis_double64) * X`, ensure that `X` has been converted to `Double64` as well.
- Estimate the **discretisation error** using both the Bauer-Fike estimates of Tasks 2 & 3 as well as the Kato-Temple estimates of Task 5. In the band structure annotate the tightest bound that is available to you.

------
"""

# ╔═╡ 33896109-d190-4992-806a-c447ca36071b
md"""
## Adaptive diagonalisation techniques

The main purpose of *a posteriori* error bounds as we have developed in the previous tasks is to understand the  error in the current computational setup. A good error bound is sharp, cheap to compute and ideally guaranteed, i.e. the actual error is always smaller. One can thus think of the error bars as safety checks. If one is not yet satisfied with the current quality of results, one can always choose a more accurate numerical setup and re-compute. For example if the estimated discretisation error is too large, one just increases $\mathcal{E}$ until one is satisfied.

As simple as this procedure is, it has one notable drawback: It leads to a nested set of loops, where for each attempted $\mathcal{E}$ (outer loop), we iteratively diagonalise the $H_k$ using LOBPCG (inner loop). A natural question is thus, whether one can fuse these two loops, i.e. adapt $\mathcal{E}$ *while the LOBPCG is converging*, such that at each step the algorithm error and discretization error are balanced.

To make this idea more clear, let us recall a basic LOBPCG implementation, which in this case only performs a diagonalisation of the Hamiltonian fiber at the gamma point of a passed model:
"""

# ╔═╡ 93552ccd-f7b5-4830-a9ad-417d3fca9af9
function nonadaptive_lobpcg(model::Model{T}, Ecut, n_bands;
                            maxiter=100, tol=1e-6, verbose=false) where {T}
	kgrid = (1, 1, 1)  # Γ point only
	basis = PlaneWaveBasis(model; Ecut, kgrid)
	ham   = Hamiltonian(basis)
	hamk  = ham[1]                   # Select Γ point
	prec  = PreconditionerTPA(hamk)  # Initialise preconditioner
	X     = DFTK.random_orbitals(hamk.basis, hamk.kpoint, n_bands)
	
	converged = false
	λ = NaN
	residual_norms = NaN
	residual_history = []
	
	P = zero(X)
	R = zero(X)
	for i in 1:maxiter
		if i > 1
			Z = hcat(X, P, R)
		else
			Z = X
		end
		Z = Matrix(qr(Z).Q)  # QR-based orthogonalisation

		# Rayleigh-Ritz
		HZ = hamk * Z
		λ, Y = eigen(Hermitian(Z' * HZ))
		λ = λ[1:n_bands]
		Y = Y[:, 1:n_bands]
		new_X = Z * Y
		
		# Compute residuals and convergence check
		R = HZ * Y - new_X * Diagonal(λ)
		residual_norms = norm.(eachcol(R))
		push!(residual_history, residual_norms)
		verbose && @printf "%3i %8.4g %8.4g\n" i λ[end] residual_norms[end]
		if maximum(residual_norms) < tol
			converged = true
			X .= new_X
			break
		end

		# Precondition and update
		DFTK.precondprep!(prec, X)
		ldiv!(prec, R)
		P .= X - new_X
		X .= new_X

		# Additional step:
		# Move to larger basis ?
	end

	(; λ, X, basis, ham, converged, residual_norms, residual_history)
end

# ╔═╡ 968a26f2-12fe-446f-aa41-977fdffc23a0
md"""
This implementation can be easily compared to the `DFTK.lobpcg_hyper` function of DFTK. For example:
"""

# ╔═╡ 7ebc18b0-618f-4d99-881f-7c30eb3bc7f5
let
	Ecut     = 10
	n_bands  = 6
	basis    = PlaneWaveBasis(model; Ecut=10, kgrid=(1, 1, 1))
	ham      = Hamiltonian(basis)
	res_dftk = diagonalize_all_kblocks(DFTK.lobpcg_hyper, ham, n_bands)
	res_dftk_λ_Γ = res_dftk.λ[1]  # Extract eigenvalues at Γ point

	# Notice: This function always only computes one k-Point
	res_nonadaptive = nonadaptive_lobpcg(model, Ecut, n_bands)

	res_nonadaptive.λ - res_dftk_λ_Γ
end

# ╔═╡ bf00ec9d-5c69-43c3-87c1-28ac9f5b4c9f
md"""
In order to make this routine discretisation-adaptive we need to add an additional step as indicated in the above source code: After we have checked for convergence and as we are updating `X`, `P` and `R` to their new values, we might additionally want to increase the discretisation basis to employ a new, increased cutoff value $\mathcal{E} + \Delta$ to discretise the Hamiltonian. Using
```julia
transfer_blochwave(X, basis_small, basis_small.kpoints[1],
                      basis_large, basis_large.kpoints[1])
```
we can then transfer the state vectors `X`, `P` and `R` to this basis.

This means while the iterative eigensolver is converging we also improve the discretisation basis by successively increasing $\mathcal{E}$. Each LOBPCG iteration may thus run using a different discretisation basis.
"""

# ╔═╡ 38db50ac-5937-4e9b-948c-fd93ced44cb2
md"""
### Task 7: Developing a discretisation-adaptive LOBPCG

Assume the LOBPCG currently runs at cutoff $\mathcal{F}$. Due to the properties of the Cohen-Bergstresser Hamiltonian (Task 2 (b)) there is a smaller cutoff $\mathcal{E} < \mathcal{F}$ such that (Task 5 (a))
```math
Q_k^{\mathcal{F}} H P_k^{\mathcal{E}} = 0.
```

**(a)** Prove that if $\|Q_k^{\mathcal{E}} \widetilde{X}_{kn}\| = 0$, then
```math
|λ_{kn} - \widetilde{λ}_{kn}| \leq \|P_k^{\mathcal{F}} r_{kn}\|,
```
i.e. that the total error is driven to zero as the iterative diagonalisation employing a cutoff $\mathcal{F}$ is converging. Based on this argue why a large value for $\|Q_k^{\mathcal{E}} \widetilde{X}_{kn}\|$ relative to $\|P_k^{\mathcal{F}} r_{kn}\|$ provides a reasonable indicator to decide when to refine the discretisation.

**(b)** Following the idea of (a) code up an adaptive LOBPCG by monitoring the ratio
```math
\frac{\|Q_k^{\mathcal{E}} \widetilde{X}_{kn}\|}
{\|P_k^{\mathcal{F}} r_{kn}\|}  \qquad \quad (5).
```
Note, that one way to compute $Q_k^{\mathcal{E}} \widetilde{X}_{kn}$ is as $\widetilde{X}_{kn} - P_k^{\mathcal{E}} \widetilde{X}_{kn}$ where $P_k^{\mathcal{E}}$ can be obtained by transferring from the large basis to the small basis and back to the large basis, i.e.
```julia
X_small = transfer_blochwave(X_large,
						   basis_large, basis_large.kpoints[1],
						   basis_small, basis_small.kpoints[1])
P_X     = transfer_blochwave(X_small,
						   basis_small, basis_small.kpoints[1],
						   basis_large, basis_large.kpoints[1])
```
Whenever (5) becomes too large, you should switch to a finer discretisation basis, e.g. by switching from a cutoff $\mathcal{E}$ to $\mathcal{E} + \Delta$. Experiment with this setup to find what are good ratios $(5)$ to indicate switching and good values for $\Delta$. Some strategy for exploration:
- Assume we eventually want to target a cutoff $\mathcal{F}^\text{final} = 80$. Save the convergence history for running the full LOBPCG at exactly this cutoff. This is your reference.
- With your adaptive methods you should try to loose as little as possible in the rate of convergence compared to the reference convergence profile, but try to use as small values of $\mathcal{F}$ for as many iterations as you can.
- To develop your approach, first only compute a single eigenvalue (`n_bands = 1`) and then start considering more than one. Note, that you will need to adapt $(5)$ and take apropriate maxima / minima over all computed eigenpairs to decide when to switch to the next bigger basis.
- Be creative and experiment. In this Task there is no "best" solution.
"""

# ╔═╡ 56c10e5c-90e0-4aa7-a343-38aadff37693
md"""
## GTH pseudopotentials

To close off this project we will now consider slightly more realistic potentials than the Cohen-Bergstresser model. We will look at the local part of the Goedecker-Teter-Hutter pseudopotentials[^GTH96]. Neglecting some minor details involving the structure of the system, these potentials can be understood in Fourier space as 
```math
\hat{V}(p) = P(p) \exp(- p^2 / σ^2),
```
i.e. a strongly decaying Gaussian times a polynomial prefactor. Instead of just a few non-zero Fourier coefficients like in the Cohen-Bergtresser case, we thus have the weaker setting, that the Fourier cofficients only strongly decay as $|G| \to \infty$. Unlike our development in Task 2 (b), we thus cannot find an elevated cutoff $\mathcal{F} > \mathcal{E}$, such that the residual can be computed exactly. However, due to the strong decay the truncated residual $P_k^\mathcal{F} r_{kn}$ of Tasks 2 becomes a better and better approximation to $r_{kn}$ the larger $\mathcal{E}$ and $\mathcal{F}$ are taken. Thus the ideas of Tasks 2 can still be employed to compute *approximate* errors for this GTH potential --- though without any guarantees that the actual error is smaller than these estimates.

[^GTH96]: S. Goedecker, M. Teter, and J. Hutter Phys. Rev. B **54**, 1703 (1996). DOI [10.1103/PhysRevB.54.1703](https://doi.org/10.1103/PhysRevB.54.1703)
"""

# ╔═╡ 7fb27a85-27eb-4111-800e-a8c306ea0f18
md"""
### Task 8: An error indicator for GTH pseudopotentials

A setup for the GTH pseudopotential model, discretised for a provided cutoff $\mathcal{E}$ and using only the $\Gamma$-point is given by the function
"""

# ╔═╡ e0f916f6-ce7f-48b1-8256-2e6a6247e171
function make_gth_basis(Ecut)
	Si = ElementPsp(:Si, psp=load_psp("hgh/lda/si-q4"))
	atoms = [Si, Si]
	positions = [ones(3)/8, -ones(3)/8]
	lattice = 5 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
	model = Model(lattice, atoms, positions; terms=[Kinetic(), AtomicLocal()])
	PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))
end

# ╔═╡ 62fd73e1-bd11-465f-8032-e87db00bfda7
md"""
Assume a secondary cutoff $\mathcal{F} > \mathcal{E}$, which is large enough that the error bounds developed in Task 2 are suffiently good to be useful. In other words we will assume that the residual computed in $\mathbb{B}^\mathcal{F}_k$, i.e.
```math
P_k^\mathcal{F} r_{kn} = H_k^{\mathcal{F}\mathcal{F}} \widetilde{X}_{kn} - \widetilde{λ}_{kn} \widetilde{X}_{kn},
```
is a good estimate of the true residual.

Focusing on the first $5$ eigenvalues of the $\Gamma$ point of the GTH Hamiltonian, investigate the Bauer-Fike error bounds based on this approximate residual. Investigate in particular:

**(a)** Take $\mathcal{E} = 100$ as a reference and plot the error bound for  $\mathcal{E} = 5$ and for varying $\mathcal{F}$ between $10$ and $100$. Does the bound always seem to hold ? Does it converge with $\mathcal{F}$ ?

**(b)** For a few offsets $\Delta$ between $5$ and $30$ set $\mathcal{F} = \mathcal{E} + \Delta$ and consider the $2$nd and $3$rd eigenvalue. Plot both the obtained error estimate as well as the error of the eigenvalue as you vary $\mathcal{E}$ between $5$ and $50$. Again take $\mathcal{E} = 100$ as a reference. What offset $\Delta$ would you recommend based on this investigation ?
"""

# ╔═╡ 3dab1ef5-ba74-4b22-89c6-8fea651fd959
X_large = transfer_blochwave(eigres_one.X, basis_one, basis_large)

# ╔═╡ 90ab127d-9054-4a0f-88ba-8a340fa3afcc
Hamiltonian(basis_large) * X_large

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Brillouin = "23470ee3-d0df-4052-8b1a-8cbd6363e7f0"
DFTK = "acf6eb54-70d9-11e9-0013-234b7a5f5337"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[compat]
Brillouin = "~0.5.14"
DFTK = "~0.6.14"
LaTeXStrings = "~1.3.1"
Measurements = "~2.11.0"
Plots = "~1.39.0"
PlutoTeachingTools = "~0.2.13"
PlutoUI = "~0.7.54"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "668bd3d4dabbb71614b44174b6317f5c857ac155"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "793501dcd3fa7ce8d375a2c878dca2296232686e"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "02f731463748db57cc2ebfbd9fbc9ce8280d3433"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "247efbccf92448be332d154d6ca56b9fcdd93c31"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.6.1"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AtomsBase]]
deps = ["LinearAlgebra", "PeriodicTable", "Printf", "Requires", "StaticArrays", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "995c2b6b17840cd87b722ce9c6cdd72f47bab545"
uuid = "a963bdd2-2df7-4f54-a1ee-49d51e6be12a"
version = "0.3.5"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.Bravais]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "08c0613c61dcdd5dd148982cf91b107654e2e811"
uuid = "ada6cbde-b013-4edf-aa94-f6abe8bd6e6b"
version = "0.1.8"

[[deps.Brillouin]]
deps = ["Bravais", "DirectQhull", "DocStringExtensions", "LinearAlgebra", "PrecompileTools", "Reexport", "Requires", "StaticArrays"]
git-tree-sha1 = "443cb0bf360741bfe41484e286d32816c4f11196"
uuid = "23470ee3-d0df-4052-8b1a-8cbd6363e7f0"
version = "0.5.14"

    [deps.Brillouin.extensions]
    BrillouinMakieExt = "Makie"
    BrillouinPlotlyJSExt = "PlotlyJS"
    BrillouinSpglibExt = "Spglib"

    [deps.Brillouin.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    PlotlyJS = "f0f68f2c-4968-5e81-91da-67840de0976a"
    Spglib = "f761d5c5-86db-4880-b97f-9680a7cccfb5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "601f7e7b3d36f18790e2caf83a882d88e9b71ff1"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.4"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "1e42ef1bdb45487ff28de16182c0df4920181dc3"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.7.0+0"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "9704e50c9158cf8896c2776b8dbc5edd136caf80"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.10.1+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e0af648f0692ec1691b5d094b8724ba1346281cf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.18.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "c0216e792f518b39b22212127d4a84dc31e4e386"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.ComponentArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "ForwardDiff", "Functors", "LinearAlgebra", "PackageExtensionCompat", "StaticArrayInterface", "StaticArraysCore"]
git-tree-sha1 = "d30eb4d89c791a64e698546c1e0e0e488cd99da5"
uuid = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
version = "0.15.5"

    [deps.ComponentArrays.extensions]
    ComponentArraysAdaptExt = "Adapt"
    ComponentArraysConstructionBaseExt = "ConstructionBase"
    ComponentArraysGPUArraysExt = "GPUArrays"
    ComponentArraysRecursiveArrayToolsExt = "RecursiveArrayTools"
    ComponentArraysReverseDiffExt = "ReverseDiff"
    ComponentArraysSciMLBaseExt = "SciMLBase"
    ComponentArraysTrackerExt = "Tracker"
    ComponentArraysZygoteExt = "Zygote"

    [deps.ComponentArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SciMLBase = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "8cfa272e8bdedfa88b6aefbbca7c19f1befac519"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.CrystallographyCore]]
deps = ["LinearAlgebra", "StaticArrays", "StructEquality"]
git-tree-sha1 = "521a8ed44455592672d887632eef056c1ba4678c"
uuid = "80545937-1184-4bc9-b283-396e91386b5c"
version = "0.3.3"

[[deps.DFTK]]
deps = ["AbstractFFTs", "Artifacts", "AtomsBase", "Brillouin", "ChainRulesCore", "Dates", "DftFunctionals", "FFTW", "ForwardDiff", "GPUArraysCore", "InteratomicPotentials", "Interpolations", "IterTools", "IterativeSolvers", "LazyArtifacts", "Libxc", "LineSearches", "LinearAlgebra", "LinearMaps", "LoopVectorization", "MPI", "Markdown", "Optim", "OrderedCollections", "PeriodicTable", "PkgVersion", "Polynomials", "PrecompileTools", "Preferences", "Primes", "Printf", "ProgressMeter", "PseudoPotentialIO", "Random", "Requires", "Roots", "SparseArrays", "SpecialFunctions", "Spglib", "StaticArrays", "Statistics", "TimerOutputs", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "f8484d6bd8a4a53341800f1d2199e67d9334151b"
uuid = "acf6eb54-70d9-11e9-0013-234b7a5f5337"
version = "0.6.14"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DftFunctionals]]
deps = ["ComponentArrays", "DiffResults", "ForwardDiff"]
git-tree-sha1 = "0ae7d5b8d65091e4b59f5859c85be4427c57de11"
uuid = "6bd331d2-b28d-4fd3-880e-1a1c7f37947f"
version = "0.2.4"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DirectQhull]]
deps = ["Qhull_jll"]
git-tree-sha1 = "5a941ad556ad4d2e310828b0f0b462678887ec2e"
uuid = "c3f9d41a-afcb-471e-bc58-0b8d83bd86f4"
version = "0.2.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "5225c965635d8c21168e32a12954675e7bea1151"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.10"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.EzXML]]
deps = ["Printf", "XML2_jll"]
git-tree-sha1 = "380053d61bb9064d6aa4a9777413b40429c79901"
uuid = "8f5d6c58-4d21-5cfd-889c-e3ad7ee6a615"
version = "1.2.0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "b4fbdd20c889804969571cc589900803edda16b7"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "35f0c0f345bff2c6d636f95fdb136323b5a796ef"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.7.0"
weakdeps = ["SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "c6e4a1fbe73b31a3dea94b1da449503b8830c306"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.21.1"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9a68d75d466ccc1218d0552a8e1631151c569545"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5eab648309e2e060198b45820af1a37182de3cce"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "eb8fed28f4994600e29beef49744639d985a04b2"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.16"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8ecb0b34472a3c98f945e3c75fc7d5428d165511"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.9.3+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ad37c091f7d7daf900963171600d7c1c5c3ede32"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InteratomicPotentials]]
deps = ["AtomsBase", "Distances", "LinearAlgebra", "NearestNeighbors", "StaticArrays", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "e52c1cff4fa468972621f0b5dd45ce2ee08dc730"
uuid = "a9efe35a-c65d-452d-b8a8-82646cd5cb04"
version = "0.2.6"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "b435d190ef8369cf4d79cc9dd5fba88ba0165307"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.3"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "9fb0b890adab1c0a4a475d4210d51f228bfc250d"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.6"

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

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "0592b1810613d1c95eeebcd22dc11fba186c2a57"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.26"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "62edfee3211981241b57ff1cedf4d74d79519277"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.15"

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

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.Libxc]]
deps = ["Libxc_GPU_jll", "Libxc_jll", "Requires"]
git-tree-sha1 = "5ed405a6c6b3580fecc205dc9799b82f5492aab8"
uuid = "66e17ffc-8502-11e9-23b5-c9248d0eb96d"
version = "0.3.17"

    [deps.Libxc.extensions]
    LibxcCudaExt = "CUDA"

    [deps.Libxc.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[[deps.Libxc_GPU_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "ee321f68686361802f2ddb978dae441a024e61ea"
uuid = "25af9330-9b41-55d4-a324-1a83c0a0a1ac"
version = "6.1.0+2"

[[deps.Libxc_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c5516f2b1655a103225e69477e3df009347580df"
uuid = "a56a6d9d-ad03-58af-ab61-878bf78270d6"
version = "6.1.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9df2ab050ffefe870a09c7b6afdb0cde381703f2"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.11.1"
weakdeps = ["ChainRulesCore", "SparseArrays", "Statistics"]

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"
    LinearMapsSparseArraysExt = "SparseArrays"
    LinearMapsStatisticsExt = "Statistics"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "0f5648fbae0d015e3abe5867bca2b362f67a5894"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.166"
weakdeps = ["ChainRulesCore", "ForwardDiff", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "eb006abbd7041c28e0d16260e50a24f8f9104913"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.2.0+0"

[[deps.MPI]]
deps = ["Distributed", "DocStringExtensions", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "PkgVersion", "PrecompileTools", "Requires", "Serialization", "Sockets"]
git-tree-sha1 = "b4d8707e42b693720b54f0b3434abee6dd4d947a"
uuid = "da04e1cc-30fd-572f-bb4f-1f8673147195"
version = "0.20.16"

    [deps.MPI.extensions]
    AMDGPUExt = "AMDGPU"
    CUDAExt = "CUDA"

    [deps.MPI.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "8a5b4d2220377d1ece13f49438d71ad20cf1ba83"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.1.2+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "8f6af051b9e8ec597fa09d8885ed79fd582f33c9"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.10"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "6979eccb6a9edbbb62681e158443e79ecc0d056a"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.3.1+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measurements]]
deps = ["Calculus", "LinearAlgebra", "Printf", "Requires"]
git-tree-sha1 = "bdcde8ec04ca84aef5b124a17684bf3b302de00e"
uuid = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
version = "2.11.0"

    [deps.Measurements.extensions]
    MeasurementsBaseTypeExt = "BaseType"
    MeasurementsJunoExt = "Juno"
    MeasurementsRecipesBaseExt = "RecipesBase"
    MeasurementsSpecialFunctionsExt = "SpecialFunctions"
    MeasurementsUnitfulExt = "Unitful"

    [deps.Measurements.weakdeps]
    BaseType = "7fbed51b-1ef5-4d67-9085-a4a9b26f478c"
    Juno = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b01beb91d20b0d1312a9471a36017b5b339d26de"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+1"

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

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "PMIx_jll", "TOML", "Zlib_jll", "libevent_jll", "prrte_jll"]
git-tree-sha1 = "694458ae803b684f09c07f90459cb79655fb377d"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.0+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc6e1927ac521b659af340e0ca45828a3ffc748f"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.12+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "01f85d9269b13fedc61e63cc72ee2213565f7a72"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.8"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PMIx_jll]]
deps = ["Artifacts", "Hwloc_jll", "JLLWrappers", "Libdl", "Zlib_jll", "libevent_jll"]
git-tree-sha1 = "8b3b19351fa24791f94d7ae85faf845ca1362541"
uuid = "32165bc3-0280-59bc-8c0b-c33b6203efab"
version = "4.2.7+0"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a935806434c9d4c506ba941871b327b96d41f2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.0"

[[deps.PeriodicTable]]
deps = ["Base64", "Test", "Unitful"]
git-tree-sha1 = "9a9731f346797126271405971dfdf4709947718b"
uuid = "7b2266bf-644c-5ea3-82d8-af4bbd25a884"
version = "1.1.4"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "542de5acb35585afcf202a6d3361b430bc1c3fbd"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.13"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "bd7c69c7f7173097e7b5e1be07cee2b8b7447f51"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.54"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase", "Setfield", "SparseArrays"]
git-tree-sha1 = "5a95b69396b77fdb2c48970a535610c4743810e2"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.5"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

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

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "1d05623b5952aed1307bf8b43bec8b8d1ef94b6e"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.5"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "00099623ffee15972c16111bcf84c58a0051257c"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.9.0"

[[deps.PseudoPotentialIO]]
deps = ["EzXML", "LinearAlgebra"]
git-tree-sha1 = "88cf9598d70015889c99920ff3dacca0eb26ae90"
uuid = "cb339c56-07fa-4cb2-923a-142469552264"
version = "0.1.1"

[[deps.Qhull_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be2449911f4d6cfddacdf7efc895eceda3eee5c1"
uuid = "784f63db-0788-585a-bace-daefebcd302b"
version = "8.0.1003+0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "a38e7d70267283888bc83911626961f0b8d5966f"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.9"

[[deps.Roots]]
deps = ["ChainRulesCore", "CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "0f1d92463a020321983d04c110f476c274bafe2e"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.22"

    [deps.Roots.extensions]
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"

    [deps.Roots.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "3aac6d68c5e57449f5b9b865c9ba50ac2970c4cf"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.42"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.Spglib]]
deps = ["CrystallographyCore", "StaticArrays", "StructEquality", "spglib_jll"]
git-tree-sha1 = "8ad7ae954fe3731ac997a7da33a14256ab499a05"
uuid = "f761d5c5-86db-4880-b97f-9680a7cccfb5"
version = "0.8.5"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "f295e0a1da4ca425659c57441bcb59abb035a4bc"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.8"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "03fec6800a986d191f64f5c0996b59ed526eda25"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.4.1"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "5ef59aea6f18c25168842bded46b16662141ab87"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.7.0"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StructEquality]]
deps = ["Compat"]
git-tree-sha1 = "192a9f1de3cfef80ab1a4ba7b150bb0e11ceedcf"
uuid = "6ec83bb0-ed9f-11e9-3b4c-2b04cb4e219c"
version = "2.1.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "242982d62ff0d1671e9029b52743062739255c7e"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.18.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulAtomic]]
deps = ["Unitful"]
git-tree-sha1 = "903be579194534af1c4b4778d1ace676ca042238"
uuid = "a7773ee8-282e-5fa2-be4e-bd808c38a91a"
version = "1.0.0"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "b182207d4af54ac64cbc71797765068fdeff475d"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.64"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "5f24e158cf4cee437052371455fe361f526da062"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.6"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "da69178aacc095066bad1f69d2f59a60a1dd8ad1"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.0+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522b8414d40c4cbbab8dee346ac3a09f9768f25d"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.5+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "47cf33e62e138b920039e8ff9f9841aafe1b733e"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.35.1+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libevent_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "OpenSSL_jll"]
git-tree-sha1 = "f04ec6d9a186115fb38f858f05c0c4e1b7fc9dcb"
uuid = "1080aeaf-3a6a-583e-a51c-c537b09f60ec"
version = "2.1.13+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.prrte_jll]]
deps = ["Artifacts", "Hwloc_jll", "JLLWrappers", "Libdl", "PMIx_jll", "libevent_jll"]
git-tree-sha1 = "5adb2d7a18a30280feb66cad6f1a1dfdca2dc7b0"
uuid = "eb928a42-fffd-568d-ab9c-3f5d54fc65b9"
version = "3.0.2+0"

[[deps.spglib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "bc328924cf4975fe49e6416f7e1622e8ceda55e8"
uuid = "ac4a9f1e-bdb2-5204-990c-47c8b2f70d4e"
version = "2.1.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─4b7ade30-8c59-11ee-188d-c3d4588f7106
# ╟─35066ab2-d52d-4a7f-9170-abbae6216996
# ╠═a9548adc-cc71-4f26-96d2-7aefc39a554f
# ╟─f09f563c-e458-480f-92be-355ab4582fb5
# ╟─363b4496-5728-4eea-a3cc-4a090952155c
# ╟─11ca4a8e-2f55-40ea-b9cc-37ba7806bb5c
# ╟─354b8072-dcf0-4182-9897-c3e9534bef5a
# ╟─0c00fca4-41b4-4c1d-b4b1-d6668c42fe65
# ╟─ef2796f3-b6d4-4403-aaa6-ed3d9d541a3a
# ╟─40f62be2-8394-4886-84e8-d595b6ff7cab
# ╠═e06ad336-9ff8-48bb-9eaf-4a606d69d51c
# ╠═05d40b5e-fd83-4e73-8c78-dfd986a42fc0
# ╠═c4393902-7c57-4126-80af-8765bea42ebd
# ╠═78cc8d4a-cb63-48d0-a2f9-b8ec8c2950e5
# ╟─0aaeb747-ce4f-4d53-a3b7-f8019c3d688b
# ╠═ffd75a52-9741-4c12-ae2f-303db8d70332
# ╟─0f6a001d-ce82-4d4b-a914-37bf0b62d604
# ╠═66b93963-272f-4e1b-827c-0f3b551ee52b
# ╟─dfe6be2f-fb6b-458b-8016-8d89dfa66ed9
# ╠═252e505d-a43d-472a-9fff-d529c3a2eeb7
# ╟─0f5d7c6c-28d7-4cf9-a279-8117ddf7d9d5
# ╠═b4ffe212-6f71-4d61-989a-570c5448877a
# ╟─b9de31a7-1e75-47ca-9901-f2a5ee3a5630
# ╠═c3a6aec7-ab22-4017-8e2f-bfe5696021dd
# ╟─8f3634ca-9c13-4d40-af3c-0c80588fd8c8
# ╠═ac0cffd2-effb-489c-a118-60864798d55e
# ╠═c50b9d2e-dcf5-42be-b401-3da8c514165b
# ╟─142ac96e-bd9d-45a7-ad31-e187f28e884a
# ╠═601c837c-1615-4826-938c-b39bb35f46d1
# ╟─e0a07aca-f81a-436b-b11e-8446120e0235
# ╠═2bf7d1a6-cd49-41ee-a670-4352febd02b6
# ╠═3856b78c-cadf-4e6c-868b-8026ab0bb670
# ╠═0f70ff13-ffb0-4b9a-8570-7e324ff4afd7
# ╠═f41b96ae-51a6-4aae-b9d2-ef0c546f2999
# ╠═189d4007-8e6f-4f54-9828-005febe2b89c
# ╠═6bfa0a42-a0e6-4bef-9dc0-0023796f087b
# ╠═682f52e7-f7a9-4146-b789-268672049fa9
# ╟─58856ccd-3a1c-4a5f-9bd2-159be331f07c
# ╟─e04087db-9973-4fad-a964-20d109fff335
# ╟─abdfcc43-245d-425a-809e-d668f03e9b45
# ╠═3e13955c-b471-466a-b416-5c4b9c874584
# ╟─d26fec73-4416-4a20-bdf3-3a4c8ea533d1
# ╠═e42ff75d-f401-4b15-9421-81b24277e01b
# ╠═983b1a76-ecb9-42cd-a109-d8847d78ca79
# ╠═faa100bb-4012-49ab-929d-303a26fa0634
# ╟─9a953b4e-2eab-4cb6-8b12-afe754640e22
# ╟─0616cc6b-c5f8-4d83-a247-849a3d8c5de8
# ╠═d363fa0a-d48b-4904-9337-098bcef015bb
# ╟─9e7d9b0f-f29d-468f-8f7c-e01f271d57c6
# ╠═d34fa8e5-903f-45f4-97f8-32abd8883e9f
# ╠═e3244165-9f04-4676-ae80-57667a6cb8d5
# ╟─9bc1e1b6-7419-4b8b-aebc-a90b076e87da
# ╠═61448d06-c628-493b-8fa0-a8f0a41acd1d
# ╟─722f34dd-e1cb-4a3c-9a13-7ff6117029fc
# ╠═06270fe2-df78-431a-8433-d69fc89ae5c3
# ╟─e56b2164-5f66-4745-b1a5-711dcc1a324b
# ╠═66992c0d-262c-4211-b5d2-1ba247f9e8ba
# ╟─86f70a52-13ce-42b5-9a28-2c01d92b022d
# ╠═2dee9a6b-c5dd-4511-a9e3-1d5bc33db946
# ╟─da8c23eb-d473-4c2f-86f9-879016659f3e
# ╠═3006a991-d017-40b3-b52b-31438c05d088
# ╠═8ad78ab5-fb3b-4a0c-a21d-fbddbf3d3f8c
# ╠═6bda48f6-a89a-4827-9ba9-b20d00e27761
# ╠═e2274211-47d9-4751-92e4-fbd2839a4b2c
# ╠═f3a8531e-cad3-4575-9944-52d2cade1f76
# ╠═2c62a65c-5b8c-44cc-a028-5860f5119e8c
# ╠═b83e56f8-a339-4f53-a765-3f1295145287
# ╠═8aa3159d-4845-463a-a1d2-336ec8f6efa6
# ╟─646242a2-8df6-4caf-81dc-933345490e6c
# ╟─d7f3cadf-6161-41ce-8a7a-20fba5182cfb
# ╟─f6efea9b-9656-4337-82a1-5b894e078338
# ╠═7121fcd7-c9ec-4d59-ad26-e8b3887ba65e
# ╠═c7c3f1ab-def9-4de9-8fc7-112be4c3631d
# ╠═cda35232-b0ef-490f-9be2-b5a2c84d4ad6
# ╠═1738ca7b-5d33-484c-a37f-eb2f70086a64
# ╠═3bff8455-bb2d-4acf-b098-268d57c101d5
# ╟─14df1c0f-1c4f-4da9-be49-3941b9c12fd3
# ╟─3fc07beb-33c1-43b3-9d66-27693d78e46a
# ╟─047d630b-e85e-45e9-9574-758955cb160e
# ╟─fc040eb6-872b-475d-a6cd-7d3ad1fae229
# ╟─74df4d8b-0345-449e-ad3a-ded44a94a40d
# ╟─48ffb85e-884d-46a4-8184-40126b603aac
# ╟─33896109-d190-4992-806a-c447ca36071b
# ╠═93552ccd-f7b5-4830-a9ad-417d3fca9af9
# ╟─968a26f2-12fe-446f-aa41-977fdffc23a0
# ╠═7ebc18b0-618f-4d99-881f-7c30eb3bc7f5
# ╟─bf00ec9d-5c69-43c3-87c1-28ac9f5b4c9f
# ╟─38db50ac-5937-4e9b-948c-fd93ced44cb2
# ╟─56c10e5c-90e0-4aa7-a343-38aadff37693
# ╟─7fb27a85-27eb-4111-800e-a8c306ea0f18
# ╠═e0f916f6-ce7f-48b1-8256-2e6a6247e171
# ╟─62fd73e1-bd11-465f-8032-e87db00bfda7
# ╠═3dab1ef5-ba74-4b22-89c6-8fea651fd959
# ╠═90ab127d-9054-4a0f-88ba-8a340fa3afcc
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
