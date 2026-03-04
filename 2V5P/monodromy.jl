using DecomposingPolynomialSystems
using LinearAlgebra
include("../utils.jl")


@var R[1:3,1:3] t[1:3] α[1:5,1:2] x[1:3,1:5,1:2] a[1:4]
eqs = vcat([vcat((R'*R - I)[:], [det(R) - 1]) for i in 1:2]...)
X = [α[i,1]*x[:,i,1] for i in 1:5] # 5 world points
for i in 1:5 # points
    append!(eqs, α[i,2]*x[:,i,2] - R*X[i] - t)
end
push!(eqs, a'*[t[:,1]; 1]) # remove scale ambiguity in the world coordinate system

F = ParametricSystem(eqs; unknowns = vcat(R[:], t[:], α[:]), parameters = vcat(x[:], a))


function fabricateSample()
    Rs = vcat([Matrix{ComplexF64}(I(3))], [c2R(randn(ComplexF64, 3)) for _ in 1:1])
    R = cat(Rs[2:end]..., dims = 3)
    println(size(R))
    ts = vcat([zeros(ComplexF64, 3)], [randn(ComplexF64, 3) for _ in 1:1])
    t = hcat(ts[2:end]...)
    println(size(t))
    X = randn(ComplexF64, 3, 5) # 5 random points
    α = randn(ComplexF64, 5, 2)
    A₁, A₂ = [diagm(α[:,i]) for i in 1:2]
    xs = [[R t]*a2p(X) for (R, t) in zip(Rs, ts)]
    xs = [xs[1]*inv(A₁), xs[2]*inv(A₂)] # apply the inverse of the projective transformation
    x = cat(xs..., dims = 3)
    println(size(x))
    a = dehomogenization_coeffs(ts[2])
    return (vcat(R[:], t[:], α[:]), vcat(x[:], a))
end

x₀, p₀ = fabricateSample()
evals = F(x₀, p₀)
norm(F(x₀, p₀))

F = run_monodromy(F, ([x₀], p₀), max_loops_no_progress=20)