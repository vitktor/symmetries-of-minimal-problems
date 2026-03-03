using DecomposingPolynomialSystems
using LinearAlgebra
include("../../utils.jl")


function generate_equations()
    @var R[1:3,1:3,1:2] t[1:3,1:2] l[1:3,1:6,1:3] a[1:4]
    eqs = vcat([vcat((R[:,:,i]'*R[:,:,i] - I)[:], [det(R[:,:,i]) - 1]) for i in 1:2]...)
    Ps = vcat([[I zeros(3)]], [[R[:,:,i] t[:,i]] for i in 1:2])
    line_inc = [[1,2],[3,4],[3,5],[3,6]]
    for inc in line_inc
        M₁ = hcat([transpose(Ps[i])*l[:,inc[1],i] for i in 1:3]...)
        M₂ = hcat([transpose(Ps[i])*l[:,inc[2],i] for i in 1:3]...)
        M = hcat(M₁, M₂)
        append!(eqs, compute_minors(M, 4))
    end
    append!(eqs, a'*[t[:,1]; 1]) # remove scale ambiguity in the world coordinate system

    return ParametricSystem(eqs; unknowns = vcat(R[:], t[:]), parameters = vcat(l[:], a))
end


function fabricateSample()
    Rs = vcat([Matrix{ComplexF64}(I(3))], [c2R(randn(ComplexF64, 3)) for _ in 1:2])
    R = cat(Rs[2:end]..., dims = 3)
    println(size(R))
    ts = vcat([zeros(ComplexF64, 3)], [randn(ComplexF64, 3) for _ in 1:2])
    t = hcat(ts[2:end]...)
    println(size(t))
    X = randn(ComplexF64, 3, 4) # 4 random points
    s = randn(ComplexF64)
    X[:,4] = s*X[:,2] + (1-s)*X[:,3] # 3rd point is on the line of the first two
    xs = [[R t]*a2p(X) for (R, t) in zip(Rs, ts)]
    l = zeros(ComplexF64, 3, 6, 3)
    for i in 1:3
        for j in 1:2
            l[:,j,i] = random_line_through_point(xs[i][:,1])
        end
        l[:,3,i] = cross(xs[i][:,2], xs[i][:,3])
        for j in 4:6
            l[:,j,i] = random_line_through_point(xs[i][:,j-2])
        end
    end
    a = dehomogenization_coeffs(ts[2])
    return (vcat(R[:], t[:]), vcat(l[:], a))
end

F = generate_equations()
x₀, p₀ = fabricateSample()
norm(F(x₀, p₀))

F = run_monodromy(F, ([x₀], p₀), max_loops_no_progress=1)
