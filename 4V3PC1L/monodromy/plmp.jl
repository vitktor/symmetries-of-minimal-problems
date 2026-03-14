using DecomposingPolynomialSystems
using LinearAlgebra
include("../../utils.jl")


function generate_equations()
    @var R[1:3,1:3,1:3] t[1:3,1:3] l[1:3,1:5,1:4] a[1:4]
    eqs = vcat([vcat((R[:,:,i]'*R[:,:,i] - I)[:], [det(R[:,:,i]) - 1]) for i in 1:3]...)
    Ps = vcat([[I zeros(3)]], [[R[:,:,i] t[:,i]] for i in 1:3])
    line_inc = [[1,2],[1,3],[1,4]]
    for inc in line_inc
        M₁ = hcat([transpose(Ps[i])*l[:,inc[1],i] for i in 1:4]...)
        M₂ = hcat([transpose(Ps[i])*l[:,inc[2],i] for i in 1:4]...)
        M = hcat(M₁, M₂)
        append!(eqs, compute_minors(M, 4))
    end
    M = hcat([transpose(Ps[i])*l[:,5,i] for i in 1:4]...)
    append!(eqs, compute_minors(M, 3))
    append!(eqs, a'*[t[:,1]; 1]) # remove scale ambiguity in the world coordinate system

    return ParametricSystem(eqs; unknowns = vcat(R[:], t[:]), parameters = vcat(l[:], a))
end


function fabricateSample()
    Rs = vcat([Matrix{ComplexF64}(I(3))], [c2R(randn(ComplexF64, 3)) for _ in 1:3])
    R = cat(Rs[2:end]..., dims = 3)
    ts = vcat([zeros(ComplexF64, 3)], [randn(ComplexF64, 3) for _ in 1:3])
    t = hcat(ts[2:end]...)
    X = randn(ComplexF64, 3, 3) # 3 random points
    s = randn(ComplexF64)
    X[:,3] = s*X[:,1] + (1-s)*X[:,2] # 3rd point is on the line of the first two
    xs = [[R t]*a2p(X) for (R, t) in zip(Rs, ts)]
    l = zeros(ComplexF64, 3, 5, 4)
    for i in 1:4
        l[:,1,i] = cross(xs[i][:,1], xs[i][:,2])
        for j in 2:4
            l[:,j,i] = random_line_through_point(xs[i][:,j-1])
        end
    end
    Y = randn(ComplexF64, 3, 2) # 2 points on the 3D line
    ys = [[R t]*a2p(Y) for (R, t) in zip(Rs, ts)]
    for i in 1:4
        l[:,5,i] = cross(ys[i][:,1], ys[i][:,2])
    end
    a = dehomogenization_coeffs(ts[2])
    return (vcat(R[:], t[:]), vcat(l[:], a))
end


F = generate_equations()
x₀, p₀ = fabricateSample()
norm(F(x₀, p₀))

F = run_monodromy(F, ([x₀], p₀), max_loops_no_progress=1)
D = aut_permutations(F)