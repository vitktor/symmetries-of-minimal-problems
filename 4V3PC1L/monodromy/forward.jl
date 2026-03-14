using DecomposingPolynomialSystems
using LinearAlgebra
include("../../utils.jl")


function generate_equations()
    @var R[1:3,1:3,1:3] t[1:3,1:3] α[1:3,1:4] x[1:3,1:2,1:4] s[1:2,1:4] l[1:3,1:4] d[1:3] m[1:3] a[1:4] b[1:4]
    eqs = vcat([vcat((R[:,:,i]'*R[:,:,i] - I)[:], [det(R[:,:,i]) - 1]) for i in 1:3]...)
    xs = Array{Expression}(undef, 3, 3, 4)
    xs[:,1:2,:] = x
    for i in 1:4
        xs[:,3,i] = x[:,1:2,i]*s[:,i]
    end
    X = [α[i,1]*xs[:,i,1] for i in 1:3] # 3 world points
    for i in 1:3 # points
        for j in 1:3 # cameras
            append!(eqs, α[i,j+1]*xs[:,i,j+1] - R[:,:,j]*X[i] - t[:,j])
        end
    end
    append!(eqs, cross(X[1]-X[2], X[2]-X[3])) # 3 points are collinear
    push!(eqs, d'*m) # plucker constraint
    Ps = vcat([[I zeros(3)]], [[R[:,:,i] t[:,i]] for i in 1:3])
    for j in 1:4 # cameras
        append!(eqs, l[:,j]'*Ps[j]*[cross(d,m); d'*d])
        append!(eqs, l[:,j]'*Ps[j]*[d; 0])
    end
    push!(eqs, b'*[d; 1]) # remove scale ambiguity in the Plücker coordinates
    push!(eqs, a'*[t[:,1]; 1]) # remove scale ambiguity in the world coordinate system

    return ParametricSystem(eqs; unknowns = vcat(R[:], t[:], α[:], d, m), parameters = vcat(x[:], s[:], l[:], a, b))
end


function fabricateSample()
    Rs = vcat([Matrix{ComplexF64}(I(3))], [c2R(randn(ComplexF64, 3)) for _ in 1:3])
    R = cat(Rs[2:end]..., dims = 3)
    ts = vcat([zeros(ComplexF64, 3)], [randn(ComplexF64, 3) for _ in 1:3])
    t = hcat(ts[2:end]...)
    X = randn(ComplexF64, 3, 3) # 3 random points
    s = randn(ComplexF64)
    X[:,3] = s*X[:,1] + (1-s)*X[:,2] # 3rd point is on the line of the first two
    α = randn(ComplexF64, 3, 4)
    As = [diagm(α[:,i]) for i in 1:4]
    xs = [[R t]*a2p(X) for (R, t) in zip(Rs, ts)]
    xs = [xs[i]*inv(As[i]) for i in 1:4] # apply the inverse of the projective transformation
    s = Array{ComplexF64}(undef, 2, 4)
    for i in 1:4
        s[:,i] = linear_comb(xs[i][:,3], xs[i][:,1:2])
    end
    x = cat([xs[i][:,1:2] for i in 1:4]..., dims = 3)
    Y = randn(ComplexF64, 3, 2) # 2 points on the line
    d = Y[:,2] - Y[:,1]
    m = cross(Y[:,1], Y[:,2])
    ys = [[R t]*a2p(Y) for (R, t) in zip(Rs, ts)]
    ls = [cross(y[:,1], y[:,2]) for y in ys]
    l = hcat(ls...)
    a = dehomogenization_coeffs(ts[2])
    b = dehomogenization_coeffs(d)
    return (vcat(R[:], t[:], α[:], d, m), vcat(x[:], s[:], l[:], a, b))
end


F = generate_equations()
x₀, p₀ = fabricateSample()
norm(F(x₀, p₀))

F = run_monodromy(F, ([x₀], p₀), max_loops_no_progress=1)
D = aut_permutations(F)