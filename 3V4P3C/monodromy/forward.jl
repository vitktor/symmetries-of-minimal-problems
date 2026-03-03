using DecomposingPolynomialSystems
using LinearAlgebra
include("../../utils.jl")


function generate_equations()
    @var R[1:3,1:3,1:2] t[1:3,1:2] α[1:4,1:3] x[1:3,1:3,1:3] s[1:2,1:3] a[1:4]
    eqs = vcat([vcat((R[:,:,i]'*R[:,:,i] - I)[:], [det(R[:,:,i]) - 1]) for i in 1:2]...)
    xs = Array{Expression}(undef, 3, 4, 3)
    xs[:,1:3,:] = x
    for i in 1:3
        xs[:,4,i] = x[:,2:3,i]*s[:,i]
    end
    X = [α[i,1]*xs[:,i,1] for i in 1:4] # 4 world points
    for i in 1:4 # points
        for j in 1:2 # cameras
            append!(eqs, α[i,j+1]*xs[:,i,j+1] - R[:,:,j]*X[i] - t[:,j])
        end
    end
    append!(eqs, cross(X[2]-X[3], X[3]-X[4])) # 3 points are collinear
    push!(eqs, a'*[t[:,1]; 1]) # remove scale ambiguity in the world coordinate system

    return ParametricSystem(eqs; unknowns = vcat(R[:], t[:], α[:]), parameters = vcat(x[:], s[:], a))
end


function fabricateSample()
    Rs = vcat([Matrix{ComplexF64}(I(3))], [c2R(randn(ComplexF64, 3)) for _ in 1:2])
    R = cat(Rs[2:end]..., dims = 3)
    ts = vcat([zeros(ComplexF64, 3)], [randn(ComplexF64, 3) for _ in 1:2])
    t = hcat(ts[2:end]...)
    X = randn(ComplexF64, 3, 4) # 4 random points
    s = randn(ComplexF64)
    X[:,4] = s*X[:,2] + (1-s)*X[:,3] # 4th point is on the line of the 2nd and 3rd
    α = randn(ComplexF64, 4, 3)
    A₁, A₂, A₃ = [diagm(α[:,i]) for i in 1:3]
    xs = [[R t]*a2p(X) for (R, t) in zip(Rs, ts)]
    xs = [xs[1]*inv(A₁), xs[2]*inv(A₂), xs[3]*inv(A₃)] # apply the inverse of the projective transformation
    s = Array{ComplexF64}(undef, 2, 3)
    for i in 1:3
        s[:,i] = linear_comb(xs[i][:,4], xs[i][:,2:3])
    end
    x = cat([xs[i][:,1:3] for i in 1:3]..., dims = 3)
    a = dehomogenization_coeffs(ts[2])
    return (vcat(R[:], t[:], α[:]), vcat(x[:], s[:], a))
end