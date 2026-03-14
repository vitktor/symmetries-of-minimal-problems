using LinearAlgebra

function compute_normal(X1, X2, d, m, C)
    # Direction of line ℓ
    n = X2 - X1

    # Point on Plücker line L
    pL = cross(d, m) / dot(d, d)

    # Parameter of intersection with plane through C normal to ℓ
    denom = dot(n, d)
    if abs(denom) < 1e-10
        error("Line L is parallel to the plane.")
    end

    t = dot(n, C - pL) / denom

    # Intersection point
    P = pL + t * d

    # Vector from intersection to C
    return C - P
end

X1, X2, d, m, C = rand(3), rand(3), rand(3), rand(3), rand(3)
n = compute_normal(X1, X2, d, m, C)

# Construct Vandermonde matrix for given monomials, variables, samples, and evals
function vandermonde_matrix(mons, vars, samples, evals)
    k = 2 * length(mons)
    A = zeros(ComplexF64, k, k)
    for i in 1:k
        for (j, mon) in enumerate(mons)
            A[i, j] = mon(vars => samples[:, i])
        end
        for (j, mon) in enumerate(mons)
            A[i, j + length(mons)] = -evals[i] * mon(vars => samples[:, i])
        end
    end
    return A
end

using DynamicPolynomials
@polyvar X[1:3,1:2] d[1:3] m[1:3] C[1:3]
vars = vcat(X[:], d[:], m[:], C[:])
mons = ∪([monomials(vars, j) for j in 1:4]...)

k = 2*length(mons)
samples = randn(15, k)
evals = [compute_normal(samples[:, i][1:3], samples[:, i][4:6], samples[:, i][7:9], samples[:, i][10:12], samples[:, i][13:15])[1] for i in 1:k]

A = vandermonde_matrix(mons, vars, samples, evals)
nullspace(A)