using LinearAlgebra, Combinatorics

"""
Quaternion to scaled rotation matrix.
"""
function q2sR(q)
    a,b,c,d = q
    return [a^2+b^2-c^2-d^2 2*(b*c-a*d) 2*(b*d+a*c);
            2*(b*c+a*d) a^2-b^2+c^2-d^2 2*(c*d-a*b);
            2*(b*d-a*c) 2*(c*d+a*b) a^2-b^2-c^2+d^2]
end

"""
Quaternion to rotation matrix.
"""
function q2R(q)
    a,b,c,d = q
    return 1/(a^2+b^2+c^2+d^2)*q2sR(q)
end

"""
Cayley vector to scaled rotation matrix.
"""
function c2sR(c)
    x, y, z = c
    return [1+x^2-y^2-z^2 2*(x*y-z) 2*(y+x*z);
            2*(x*y+z) 1-x^2+y^2-z^2 2*(y*z-x);
            2*(x*z-y) 2*(x+y*z) 1-x^2-y^2+z^2]
end

"""
Cayley vector to rotation matrix.
"""
function c2R(c)
    x, y, z = c
    return 1/(1+x^2+y^2+z^2)*c2sR(c)
end

"""
Quaternion to Cayley vector.
"""
q2c(q) = [q[2]/q[1], q[3]/q[1], q[4]/q[1]]

"""
Rotation matrix to Cayley vector.
"""
R2c(R) = 1/(1+tr(R))*xx2v((R-transpose(R)))

function dehomogenization_coeffs(x)
    n = nullspace(reshape([x; 1], 1, length(x)+1))
    a = vec(n*randn(ComplexF64, size(n, 2)))
    return a
end


function compute_minors(A::AbstractMatrix, d::Int)
    m, n = size(A)
    if d > m || d > n
        error("d must be ≤ number of rows and columns of A")
    end
    
    row_combos = collect(combinations(1:m, d))
    col_combos = collect(combinations(1:n, d))
    
    minors = []
    
    for r in row_combos
        for c in col_combos
            push!(minors, det(A[r, c]))
        end
    end
    
    return minors
end

function random_line_through_point(x)
    n = nullspace(reshape(x, 1, length(x)))
    return vec(n*randn(ComplexF64, size(n, 2)))
end

function linear_comb(x, xs)
    A = hcat(xs, x)
    n = nullspace(A)
    v = vec(n*randn(ComplexF64, size(n, 2)))
    v = v / v[end] # normalize so that the coefficient of x is 1
    return -v[1:end-1]
end