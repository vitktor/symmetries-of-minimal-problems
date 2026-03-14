using LinearAlgebra


function plucker_from_point_direction(X::AbstractVector, d::AbstractVector)
    # Input checks
    if length(X) != 3 || length(d) != 3
        error("X and d must be 3D vectors")
    end

    # Moment vector
    m = cross(X, X + d)

    return (d, m)
end

# Example usage
X = [1.0, 2.0, 3.0]
d  = [0.0, 1.0, 0.0]

d_plucker, m_plucker = plucker_from_point_direction(X, d)

println("Direction d: ", d_plucker)
println("Moment m: ", m_plucker)




# --- Method 1: project two points ---
function image_line_from_point_direction(P::AbstractMatrix,
                                         X::AbstractVector,
                                         d::AbstractVector)
    # Two points on the line
    X1_h = [X; 1.0]         # homogeneous 3D point
    X2_h = [X + d; 1.0]     # second point along the direction

    # Project points into image
    x1 = P * X1_h
    x2 = P * X2_h

    # Compute image line as cross product
    l = cross(x1, x2)

    return l
end

# --- Method 2: from Plücker coordinates directly ---
function image_line_from_plucker(P::AbstractMatrix,
                                 d::AbstractVector,
                                 m::AbstractVector)
    # Convert to homogeneous 4-vectors
    d_h = [d; 0.0]  # direction as point at infinity
    X_h = [cross(d, m); sum(d.*d)]  # moment as homogeneous point

    # Project into the image
    pd = P * d_h
    pm = P * X_h

    # Cross product to get the line in P^2
    l = cross(pd, pm)

    return l
end

# --- Example Usage ---
# Camera projection matrix
P = [500.0 0.0 320.0 1.0;
     0.0 500.0 240.0 2.0;
     0.0 0.0 1.0 3.0]

# 3D point and direction
X = randn(ComplexF64, 3)  # random 3D point
d = randn(ComplexF64, 3)  # random 3D direction

# Method 1
l1 = image_line_from_point_direction(P, X, d)
println("Image line (method 1): ", l1)

# Method 2: Plücker coordinates
d, m = plucker_from_point_direction(X, d)
l2 = image_line_from_plucker(P, d, m)
println("Image line (method 2): ", l2)

cross(l1, l2)