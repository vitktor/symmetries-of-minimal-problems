using DecomposingGroupRepresentations

@polyvar R[1:3,1:3,1:3] t[1:3,1:3] α[1:3,1:4] x[1:3,1:2,1:4] X[1:3,1:3] s[1:2,1:4] l[1:3,1:4] d[1:3] m[1:3] a[1:4] b[1:4]
SO3 = LieGroup("SO", 3)
a₁ = MatrixGroupAction(SO3, vcat(eachrow(R[:,:,1]), eachrow(R[:,:,2]), eachrow(R[:,:,3]), eachcol(x[:,:,1]), eachcol(X), [l[:,1], d, m]))
a₂ = MatrixGroupAction(SO3, vcat(eachcol(R[:,:,1]), [t[:,1]], eachcol(x[:,:,2]), [l[:,2]]))
a₃ = MatrixGroupAction(SO3, vcat(eachcol(R[:,:,2]), [t[:,2]], eachcol(x[:,:,3]), [l[:,3]]))
a₄ = MatrixGroupAction(SO3, vcat(eachcol(R[:,:,3]), [t[:,3]], eachcol(x[:,:,4]), [l[:,4]]))

a₅ = ScalingLieGroupAction(variables=vcat(t[:], α[:], x[:], X[:], s[:], l[:], d, m))
push!(a₅, [1=>vcat(t[:],α[:],X[:])])
push!(a₅, [1=>vcat(d,m)])
for i in 1:4
    push!(a₅, [1=>l[:,i]])
end
for i in 1:3
    for j in 1:4
        if i <= 2
            push!(a₅, [1=>[α[i,j],s[i,j]], -1=>x[:,i,j]])
        else
            push!(a₅, [1=>[α[i,j]], -1=>s[:,j]])
        end
    end
end
a₅

a = a₁ × a₂ × a₃ × a₄ × a₅

vars = vcat(R[:], t[:], α[:], x[:], X[:], s[:], l[:], d, m)
V = VariableSpace(vars)
ρ = GroupRepresentation(a, V)
vars_irrs = irreducibles(ρ)

D = 4
V = BoundedDegreePolynomials(vars, D)
ρ = GroupRepresentation(a, V)
irrs = irreducibles(ρ)
iso = isotypics(irrs)

