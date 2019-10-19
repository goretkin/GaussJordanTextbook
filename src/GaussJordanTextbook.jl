module GaussJordanTextbook

using LinearAlgebra
"""
Square matrix
multiply on the left to permute rows.
multiply on the right to permute columns
"""
struct PermutationMatrix{P} <: AbstractArray{Bool, 2}
  # if there is a 1 and (i,j), then row[i] = j
  row_ij::P
end

Base.size(a::PermutationMatrix) = let n = length(a.row_ij); (n, n); end
Base.getindex(a::PermutationMatrix, i::Integer, j::Integer) = (a.row_ij[i] == j)
# TODO define one-hot vectors when getindex gets `:`
# TODO define 1-d index

# construct identity
PermutationMatrix(n::Int64) = PermutationMatrix(1:n)

function PermutationMatrix(a::AbstractArray{T, 2}) where {T}
  (n, m) = size(a)
  @assert n == m
  p = Int64[]
  for row_i = 1:n
    row = a[row_i, :] # as a 1D vector
    js = findall(row .== 1)
    j = js[] # error if not permutation
    push!(p, j)
  end
  return PermutationMatrix(p)
end

# TODO: O(N^3) when it could be O(N), but uses built-in definition of matrix multiplication
Base.:*(a::PermutationMatrix, b::PermutationMatrix) = PermutationMatrix(collect(a) * collect(b))

"""
multiply on the left to do
A[mutate_row,:] = A[mutate_row,:] - A[neg_row, :]
"""
# TODO T <: Signed
struct SubtractScaledRow{T} <: AbstractArray{T, 2}
  mutate_row::Int64
  scale::T
  neg_row::Int64
  n::Int64
end

Base.size(a::SubtractScaledRow) = (a.n, a.n)
function Base.getindex(a::SubtractScaledRow, i::Integer, j::Integer)
  sub = (i==a.mutate_row) * ( -a.scale * (j==a.neg_row))
  id = (i==j)
  sub + id
end

UnitVector(i, n) = Bool[j==i for j in 1:n]
ScaleRow(s::T, i, n) where T = Diagonal(T[if j==i; s; else; 1; end for j in 1:n])

function gauss_jordan!(A)
  (n, m) = size(A)
  L = one(A)
  U = A # for suggesting naming.
  for pivot_i = 1:n
    pivot = A[pivot_i, pivot_i]
    inv_pivot = inv(pivot)

    for row_below_pivot_i = pivot_i+1:n
      sr = SubtractScaledRow(row_below_pivot_i, A[row_below_pivot_i, pivot_i] * inv_pivot, pivot_i, n)
      operation = sr
      # notice that invariant LU = A is preserved with this update
      U = operation * U
      L = L * inv(collect(operation))
    end

  end

  # inv(bookkeep) * A = Astart
  return L, U
end

end # module
