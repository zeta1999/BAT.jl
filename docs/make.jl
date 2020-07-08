@info("BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN")
println("BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN")

function all_lteq(A::AbstractArray, B::AbstractArray, C::AbstractArray)
    axes(A) == axes(B) == axes(C) || throw(DimensionMismatch("A, B and C must have the same indices"))
    result::Int = 0
    for i in eachindex(A, B, C)
        result += ifelse(A[i] <= B[i] <= C[i], 1, 0)
    end
    result == length(eachindex(A))
end

A = Float32[-1.0, -1.0]
B = [0.0, 0.0]
C = Float32[1.0, 1.0]
all_lteq(A, B, C)

@info("DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE")
println("DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE")
