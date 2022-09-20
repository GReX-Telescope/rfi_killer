using Test
using RFIKiller

@testset "Statistics" begin
    A = [
        1.0 2 3 4
        5 6 7 8
        9 10 11 12
        13 14 15 16
        17 18 19 20]
    mask = [
        false true true true
        true false true true
        true true false true
        true true true false
        false true true false]
    @test RFIKiller.masked_mean(A, mask, 1)' ≈ sum(A .* mask, dims=1) ./ count.(eachcol(mask))'
    @test RFIKiller.masked_mean(A, mask, 2) ≈ sum(A .* mask, dims=2) ./ count.(eachrow(mask))
    @test RFIKiller.masked_var(A, mask, 1) ≈ [16, 140 / 3, 160 / 3, 16]
    @test RFIKiller.masked_var(A, mask, 2) ≈ [1, 7 / 3, 7 / 3, 1, 0.5]
end