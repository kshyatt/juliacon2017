function bit_rep(element::Integer, L::Integer)
    bit_rep = falses(L)
    for site in 1:L
       bit_rep[site] = (element >> (site - 1)) & 1
    end
    return bit_rep
end

function int_rep(element::BitVector, L::Integer)
    int = 1
    for site in 1:L
       int += (element[site] << (site - 1))
    end
    return int
end

function generate_basis(L::Integer)
    basis = fill(falses(L), 2^L)
    for elem in 1:2^L
        basis[elem] = bit_rep(elem-1, L)
    end
    return basis
end

function magnetization(state::Vector, basis)::Float64
    M = 0.
    for (index, element) in enumerate(basis)
        element_M = 0.
        for spin in element
            element_M += (state[index]^2 * (spin ? 1 : -1))/length(element)
        end
        @assert abs(element_M) <= 1
        M += abs(element_M)
    end
    return M
end

abstract type AbstractHamiltonian{T, S<:AbstractMatrix} <: AbstractArray{T,2} end
# Hamiltonians may be real or complex

type TransverseFieldIsing{Tv, S<:AbstractMatrix} <: AbstractHamiltonian{Tv, S}
    L::Int
    basis::Vector # may be a vector of BitVectors/BitMatrices/Vectors/Matrices
    Mat::S #this may be sparse, or Diagonal, or something else
    h::Real
    function TransverseFieldIsing{Tv, S}(L::Integer, h::Real=0.) where Tv where S <: Matrix
        basis = generate_basis(L)
        H = zeros(2^L, 2^L)
        for (index, element) in enumerate(basis)
            # the diagonal part is easy
            diag_term = 0.
            for site in 1:L-1
                diag_term -= !xor(element[site], element[site+1])
            end
            H[index, index] = diag_term
            # off diagonal part
            for site in 1:L-1
                mask = falses(L)
                mask[site] = true
                new_element = xor.(element, mask)
                new_index = int_rep(new_element, L)
                H[index, new_index] = -h
            end
        end
        new(L, basis, Hermitian(H), h)
    end
    function TransverseFieldIsing{Tv, S}(L::Integer, h::Real=0.) where Tv where S <: SparseMatrixCSC
        basis = generate_basis(L)
        H_Is = Integer[]
        H_Js = Integer[]
        H_Vs = Tv[]
        for (index, element) in enumerate(basis)
            # the diagonal part is easy
            diag_term = 0.
            for site in 1:L-1
                diag_term -= !xor(element[site], element[site+1])
            end
            push!(H_Is, index)
            push!(H_Js, index)
            push!(H_Vs, diag_term)
            # off diagonal part
            for site in 1:L-1
                mask = falses(L)
                mask[site] = true
                new_element = xor.(element, mask)
                new_index = int_rep(new_element, L)
                push!(H_Is, index)
                push!(H_Js, new_index)
                push!(H_Vs, -h)
            end
        end
        new(L, basis, Hermitian(sparse(H_Is, H_Js, H_Vs, 2^L, 2^L)), h)
    end
end

σˣ = [0 1; 1 0]
σʸ = [0 -im; im 0]
σᶻ = [1 0; 0 -1]
σ⁺ = (σˣ + im*σʸ)/2
σ⁻ = (σˣ - im*σʸ)/2

function σᶻσᶻ(ψ::BitVector, site_i::Int, site_j::Int)
    return 2.0*xor(ψ[site_i], ψ[site_j]) - 1.0
end

function σ⁺σ⁻(ψ::BitVector, basis, site_i::Int, site_j::Int)
    i_bit = ψ[site_i]
    j_bit = ψ[site_j]
    if xor(i_bit, j_bit)
        flipped_ψ = copy(ψ)
        flipped_ψ[site_i] ⊻= true
        flipped_ψ[site_j] ⊻= true
        return findfirst(basis, flipped_ψ)
    else
        return 0
    end
end

type Heisenberg{Tv, S} <: AbstractHamiltonian{Tv, S}
    L::Int
    basis::Vector
    Mat::S
    function Heisenberg{Tv, S}(L::Integer) where Tv where S <: Matrix
        basis = generate_basis(L)
        H = zeros(2^L, 2^L)
        for (index, element) in enumerate(basis)
            # the diagonal part is easy
            diag_term = 0.
            for site in 1:L-1
                diag_term += σᶻσᶻ(element, site, site+1)
            end
            H[index, index] = diag_term
            # off diagonal part
            for site in 1:L-1
                new_index = σ⁺σ⁻(element, basis, site, site+1)
                if new_index > 0
                    H[index, new_index] = -4.0
                end
            end
        end
        new(L, basis, Hermitian(H))
    end
    function Heisenberg{Tv, S}(L::Integer) where Tv where S <: SparseMatrixCSC
        basis = generate_basis(L)
        H_Is = Integer[]
        H_Js = Integer[]
        H_Vs = Tv[]
        for (index, element) in enumerate(basis)
            # the diagonal part is easy
            diag_term = 0.
            for site in 1:L-1
                diag_term += σᶻσᶻ(element, site, site+1)
            end
            push!(H_Is, index)
            push!(H_Js, index)
            push!(H_Vs, diag_term)
            # off diagonal part
            for site in 1:L-1
                new_index = σ⁺σ⁻(element, basis, site, site+1)
                if new_index > 0
                    push!(H_Is, index)
                    push!(H_Js, new_index)
                    push!(H_Vs, -4.0)
                end
            end
        end
        new(L, basis, Hermitian(sparse(H_Is, H_Js, H_Vs, 2^L, 2^L)))
    end
end

type XXZ{Tv, S<:AbstractMatrix} <: AbstractHamiltonian{Tv, S}
    L::Int
    basis::Vector
    Mat::S
    Δ::Tv
    function XXZ{Tv, S}(L::Integer, Δ::Tv) where Tv where S <: Matrix
        basis = generate_basis(L)
        H = zeros(2^L, 2^L)
        for (index, element) in enumerate(basis)
            # the diagonal part is easy
            diag_term = 0.
            for site in 1:L-1
                diag_term += Δ*σᶻσᶻ(element, site, site+1)
            end
            H[index, index] = diag_term
            # off diagonal part
            for site in 1:L-1
                new_index = σ⁺σ⁻(element, basis, site, site+1)
                if new_index > 0
                    H[index, new_index] = -4.0
                end
            end
        end
        new(L, basis, Hermitian(H), Δ)
    end
    function XXZ{Tv, S}(L::Integer, Δ::Tv) where Tv where S <: SparseMatrixCSC
        basis = generate_basis(L)
        H_Is = Integer[]
        H_Js = Integer[]
        H_Vs = Tv[]
        for (index, element) in enumerate(basis)
            # the diagonal part is easy
            diag_term = 0.
            for site in 1:L-1
                diag_term += Δ*σᶻσᶻ(element, site, site+1)
            end
            push!(H_Is, index)
            push!(H_Js, index)
            push!(H_Vs, diag_term)
            # off diagonal part
            for site in 1:L-1
                new_index = σ⁺σ⁻(element, basis, site, site+1)
                if new_index > 0
                    push!(H_Is, index)
                    push!(H_Js, new_index)
                    push!(H_Vs, -4.0)
                end
            end
        end
        new(L, basis, Hermitian(sparse(H_Is, H_Js, H_Vs, 2^L, 2^L)), Δ)
    end
end

# A little more fun with types

Base.eigfact(A::TransverseFieldIsing; kwargs...) = eigfact(A.Mat; kwargs...)
Base.eigvals(A::TransverseFieldIsing; kwargs...) = eigvals(A.Mat; kwargs...)
Base.eigvecs(A::TransverseFieldIsing; kwargs...) = eigvecs(A.Mat; kwargs...)
Base.eig(A::TransverseFieldIsing; kwargs...)     = eig(A.Mat; kwargs...)
Base.ishermitian(A::TransverseFieldIsing) = true
Base.issparse(A::TransverseFieldIsing) = issparse(A.Mat)
Base.size(A::TransverseFieldIsing) = size(A.Mat)
Base.size(A::TransverseFieldIsing, dim::Int) = size(A.Mat, dim)
Base.ndims(A::TransverseFieldIsing) = 2
Base.eigfact(A::XXZ; kwargs...) = eigfact(A.Mat; kwargs...)
Base.eigvals(A::XXZ; kwargs...) = eigvals(A.Mat; kwargs...)
Base.eigvecs(A::XXZ; kwargs...) = eigvecs(A.Mat; kwargs...)
Base.eig(A::XXZ; kwargs...)     = eig(A.Mat; kwargs...)
Base.ishermitian(A::XXZ) = true
Base.issparse(A::XXZ) = issparse(A.Mat)
Base.size(A::XXZ) = size(A.Mat)
Base.size(A::XXZ, dim::Int) = size(A.Mat, dim)
Base.ndims(A::XXZ) = 2
Base.eigfact(A::Heisenberg; kwargs...) = eigfact(A.Mat; kwargs...)
Base.eigvals(A::Heisenberg; kwargs...) = eigvals(A.Mat; kwargs...)
Base.eigvecs(A::Heisenberg; kwargs...) = eigvecs(A.Mat; kwargs...)
Base.eig(A::Heisenberg; kwargs...)     = eig(A.Mat; kwargs...)
Base.ishermitian(A::Heisenberg) = true
Base.issparse(A::Heisenberg) = issparse(A.Mat)
Base.size(A::Heisenberg) = size(A.Mat)
Base.size(A::Heisenberg, dim::Int) = size(A.Mat, dim)
Base.ndims(A::Heisenberg) = 2

XXZHam = XXZ{Float64, Matrix}(10, 1.0)
HeisHam = Heisenberg{Float64, Matrix}(10)
@assert eigvals(XXZHam) ≈ eigvals(HeisHam)

function get_groundstate(L::Integer, h::Float64=0.)
    H = TransverseFieldIsing{Float64, Matrix}(L, h)
    return eigvecs(H)[:,1], H
end
