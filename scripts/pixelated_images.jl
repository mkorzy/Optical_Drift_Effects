# scripts/halfPlane_ray_shooting.jl
using Optical_Drift_Effects
using StaticArrays
using LinearAlgebra
using Plots
using Contour  
using Random

# -----------------------------
# Lens set-up
# -----------------------------
# d=1, e=1 
lens = generic_cusp(1, 1)   # <-- change this line to your lens object / parameters 

# ------------------------------
# Source set-up
# ------------------------------
normalize2(v::SVector{2,Float64}) = v / hypot(v[1], v[2])

βstar = SVector{2,Float64}(0.0, 0.0)              # point on boundary line
n     = normalize2(SVector{2,Float64}(0.3, 1.0))  # normal toward bright side
tangent_from_normal(n::SVector{2,Float64}) = SVector{2,Float64}(-n[2], n[1])
w     = 0.02                                      # steepness scale in β-units 

src = HalfPlaneSource(1.0, 0.0, βstar, n, w)

# -----------------------------
# Grid settings
# -----------------------------

os = 20           # oversampling factor (pixels per output pixel, per axis)

Nx_pix, Ny_pix = 25, 25                 # desired output (pixelated) resolution
Nx_hi,  Ny_hi  = os * Nx_pix, os * Ny_pix  # high-res ray grid

xmin, xmax = -2.0, 2.0
ymin, ymax = -2.0, 2.0

# High-res ray grid (pixel-centre sampling)
xs_hi = range(xmin, xmax; length=Nx_hi)
ys_hi = range(ymin, ymax; length=Ny_hi)

# Output pixel centres (for plot axes)
xs_pix = range(xmin, xmax; length=Nx_pix)
ys_pix = range(ymin, ymax; length=Ny_pix)


detJ = zeros(Float64, Ny_hi, Nx_hi)  # lensing jacobian

# ---------------------------------------------
# lensing functions at a point (wrapper for your lens functions)
# ---------------------------------------------             
deflection_at(lens, θ) = deflection(lens, θ)               # returns SVector(αx, αy)
jacobian_at(lens, θ) = deflection_jacobian(lens, θ)        # returns 2x2 (SMatrix ok)

# ---------------------------------------------
# Evaluate on grid
# ---------------------------------------------
for (j, y) in enumerate(ys_hi), (i, x) in enumerate(xs_hi)
    θ = @SVector [x, y]

    J = jacobian_at(lens, θ)
    detJ[i, j] = det(Matrix(J))   
end

# -----------------------------
# Plot: caustic curves (recalculate critical curves (det(J)=0) 
# -----------------------------
# You can add contour lines to p4 to show where det(J)=0, which are the critical curves. For example:

levs = [0.0]
cs = contours(xs_hi, ys_hi, detJ, levs)

critical_polylines = Vector{Vector{SVector{2,Float64}}}()
caustic_polylines = Vector{Vector{SVector{2,Float64}}}()

for lvl in levels(cs)
    for line in lines(lvl)
        xline, yline = coordinates(line)   # <- two vectors
        poly = [@SVector [xline[k], yline[k]] for k in eachindex(xline)]
        push!(critical_polylines, poly)
    end
end

for poly in critical_polylines
    mapped = [θ - deflection_at(lens, θ) for θ in poly]
    push!(caustic_polylines, mapped)
end

# --------------------------------------------
# Image positions for a specific source position β
# ---------------------------------------------
# --- helper: real cube root ---
cbrt_real(x::Real) = sign(x) * abs(x)^(1/3)

# Solve y^3 + p y + q = 0 for real roots
function depressed_cubic_real_roots(p::Float64, q::Float64)
    Δ = (q/2)^2 + (p/3)^3

    if Δ > 0
        # one real root
        u = cbrt_real(-q/2 + sqrt(Δ))
        v = cbrt_real(-q/2 - sqrt(Δ))
        return [u + v]
    elseif abs(Δ) ≤ 1e-14
        # multiple root case (on/near caustic)
        u = cbrt_real(-q/2)
        return [2u, -u]  # (double root at -u)
    else
        # three real roots
        r = 2 * sqrt(-p/3)
        φ = acos( (3q/(2p)) * sqrt(-3/p) )
        return [
            r * cos(φ/3),
            r * cos((φ + 2π)/3),
            r * cos((φ + 4π)/3)
        ]
    end
end

function image_positions(lens, β::SVector{2,Float64})
    d, e = lens.d, lens.e
    βx, βy = β[1], β[2]

    a = e - 0.5*d^2
    b = d * βx
    c = -βy

    p = b / a
    q = c / a

    ys_roots = depressed_cubic_real_roots(p, q)

    # recover x from βx = x + (d/2) y^2
    imgs = SVector{2,Float64}[]
    for y in ys_roots
        x = βx - 0.5*d*y^2
        push!(imgs, @SVector [x, y])
    end
    return imgs
end

# ---------------------------------------------
# Ray-shooting approach for extended sources
# ---------------------------------------------

# lens equation mapping θ -> β
β_from_θ(lens, θ::SVector{2,Float64}) = θ - deflection_at(lens, θ)

"""
Return image-plane intensity map I(θ) on a grid (xcent × ycent).
I_img(θ) = I_src(β(θ)).
"""
function ray_shoot_intensity_map(lens, src, xs, ys)
    Nx = length(xs)
    Ny = length(ys)
    I = Matrix{Float64}(undef, Ny, Nx)   # (y,x) for heatmap

    @inbounds for j in 1:Ny
        y = ys[j]
        for i in 1:Nx
            x = xs[i]
            θ = SVector{2,Float64}(x, y)
            β = β_from_θ(lens, θ)
            I[j, i] = float(intensity(src, β))
        end
    end
    return I
end

# --- draw half-plane in the source plane: boundary + shaded bright side ---
function add_halfplane_to_sourceplot!(p, βstar::SVector{2,Float64}, n::SVector{2,Float64};
        L::Float64=5.0, α::Float64=0.25)

    n_hat = normalize2(n)
    tdir = tangent_from_normal(n_hat)

    # boundary line (for display)
    tt = range(-L, L; length=200)
    xs = [ (βstar + τ*tdir)[1] for τ in tt ]
    ys = [ (βstar + τ*tdir)[2] for τ in tt ]
    plot!(p, xs, ys; lw=2, linestyle=:dash)

    # shaded bright side: big rectangle extending along +n_hat
    a  = βstar - L*tdir
    b  = βstar + L*tdir
    a2 = a + L*n_hat
    b2 = b + L*n_hat

    polyx = [a[1], b[1], b2[1], a2[1]]
    polyy = [a[2], b[2], b2[2], a2[2]]
    plot!(p, polyx, polyy; seriestype=:shape, opacity=α, linealpha=0.0)

    return p
end

# --- helper: block-mean downsampling for display purposes ---
function block_mean(A::AbstractMatrix{<:Real}, os::Int)
    Ny_hi, Nx_hi = size(A)
    @assert Nx_hi % os == 0 "Nx_hi=$(Nx_hi) is not divisible by os=$(os)"
    @assert Ny_hi % os == 0 "Ny_hi=$(Ny_hi) is not divisible by os=$(os)"
    Ny = Ny_hi ÷ os
    Nx = Nx_hi ÷ os
    B = Matrix{Float64}(undef, Ny, Nx)
    inv_os2 = 1.0 / (os * os)

    @inbounds for j in 1:Ny
        j0 = (j-1)*os + 1
        for i in 1:Nx
            i0 = (i-1)*os + 1
            s = 0.0
            for jj in j0:j0+os-1, ii in i0:i0+os-1
                s += A[jj, ii]
            end
            B[j,i] = s * inv_os2
        end
    end
    return B
end


#---generate ray-shooting intensity map and plot with caustics/critical curves ---

I_hi  = ray_shoot_intensity_map(lens, src, xs_hi, ys_hi)
I_pix = block_mean(I_hi, os)
Iplot = log10.(I_pix .+ 1e-12)

p_lens_hi = heatmap(xs_hi, ys_hi, log10.(I_hi .+ 1e-12);  # log for display
    aspect_ratio=:equal,
    xlabel="θx", ylabel="θy",
    title="Lens plane: high-res",
    colorbar=false, legend=false
)

p_lens = heatmap(xs_pix, ys_pix, Iplot;
    aspect_ratio=:equal,
    xlabel="θx", ylabel="θy",
    title="Lens plane: pixelated",
    colorbar=false, legend=false
)

for poly in critical_polylines
    plot!(p_lens, first.(poly), last.(poly); lw=2, linecolor=:white)
end
println("Plotted lens plane intensity map.")

p_src = plot(; aspect_ratio=:equal,
    xlabel="βx", ylabel="βy",
    title="Source plane",
    legend=false
)

for poly in caustic_polylines
    plot!(p_src, first.(poly), last.(poly); lw=2)
end
println("Plotted caustic curves in source plane.")

add_halfplane_to_sourceplot!(p_src, βstar, n; L=5.0, α=0.25)
scatter!(p_src, [βstar[1]], [βstar[2]]; markersize=5)

p_overlay = plot(p_lens_hi, p_lens, p_src; layout=(1,3), size=(1200, 600))

savefig(p_overlay, "halfplane_rayshooting_truth.png")
println("Saved: halfplane_rayshooting_truth.png")


