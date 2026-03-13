# scripts/checkerboard_lensing.jl
using Optical_Drift_Effects
using StaticArrays
using LinearAlgebra
using Plots

# -----------------------------
# Lens set-up
# -----------------------------
# d=1, e=1 
lens = generic_cusp(1, 1)   # <-- change this line to your lens object / parameters 

# ------------------------------
# Source set-up
# ------------------------------
cell_size = 0.2
src = CheckerboardSource(cell_size=cell_size, ϕ = 0.0 , hue_gradient=true, x_range=(-2.0, 2.0))

# -----------------------------
# Grid settings
# -----------------------------

os = 5           # oversampling factor (pixels per output pixel, per axis)

Nx_pix, Ny_pix = 200, 200                 # desired output (pixelated) resolution
Nx_hi,  Ny_hi  = os * Nx_pix, os * Ny_pix  # high-res ray grid

xmin, xmax = -2.0, 2.0
ymin, ymax = -2.0, 2.0

# High-res ray grid 
xs_hi = range(xmin, xmax; length=Nx_hi)
ys_hi = range(ymin, ymax; length=Ny_hi)

# Output pixel centres (for plot axes)
xs_pix = range(xmin, xmax; length=Nx_pix)
ys_pix = range(ymin, ymax; length=Ny_pix)


# ---------------------------------------------
# lensing functions at a point (wrapper for your lens functions)
# ---------------------------------------------             
deflection_at(lens, θ) = deflection(lens, θ)               # returns SVector(αx, αy)
jacobian_at(lens, θ) = deflection_jacobian(lens, θ)        # returns 2x2 (SMatrix ok)


# ---------------------------------------------
#  Critical curves: find det(J)=0 contours in image plane
# ---------------------------------------------
critical_polylines = critical_curves(lens, xs_hi, ys_hi)
caustic_polylines = caustic_curves(lens, critical_polylines)

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


#---generate ray-shooting intensity map and plot with caustics/critical curves ---

I_hi  = ray_shoot_intensity_map(lens, src, xs_hi, ys_hi)
p_lens = heatmap(xs_hi, ys_hi, I_hi;  
    aspect_ratio=:equal,
    xlabel="θx", ylabel="θy",
    title="Lens plane",
    colorbar=false, legend=false,
    color=:RdBu, clims=(0, 1),
    background_color_inside=:black
)

for poly in critical_polylines
    plot!(p_lens, first.(poly), last.(poly); lw=2, linecolor=:cyan)
end

xs_src = range(xmin, xmax; length=Nx_pix)
ys_src = range(ymin, ymax; length=Ny_pix)

I_src_map = [intensity(src, SVector{2,Float64}(x, y))
             for y in ys_src, x in xs_src]   # (Ny × Nx), matching heatmap layout

p_src = heatmap(xs_src, ys_src, I_src_map;
    aspect_ratio=:equal,
    xlabel="βx", ylabel="βy",
    title="Source plane",
    colorbar=false, legend=false,
    color=:RdBu, clims=(0, 1),
    background_color_inside=:black
)

for poly in caustic_polylines
    plot!(p_src, first.(poly), last.(poly); lw=2, linecolor=:cyan)
end

p_overlay = plot(p_lens, p_src; layout=(1,2), size=(1200, 600), left_margin=12Plots.mm, right_margin=6Plots.mm,
    top_margin=6Plots.mm,   bottom_margin=12Plots.mm)

savefig(p_overlay, "checkerboard_horizontalHue.png")
println("Saved: checkerboard.png")