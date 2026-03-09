# src/Sources/halfPlane.jl
using StaticArrays


""" 
    HalfPlaneSource <: AbstractSourceModel

A half-plane source with a steep gradient across a line.
Bright side is where dot(n, β-βstar) > 0.

Parameters:
 - Ibright: intensity on bright side
 - Idim: intensity on dim side
 - βstar: point on the boundary line
 - n: normal direction (need not be unit, but good if it is)
 - w: smoothing length in β units (smaller w = sharper boundary)
"""
struct HalfPlaneSource <: AbstractSourceModel
    Ibright::Float64
    Idim::Float64
    βstar::SVector{2,Float64}     
    n::SVector{2,Float64}      
    w::Float64                 
end

function HalfPlaneSource(; Ibright::Real=1.0, Idim::Real=0.0,
                          βstar=(0.0, 0.0),
                          n=(0.3, 1.0),
                          w::Real=0.02)
    return HalfPlaneSource(float(Ibright), float(Idim),
                           SVector{2,Float64}(βstar[1], βstar[2]),
                           SVector{2,Float64}(n[1], n[2]),
                           float(w))
end

# ----- helpers -----------------------------------------------------------------
# normalize2(v::SVector{2,Float64}) = v / hypot(v[1], v[2])

"""
    smoothstep_tanh(s, w)

Smoothed Heaviside step function: 0..1, with transition around s and width ~w.

"""
@inline smoothstep_tanh(s, w) = 0.5 * (1 + tanh(s / w))

# ------ Public API -----------------------------------------------------------------
"""
    intensity(src::HalfPlaneSource, β)

Evaluate the source intensity at position β on the source plane.
"""
function intensity(src::HalfPlaneSource, β::SVector{2,Float64})
    s = dot(src.n, β - src.βstar)
    H = smoothstep_tanh(s, src.w)
    return src.Idim + (src.Ibright - src.Idim) * H
end



