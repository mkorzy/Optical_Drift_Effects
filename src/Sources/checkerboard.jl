#  src/Sources/Checkerboard.jl
# using StaticArrays

# """
#     CheckerboardSource <: AbstractSourceModel

# Axis-aligned checkerboard (grid) surface-brightness pattern on the source plane.
# Useful for visualising distortion, shear, and magnification near caustics.

# Parameters
# - cell_size:   side length of each square cell (same units as β)
# - I_hi:        intensity of the "white" squares  (default 1.0)
# - I_lo:        intensity of the "black" squares  (default 0.0)
# - β0:          origin of the grid in the source plane (βx0, βy0); a grid vertex
#                lands exactly on this point
# - ϕ:           rotation angle of the grid (radians, CCW).  ϕ=0 gives an axis-
#                aligned board.
# - window_size: if > 0, the pattern is multiplied by a smooth super-Gaussian
#                envelope exp(-(r/window_size)^window_power) so the board fades
#                to zero far from β0, which avoids sharp artefacts at the edges of
#                your ray-tracing field.  Set to 0.0 to disable.
# - window_power: exponent of the envelope (default 6, giving a very flat top).
# """
# struct CheckerboardSource <: AbstractSourceModel
#     cell_size   ::Float64
#     I_hi        ::Float64
#     I_lo        ::Float64
#     β0          ::SVector{2,Float64}
#     ϕ           ::Float64   # grid rotation
#     window_size ::Float64   # 0 → no envelope
#     window_power::Float64
# end

# function CheckerboardSource(;
#         cell_size   ::Real   = 0.1,
#         I_hi        ::Real   = 1.0,
#         I_lo        ::Real   = 0.0,
#         β0                   = (0.0, 0.0),
#         ϕ           ::Real   = 0.0,
#         window_size ::Real   = 0.0,
#         window_power::Real   = 6.0)

#     return CheckerboardSource(
#         float(cell_size),
#         float(I_hi), float(I_lo),
#         SVector{2,Float64}(β0[1], β0[2]),
#         float(ϕ),
#         float(window_size),
#         float(window_power),
#     )
# end

#  ---------------------------------------------------------------------------
#  Internal helpers
#  ---------------------------------------------------------------------------

#  Rotate a 2-vector by angle ϕ (same convention as SersicSource)
# @inline function _cb_rotate(ϕ::T, v::SVector{2,T}) where {T<:Number}
#     c = cos(ϕ); s = sin(ϕ)
#     return @SVector [ c*v[1] + s*v[2],
#                      -s*v[1] + c*v[2] ]
# end

#  ---------------------------------------------------------------------------
#  Public API  –  same signature as every other source model
#  ---------------------------------------------------------------------------

# """
#     intensity(src::CheckerboardSource, β::SVector{2,T}) where T<:Number

# Returns I_hi or I_lo depending on which checkerboard cell β falls in, optionally
# multiplied by a smooth radial envelope.

# The cell index (ix, iy) is computed after rotating β into the grid frame.
# A cell is "white"  when  (ix + iy) is even, and "black" otherwise.
# """
# function intensity(src::CheckerboardSource, β::SVector{2,T}) where {T<:Number}

#     # --- 1. shift to grid origin, then rotate into grid frame ---------------
#     d  = β - SVector{2,T}(T(src.β0[1]), T(src.β0[2]))
#     xy = _cb_rotate(T(src.ϕ), d)       # grid-aligned coordinates

#     # --- 2. cell indices (floor division → integer parity) ------------------
#     cs  = T(src.cell_size)
#     ix  = floor(Int, xy[1] / cs)
#     iy  = floor(Int, xy[2] / cs)
#     parity = (ix + iy) & 1             # 0 for white, 1 for black

#     base = parity == 0 ? T(src.I_hi) : T(src.I_lo)

#     # --- 3. optional smooth envelope ----------------------------------------
#     ws = T(src.window_size)
#     if ws > zero(T)
#         r   = sqrt(d[1]*d[1] + d[2]*d[2])
#         env = exp(-(r / ws)^T(src.window_power))
#         return base * env
#     else
#         return base
#     end
# end

# src/Sources/Checkerboard.jl
using StaticArrays

"""
    CheckerboardSource <: AbstractSourceModel

Axis-aligned checkerboard (grid) surface-brightness pattern on the source plane.
Useful for visualising distortion, shear, and magnification near caustics.

Parameters
- cell_size:    side length of each square cell (same units as β)
- I_hi:         intensity of the "white" squares  (default 1.0)
- I_lo:         intensity of the "black" squares  (default 0.0)
- β0:           origin of the grid in the source plane (βx0, βy0); a grid vertex
                lands exactly on this point
- ϕ:            rotation angle of the grid (radians, CCW).  ϕ=0 gives an axis-
                aligned board.
- window_size:  if > 0, pixels outside this radius from β0 return NaN so they
                render as background in the heatmap.  In standard mode the
                envelope fades smoothly to zero; in hue_gradient mode a hard
                mask is applied so the colour encoding is not distorted by a
                brightness ramp.  Set to 0.0 to disable.
- window_power: exponent of the smooth envelope in standard mode (default 6).
- hue_gradient: if true, intensity() returns a value in [0, 1] that encodes
                x-position as a pure hue ramp, with checker parity applied as
                a discrete brightness step (dark_fraction).  No smooth envelope
                is applied so the hue is not corrupted by intensity variation.
                Use with a cyclic colormap such as :hsv and clims=(0, 1).
                When false (default), the original scalar I_hi/I_lo behaviour
                is preserved.
- x_range:      (xmin, xmax) over which the hue ramps 0 → 1 (source-plane
                units, applied to the unrotated x-offset from β0).  Points
                outside are clamped to [0, 1].  Defaults to
                (-window_size, +window_size) when window_size > 0, else (-1, 1).
- dark_fraction: brightness of "black" cells in hue_gradient mode as a fraction
                of the "white" value (default 0.5).
"""
struct CheckerboardSource <: AbstractSourceModel
    cell_size    ::Float64
    I_hi         ::Float64
    I_lo         ::Float64
    β0           ::SVector{2,Float64}
    ϕ            ::Float64
    window_size  ::Float64
    window_power ::Float64
    hue_gradient ::Bool
    x_range      ::SVector{2,Float64}
    dark_fraction::Float64
end

function CheckerboardSource(;
        cell_size    ::Real   = 0.1,
        I_hi         ::Real   = 1.0,
        I_lo         ::Real   = 0.0,
        β0                    = (0.0, 0.0),
        ϕ            ::Real   = 0.0,
        window_size  ::Real   = 0.0,
        window_power ::Real   = 6.0,
        hue_gradient ::Bool   = false,
        x_range               = nothing,
        dark_fraction::Real   = 0.5)

    if x_range === nothing
        half = window_size > 0.0 ? float(window_size) : 1.0
        xr = SVector{2,Float64}(-half, half)
    else
        xr = SVector{2,Float64}(float(x_range[1]), float(x_range[2]))
    end

    return CheckerboardSource(
        float(cell_size),
        float(I_hi), float(I_lo),
        SVector{2,Float64}(β0[1], β0[2]),
        float(ϕ),
        float(window_size),
        float(window_power),
        hue_gradient,
        xr,
        float(dark_fraction),
    )
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@inline function _cb_rotate(ϕ::T, v::SVector{2,T}) where {T<:Number}
    c = cos(ϕ); s = sin(ϕ)
    return @SVector [ c*v[1] + s*v[2],
                     -s*v[1] + c*v[2] ]
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    intensity(src::CheckerboardSource, β::SVector{2,T}) where T<:Number

Default (hue_gradient=false):
  Returns I_hi or I_lo depending on checker parity, optionally modulated by
  a smooth radial envelope. Returns NaN outside window_size (after the envelope
  reaches zero), which renders as background in Plots.heatmap.

Hue-gradient mode (hue_gradient=true):
  Returns a value in [0, 1] encoding x-position as a pure hue ramp.
  "White" cells return h(x); "black" cells return h(x) * dark_fraction.
  Points outside window_size return NaN (hard mask, no smooth fade).
  No intensity envelope is applied — the scalar encodes position only.
  Recommended plot call:
      heatmap(xs, ys, I; color=:hsv, clims=(0,1), background_color=:black)
"""
function intensity(src::CheckerboardSource, β::SVector{2,T}) where {T<:Number}

    # --- 1. offset from grid origin (unrotated — used for window and hue) ---
    d = β - SVector{2,T}(T(src.β0[1]), T(src.β0[2]))

    ws = T(src.window_size)

    # --- 2. window handling -------------------------------------------------
    if ws > zero(T)
        r = sqrt(d[1]*d[1] + d[2]*d[2])
        if src.hue_gradient
            # hard mask: outside window → NaN (background), no brightness ramp
            r > ws && return T(NaN)
        else
            env = exp(-(r / ws)^T(src.window_power))
        end
    end

    # --- 3. checker parity (rotated grid frame) -----------------------------
    xy     = _cb_rotate(T(src.ϕ), d)
    cs     = T(src.cell_size)
    ix     = floor(Int, xy[1] / cs)
    iy     = floor(Int, xy[2] / cs)
    parity = (ix + iy) & 1            # 0 = "white", 1 = "black"

    # --- 4. hue-gradient mode: pure position encoding, no envelope ----------
    if src.hue_gradient
        xmin = T(src.x_range[1])
        xmax = T(src.x_range[2])
        h = clamp((xy[1] - xmin) / (xmax - xmin), zero(T), one(T))
        parity_offset = T(src.dark_fraction)   # reused field: now means hue offset
        if parity == 0
            return h * (one(T) - parity_offset)
        else
            return h * (one(T) - parity_offset) + parity_offset
        end
    end

    # --- 5. standard scalar mode --------------------------------------------
    base = parity == 0 ? T(src.I_hi) : T(src.I_lo)
    return ws > zero(T) ? base * env : base
end