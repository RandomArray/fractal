#!/usr/bin/env python3
"""
fractal.py - random deep-zoom fractal video generator.

Every run picks a random fractal family, hunts for a visually rich zoom target,
picks a palette + lighting style, and renders a smooth, eased zoom straight
into an MP4 (via ffmpeg).

Highlights
  * Fractals: Mandelbrot (dives into real minibrots), Multibrot (z^3..z^5), Julia sets.
  * Smart target search: descends level-by-level, scoring candidate views with
    look-ahead renders so the camera lands on spirals / minibrots, never on
    blank space or a smooth cardioid edge.
  * Analytic derivatives -> 3D "normal map" lighting with specular highlights,
    plus distance-estimate ink/glow on filaments.
  * Keyframe renderer: each 2x zoom band is rendered once at high resolution
    and resampled (trilinear, linear-light) for every video frame. That's
    supersampled quality at a fraction of the cost and rock-steady motion.
  * Adaptive iteration limits and per-depth color normalisation so every depth
    uses the whole palette.
  * Periodicity checking + cardioid/bulb tests make interiors cheap.

Usage
  python fractal.py                     # one random 1080p zoom
  python fractal.py --preview           # fast 640x360 test
  python fractal.py --count 5           # five different videos
  python fractal.py --seed 1234         # reproduce a previous video
  python fractal.py --contact           # just a contact sheet of the path
  python fractal.py --fractal julia --palette lava --duration 40
  python fractal.py --help

Requirements: numpy, numba, pillow, (tqdm optional), ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import numba
from numba import njit, prange
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:
    numba.set_parallel_chunksize(2)  # dynamic scheduling: rows vary wildly in cost
except Exception:  # pragma: no cover
    pass

KIND_MANDEL, KIND_JULIA = 0, 1
BAILOUT = 1.0e3
BAIL2 = BAILOUT * BAILOUT
# Periodicity tolerance lives in z-space. Near a cycle |dz/dc| is huge, so even
# 1e-9 in z corresponds to far less than a pixel in c, even at 1e-12 zooms.
PERIOD_EPS2 = 1e-9 ** 2
EPS = 2.220446049250313e-16

# ---------------------------------------------------------------------------
# Core iteration (escape time + derivative)
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def escape(kind, power, px, py, jx, jy, max_iter, eps2):
    """Iterate one point.

    Returns (nu, nx, ny, de, iterations_used):
      nu  smooth iteration count (< 0 means inside the set)
      nx, ny  unit gradient of the potential (surface normal direction)
      de  distance estimate to the set boundary, in plane units
    """
    if kind == KIND_JULIA:
        x = px
        y = py
        cx = jx
        cy = jy
        dx = 1.0
        dy = 0.0
    else:
        x = 0.0
        y = 0.0
        cx = px
        cy = py
        dx = 0.0
        dy = 0.0
        if kind == KIND_MANDEL and power == 2:
            xq = px - 0.25
            q = xq * xq + py * py
            if q * (q + xq) <= 0.25 * py * py:
                return -1.0, 0.0, 0.0, 0.0, 0
            xb = px + 1.0
            if xb * xb + py * py <= 0.0625:
                return -1.0, 0.0, 0.0, 0.0, 0
    sx = x
    sy = y
    period = 0
    check = 3
    logd = math.log(power)
    logbail = math.log(BAILOUT)
    for n in range(max_iter):
        r2 = x * x + y * y
        if r2 > BAIL2:
            logz = 0.5 * math.log(r2)
            nu = n + 1.0 - math.log(logz / logbail) / logd
            gx = (dx * x + dy * y) / r2
            gy = -(dy * x - dx * y) / r2
            gl = math.sqrt(gx * gx + gy * gy)
            if gl > 0.0 and gl < 1e300:
                return nu, gx / gl, gy / gl, logz / gl, n
            return nu, 0.0, 0.0, 1e30, n
        if power == 2:
            ndx = 2.0 * (x * dx - y * dy)
            ndy = 2.0 * (x * dy + y * dx)
            if kind != KIND_JULIA:
                ndx += 1.0
            dx = ndx
            dy = ndy
            ny_ = 2.0 * x * y + cy
            x = x * x - y * y + cx
            y = ny_
        else:
            qx = x
            qy = y
            for _ in range(power - 2):
                t = qx * x - qy * y
                qy = qx * y + qy * x
                qx = t
            ndx = power * (qx * dx - qy * dy)
            ndy = power * (qx * dy + qy * dx)
            if kind != KIND_JULIA:
                ndx += 1.0
            dx = ndx
            dy = ndy
            nx_ = qx * x - qy * y + cx
            y = qx * y + qy * x + cy
            x = nx_
        # Brent-style periodicity check: attracting cycles => interior
        ex = x - sx
        ey = y - sy
        if ex * ex + ey * ey < eps2:
            return -1.0, 0.0, 0.0, 0.0, n
        period += 1
        if period >= check:
            period = 0
            check = check * 2
            sx = x
            sy = y
    return -1.0, 0.0, 0.0, 0.0, max_iter


@njit(parallel=True, cache=True, fastmath=True)
def probe_grid(kind, power, jx, jy, cx, cy, s, w, h, cosr, sinr, max_iter, eps2):
    """Low-res smooth-iteration grid of a (rotated) view of width s."""
    out = np.empty((h, w), np.float64)
    cost = np.empty((h, w), np.float64)
    px = s / w
    for j in prange(h):
        for i in range(w):
            ox = (i + 0.5 - 0.5 * w) * px
            oy = (0.5 * h - j - 0.5) * px
            x = cx + ox * cosr - oy * sinr
            y = cy + ox * sinr + oy * cosr
            r = escape(kind, power, x, y, jx, jy, max_iter, eps2)
            out[j, i] = r[0]
            cost[j, i] = r[4] + 1.0
    return out, cost


# ---------------------------------------------------------------------------
# Coloring
# ---------------------------------------------------------------------------

# indices into the color-parameter vector `cp`
CP_LO, CP_INV, CP_CYC, CP_PHASE = 0, 1, 2, 3
CP_LIGHT, CP_LX, CP_LY, CP_H2, CP_AMB, CP_DIF, CP_SPEC, CP_SHINE = 4, 5, 6, 7, 8, 9, 10, 11
CP_EDGE, CP_EDGE_W, CP_EDGE_MIN, CP_GLOW, CP_GR, CP_GG, CP_GB = 12, 13, 14, 15, 16, 17, 18
CP_IR, CP_IG, CP_IB = 19, 20, 21
CP_SIZE = 22


@njit(cache=True, fastmath=True)
def colorize(nu, nx, ny, de_px, cp, lut):
    if nu < 0.0:
        return cp[CP_IR], cp[CP_IG], cp[CP_IB]
    t = (math.log(nu + 1.0) - cp[CP_LO]) * cp[CP_INV]
    p = t * cp[CP_CYC] + cp[CP_PHASE]
    p -= math.floor(p)
    n = lut.shape[0]
    f = p * n
    i0 = int(f)
    fr = f - i0
    i0 = i0 % n
    i1 = (i0 + 1) % n
    r = lut[i0, 0] * (1.0 - fr) + lut[i1, 0] * fr
    g = lut[i0, 1] * (1.0 - fr) + lut[i1, 1] * fr
    b = lut[i0, 2] * (1.0 - fr) + lut[i1, 2] * fr
    if cp[CP_LIGHT] > 0.0:
        h2 = cp[CP_H2]
        lam = (nx * cp[CP_LX] + ny * cp[CP_LY] + h2) / (1.0 + h2)
        if lam < 0.0:
            lam = 0.0
        k = cp[CP_AMB] + cp[CP_DIF] * lam
        sp = cp[CP_SPEC] * lam ** cp[CP_SHINE]
        r = r * k + sp
        g = g * k + sp
        b = b * k + sp
    style = cp[CP_EDGE]
    if style == 1.0:  # ink: darken thin filaments
        m = de_px * cp[CP_EDGE_W]
        if m > 1.0:
            m = 1.0
        m = m * m * (3.0 - 2.0 * m)
        fdim = cp[CP_EDGE_MIN] + (1.0 - cp[CP_EDGE_MIN]) * m
        r *= fdim
        g *= fdim
        b *= fdim
    elif style == 2.0:  # glow: filaments light up
        gl = cp[CP_GLOW] * math.exp(-de_px * cp[CP_EDGE_W])
        r += gl * cp[CP_GR]
        g += gl * cp[CP_GG]
        b += gl * cp[CP_GB]
    return r, g, b


@njit(parallel=True, cache=True, fastmath=True)
def render_key(kind, power, jx, jy, x0, y0, kpx, nxp, nyp, ss, max_iter, eps2, cp, lut):
    """Render an axis-aligned key image (linear RGB float32).

    Pixel (i, j) covers plane [x0 + i*kpx, x0 + (i+1)*kpx] x [y0 - (j+1)*kpx, y0 - j*kpx].
    """
    img = np.empty((nyp, nxp, 3), np.float32)
    inv = 1.0 / (ss * ss)
    for j in prange(nyp):
        for i in range(nxp):
            r = 0.0
            g = 0.0
            b = 0.0
            for sy in range(ss):
                for sx in range(ss):
                    x = x0 + (i + (sx + 0.5) / ss) * kpx
                    y = y0 - (j + (sy + 0.5) / ss) * kpx
                    nu, nx, ny, de, _ = escape(kind, power, x, y, jx, jy, max_iter, eps2)
                    cr, cg, cb = colorize(nu, nx, ny, de / kpx, cp, lut)
                    r += cr
                    g += cg
                    b += cb
            img[j, i, 0] = r * inv
            img[j, i, 1] = g * inv
            img[j, i, 2] = b * inv
    return img


# ---------------------------------------------------------------------------
# Compositing (key images -> video frame)
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def bilerp(img, fx, fy):
    h = img.shape[0]
    w = img.shape[1]
    fx -= 0.5
    fy -= 0.5
    xf = math.floor(fx)
    yf = math.floor(fy)
    tx = fx - xf
    ty = fy - yf
    x0 = min(max(int(xf), 0), w - 1)
    x1 = min(max(int(xf) + 1, 0), w - 1)
    y0 = min(max(int(yf), 0), h - 1)
    y1 = min(max(int(yf) + 1, 0), h - 1)
    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty
    r = img[y0, x0, 0] * w00 + img[y0, x1, 0] * w10 + img[y1, x0, 0] * w01 + img[y1, x1, 0] * w11
    g = img[y0, x0, 1] * w00 + img[y0, x1, 1] * w10 + img[y1, x0, 1] * w01 + img[y1, x1, 1] * w11
    b = img[y0, x0, 2] * w00 + img[y0, x1, 2] * w10 + img[y1, x0, 2] * w01 + img[y1, x1, 2] * w11
    return r, g, b


@njit(cache=True, fastmath=True)
def sample_key(m0, m1, m2, kx0, ky0, kpx, x, y, lod):
    fx = (x - kx0) / kpx
    fy = (ky0 - y) / kpx
    if lod <= 0.0:
        return bilerp(m0, fx, fy)
    if lod < 1.0:
        r0, g0, b0 = bilerp(m0, fx, fy)
        r1, g1, b1 = bilerp(m1, fx * 0.5, fy * 0.5)
        t = lod
    else:
        r0, g0, b0 = bilerp(m1, fx * 0.5, fy * 0.5)
        r1, g1, b1 = bilerp(m2, fx * 0.25, fy * 0.25)
        t = min(lod - 1.0, 1.0)
    return r0 + (r1 - r0) * t, g0 + (g1 - g0) * t, b0 + (b1 - b0) * t


@njit(cache=True, fastmath=True)
def srgb_encode(c):
    if c <= 0.0:
        return 0
    if c >= 1.0:
        return 255
    if c <= 0.0031308:
        v = 12.92 * c
    else:
        v = 1.055 * c ** (1.0 / 2.4) - 0.055
    return int(v * 255.0 + 0.5)


@njit(parallel=True, cache=True, fastmath=True)
def composite(W, H, cx, cy, s, cosr, sinr,
              a0, a1, a2, ax0, ay0, apx,
              b0, b1, b2, bx0, by0, bpx, wb,
              hue, vignette, exposure, out):
    opx = s / W
    lod_a = min(max(math.log2(opx / apx), 0.0), 2.0)
    lod_b = min(max(math.log2(opx / bpx), 0.0), 2.0)
    norm = 1.0 / (1.0 - math.exp(-exposure))
    for j in prange(H):
        for i in range(W):
            ox = (i + 0.5 - 0.5 * W) * opx
            oy = (0.5 * H - j - 0.5) * opx
            x = cx + ox * cosr - oy * sinr
            y = cy + ox * sinr + oy * cosr
            r, g, b = sample_key(a0, a1, a2, ax0, ay0, apx, x, y, lod_a)
            if wb > 0.0:
                rB, gB, bB = sample_key(b0, b1, b2, bx0, by0, bpx, x, y, lod_b)
                r += (rB - r) * wb
                g += (gB - g) * wb
                b += (bB - b) * wb
            rr = hue[0] * r + hue[1] * g + hue[2] * b
            gg = hue[3] * r + hue[4] * g + hue[5] * b
            bb = hue[6] * r + hue[7] * g + hue[8] * b
            u = (i + 0.5) / W - 0.5
            v = ((j + 0.5) / H - 0.5) * H / W
            vf = 1.0 - vignette * (u * u + v * v) * 2.2
            if vf < 0.0:
                vf = 0.0
            rr = (1.0 - math.exp(-max(rr, 0.0) * vf * exposure)) * norm
            gg = (1.0 - math.exp(-max(gg, 0.0) * vf * exposure)) * norm
            bb = (1.0 - math.exp(-max(bb, 0.0) * vf * exposure)) * norm
            out[j, i, 0] = srgb_encode(rr)
            out[j, i, 1] = srgb_encode(gg)
            out[j, i, 2] = srgb_encode(bb)


# ---------------------------------------------------------------------------
# Fractal families, palettes, styles
# ---------------------------------------------------------------------------

JULIA_SEEDS = [
    (-0.8, 0.156), (-0.7269, 0.1889), (0.285, 0.01), (-0.4, 0.6),
    (-0.835, -0.2321), (-0.70176, -0.3842), (0.355, 0.355), (-0.54, 0.54),
    (-0.75, 0.11), (-0.1, 0.651), (0.355534, -0.337292), (-0.123, 0.745),
    (-0.391, -0.587), (0.3, -0.5), (-1.476, 0.0), (-0.162, 1.04),
]


@dataclass
class FractalSpec:
    name: str
    kind: int
    power: int = 2
    jx: float = 0.0
    jy: float = 0.0
    home_x: float = -0.6
    home_y: float = 0.0
    home_s: float = 3.4  # view width
    iter_cap: int = 0  # 0 = use the global cap

    @property
    def cap(self):
        return min(ITER_CAP, self.iter_cap) if self.iter_cap else ITER_CAP

    def probe(self, cx, cy, s, w, h, max_iter, angle=0.0):
        return self.probe_cost(cx, cy, s, w, h, max_iter, angle)[0]

    def probe_cost(self, cx, cy, s, w, h, max_iter, angle=0.0):
        return probe_grid(self.kind, self.power, self.jx, self.jy, cx, cy, s, w, h,
                          math.cos(angle), math.sin(angle), int(max_iter), PERIOD_EPS2)


def make_fractal(name, rng):
    if name == "mandelbrot":
        return FractalSpec("mandelbrot", KIND_MANDEL, 2, home_x=-0.65, home_y=0.0, home_s=3.6)
    if name.startswith("multibrot"):
        p = int(name[-1]) if name[-1].isdigit() else int(rng.choice([3, 4, 5]))
        return FractalSpec(f"multibrot{p}", KIND_MANDEL, p, home_x=-0.1 if p == 3 else 0.0,
                           home_y=0.0, home_s=3.2)
    if name == "julia":
        jx, jy = JULIA_SEEDS[rng.integers(len(JULIA_SEEDS))]
        jx += rng.normal(0, 0.002)
        jy += rng.normal(0, 0.002)
        return FractalSpec("julia", KIND_JULIA, 2, jx=jx, jy=jy, home_x=0.0, home_y=0.0, home_s=3.6)
    raise ValueError(name)


PALETTES = {
    "classic": ["000764", "206bcb", "edffff", "ffaa00", "000200"],
    "ice": ["02010a", "0b2d5b", "2e86c1", "aee8ff", "ffffff", "5dade2", "1b2631"],
    "sunset": ["1a0533", "5b1a72", "c2185b", "ff6f3c", "ffd166", "fff3c4", "3d0a4f"],
    "neon": ["090018", "3a0ca3", "7209b7", "f72585", "ffbe0b", "4cc9f0", "0a0a2a"],
    "emerald": ["01110a", "064e3b", "10b981", "d1fae5", "fde68a", "92400e", "1c1917"],
    "lava": ["0a0000", "4a0000", "b91c1c", "f97316", "fde047", "fff7d6", "7c2d12"],
    "ocean": ["001219", "005f73", "0a9396", "94d2bd", "e9d8a6", "ee9b00", "ca6702", "9b2226"],
    "aurora": ["03071e", "14213d", "2ec4b6", "cbf3f0", "a2d2ff", "b5179e", "3c096c"],
    "gold": ["0b0b0b", "3d2b1f", "8c6239", "d4af37", "fff1c1", "c08a2e", "2b1d0e"],
    "candy": ["1b0a2a", "ff499e", "ffd6e8", "7bdff2", "b2f7ef", "4d2c91"],
    "silver": ["000000", "3a3a3a", "a0a0a0", "ffffff", "707070", "101010"],
    "electric": ["000814", "001d3d", "003566", "ffc300", "ffd60a", "ffffff", "0077b6"],
    "vapor": ["120458", "7a04eb", "ff00a0", "fe75fe", "ffd8ff", "00d4ff", "0b0033"],
    "forest": ["0b1d13", "1e4d2b", "6a994e", "f2e8cf", "bc4749", "386641"],
}


def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def build_lut(name, rng, n=2048):
    """Cyclic palette LUT in linear RGB. Returns (lut, darkest_linear_color, label)."""
    t = np.arange(n) / n
    if name == "cosine":
        d = rng.uniform(0, 1, 3)
        b = rng.uniform(0.35, 0.5)
        srgb = 0.5 + b * np.cos(2 * np.pi * (t[:, None] + d[None, :]))
        label = "cosine(" + ",".join(f"{v:.2f}" for v in d) + ")"
    else:
        stops = np.array([[int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] for h in PALETTES[name]])
        if rng.random() < 0.5:
            stops = stops[::-1]
        stops = np.roll(stops, int(rng.integers(len(stops))), axis=0)
        k = len(stops)
        pos = t * k
        i0 = np.floor(pos).astype(int) % k
        i1 = (i0 + 1) % k
        fr = pos - np.floor(pos)
        fr = fr * fr * (3 - 2 * fr)  # smooth blend between stops
        srgb = stops[i0] * (1 - fr[:, None]) + stops[i1] * fr[:, None]
        label = name
    lut = srgb_to_linear(np.clip(srgb, 0, 1)).astype(np.float64)
    lum = lut @ np.array([0.2126, 0.7152, 0.0722])
    return lut, lut[int(np.argmin(lum))], label


def hue_matrix(deg):
    """Rotate colors around the gray axis (linear RGB)."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    k = 1.0 / 3.0
    sq = math.sqrt(k)
    return np.array([
        c + (1 - c) * k, k * (1 - c) - sq * s, k * (1 - c) + sq * s,
        k * (1 - c) + sq * s, c + k * (1 - c), k * (1 - c) - sq * s,
        k * (1 - c) - sq * s, k * (1 - c) + sq * s, c + k * (1 - c),
    ], np.float64)


@dataclass
class Style:
    palette: str
    cycles: float
    phase: float
    lighting: bool
    light_angle: float
    h2: float
    ambient: float
    diffuse: float
    specular: float
    shine: float
    edge: int  # 0 none, 1 ink, 2 glow
    edge_w: float
    edge_min: float
    glow: float
    interior: str
    hue_start: float
    hue_total: float
    vignette: float
    exposure: float


def random_style(rng, palette=None):
    pal = palette or (
        "cosine" if rng.random() < 0.15 else str(rng.choice(list(PALETTES.keys()))))
    lighting = rng.random() < 0.8
    return Style(
        palette=pal,
        cycles=float(rng.uniform(1.5, 4.0)),
        phase=float(rng.uniform(0, 1)),
        lighting=lighting,
        light_angle=float(rng.uniform(0, 2 * math.pi)),
        h2=float(rng.uniform(1.0, 2.0)),
        ambient=float(rng.uniform(0.12, 0.3)),
        diffuse=float(rng.uniform(0.9, 1.2)),
        specular=float(rng.uniform(0.0, 0.35)) if lighting else 0.0,
        shine=float(rng.uniform(12, 40)),
        edge=int(rng.choice([0, 1, 1, 1, 2])),
        edge_w=float(rng.uniform(0.4, 0.9)),
        edge_min=float(rng.uniform(0.15, 0.5)),
        glow=float(rng.uniform(0.15, 0.4)),
        interior=str(rng.choice(["black", "black", "tint"])),
        hue_start=0.0,
        hue_total=float(rng.choice([0.0, 0.0, rng.uniform(-120, 120)])),
        vignette=float(rng.uniform(0.15, 0.4)),
        exposure=float(rng.uniform(1.2, 2.0)),
    )


def color_params(style, lut_dark, lo, hi):
    cp = np.zeros(CP_SIZE, np.float64)
    cp[CP_LO] = lo
    cp[CP_INV] = 1.0 / max(hi - lo, 0.05)
    cp[CP_CYC] = style.cycles
    cp[CP_PHASE] = style.phase
    cp[CP_LIGHT] = 1.0 if style.lighting else 0.0
    cp[CP_LX] = math.cos(style.light_angle)
    cp[CP_LY] = math.sin(style.light_angle)
    cp[CP_H2] = style.h2
    cp[CP_AMB] = style.ambient
    cp[CP_DIF] = style.diffuse
    cp[CP_SPEC] = style.specular
    cp[CP_SHINE] = style.shine
    cp[CP_EDGE] = float(style.edge)
    cp[CP_EDGE_W] = style.edge_w if style.edge == 1 else style.edge_w * 0.6
    cp[CP_EDGE_MIN] = style.edge_min
    cp[CP_GLOW] = style.glow
    cp[CP_GR], cp[CP_GG], cp[CP_GB] = 1.0, 0.95, 0.85
    if style.interior == "tint":
        cp[CP_IR], cp[CP_IG], cp[CP_IB] = lut_dark * 0.35
    return cp


# ---------------------------------------------------------------------------
# Minibrot finding (Mandelbrot z^2 + c): period -> Newton nucleus -> size
# ---------------------------------------------------------------------------


@njit(cache=True)
def ball_period(cr, ci, r, maxp):
    """Lowest period of an atom whose domain touches the disc |c - c0| < r (0 if none)."""
    c = complex(cr, ci)
    z = 0j
    dz = 0j
    for n in range(1, maxp + 1):
        dz = 2.0 * z * dz + 1.0
        z = z * z + c
        az = abs(z)
        if az > 1e6:
            return 0
        if az < r * abs(dz):
            return n
    return 0


@njit(cache=True)
def nucleus(cr, ci, p, steps):
    c = complex(cr, ci)
    for _ in range(steps):
        z = 0j
        dz = 0j
        for _ in range(p):
            dz = 2.0 * z * dz + 1.0
            z = z * z + c
        if dz == 0:
            break
        step = z / dz
        c -= step
        if abs(step) < 1e-17 * max(abs(c), 1.0):
            break
    return c.real, c.imag


@njit(cache=True)
def mini_size(cr, ci, p):
    """Complex size/orientation of the minibrot with nucleus c and period p."""
    c = complex(cr, ci)
    z = 0j
    l = 1 + 0j
    b = 1 + 0j
    for _ in range(1, p):
        z = z * z + c
        l = 2.0 * z * l
        if l == 0:
            return 0.0, 0.0
        b = b + 1.0 / l
    d = b * l * l
    if d == 0:
        return 0.0, 0.0
    s = 1.0 / d
    return s.real, s.imag


def find_minibrot(spec, rng, path, s_floor, want_doublings, log=print):
    """Look for minibrots along a descent path; return (cx, cy, size_complex, period)."""
    found = []
    for (cx, cy, s) in path:
        for frac in (0.5, 0.25, 0.12):
            r = s * frac
            p = ball_period(cx, cy, r, 3000)
            if p == 0:
                continue
            nr, ni = nucleus(cx, cy, p, 64)
            if not (math.isfinite(nr) and math.isfinite(ni)):
                continue
            if math.hypot(nr - cx, ni - cy) > 3 * s:
                continue
            sr, si = mini_size(nr, ni, p)
            size = complex(sr, si)
            if not (0 < abs(size) < 1) or not math.isfinite(abs(size)):
                continue
            if abs(size) * spec.home_s * 1.2 < s_floor:
                continue
            found.append((nr, ni, size, p))
    if not found:
        return None
    # prefer the one whose depth is closest to what we wanted
    d = np.array([math.log2(spec.home_s / (abs(f[2]) * spec.home_s)) for f in found])
    per = np.array([f[3] for f in found], np.float64)
    # low periods are much cheaper to render (escape counts scale with period)
    best = np.argsort(np.abs(d - want_doublings) + 0.003 * per)[: 3]
    return found[int(rng.choice(best))]


# ---------------------------------------------------------------------------
# Target search
# ---------------------------------------------------------------------------


def view_score(grid):
    """How visually interesting is a probe grid? 0 = boring."""
    inside = grid < 0
    fi = inside.mean()
    if fi > 0.75:
        return 0.0
    esc = grid[~inside]
    if esc.size < 16:
        return 0.0
    L = np.log(np.where(inside, 1.0, grid) + 1.0)
    lo, hi = np.percentile(np.log(esc + 1.0), [2, 98])
    rng_ = hi - lo
    if rng_ < 1e-4:
        return 0.0
    L = np.where(inside, hi + 0.5 * rng_, L)  # treat interior as an edge
    thr = 0.06 * rng_
    dx = np.abs(np.diff(L, axis=1)) > thr
    dy = np.abs(np.diff(L, axis=0)) > thr
    detail = 0.5 * (dx.mean() + dy.mean())
    if detail > 0.3:  # beyond this it reads as noise, not structure
        detail = max(0.0, 0.3 - (detail - 0.3) * 0.6)
    # reward detail being spread over the frame, not crammed in a corner
    h, w = grid.shape
    tiles = 0
    for ty in range(3):
        for tx in range(3):
            sub = dx[ty * h // 3:(ty + 1) * h // 3, tx * (w - 1) // 3:(tx + 1) * (w - 1) // 3]
            if sub.size and sub.mean() > 0.02:
                tiles += 1
    spread = tiles / 9.0
    return float(detail * (0.4 + 0.6 * spread) * (1.0 - 0.8 * max(0.0, fi - 0.3)))


ITER_CAP = 50000


def adapt_iters(grid, current, cap=None):
    cap = cap or ITER_CAP
    esc = grid[grid >= 0]
    if esc.size == 0:
        return current
    need = int(1.6 * np.percentile(esc, 99.7) + 600)
    # grow gradually: a few slow-escaping pixels shouldn't explode the budget
    if current:
        need = min(need, current * 4)
    return int(min(cap, max(1000, need)))


def find_target(spec, rng, s_end, aspect, log=print):
    """Descend from the home view toward s_end, choosing rich spots."""
    gw = 128
    gh = max(8, int(round(gw * aspect)))
    lw, lh = 64, max(6, int(round(64 * aspect)))
    cx, cy, s = spec.home_x, spec.home_y, spec.home_s
    mi = min(2000, spec.cap)
    scores = []
    path = []
    while s > s_end * 1.0001:
        step = min(8.0, s / s_end)
        grid = spec.probe(cx, cy, s, gw, gh, min(mi * 2, spec.cap))
        mi = min(spec.cap // 2, max(mi, adapt_iters(grid, mi, spec.cap)))
        # candidate pixels: escaped, in the central part of the view
        jj, ii = np.mgrid[0:gh, 0:gw]
        central = (np.abs(ii + 0.5 - gw / 2) < gw * 0.3) & (np.abs(jj + 0.5 - gh / 2) < gh * 0.3)
        mask = central & (grid >= 0)
        if mask.sum() < 4:
            return None
        vals = grid[mask]
        cut = np.percentile(vals, 93)
        pick_idx = np.flatnonzero(mask.ravel() & (grid.ravel() >= cut))
        n_c = min(14, pick_idx.size)
        choices = rng.choice(pick_idx, size=n_c, replace=False)
        px = s / gw
        best = []
        for idx in choices:
            j, i = divmod(int(idx), gw)
            # jitter within the pixel so we don't sit on grid lattice points
            ccx = cx + (i + rng.uniform(0.2, 0.8) - gw / 2) * px
            ccy = cy + (gh / 2 - j - rng.uniform(0.2, 0.8)) * px
            sub = spec.probe(ccx, ccy, s / step, lw, lh, mi)
            sc = view_score(sub)
            # also peek one more level down so we don't walk into a dead end
            if sc > 0:
                sub2 = spec.probe(ccx, ccy, s / step / 8, lw, lh, mi)
                sc = 0.5 * sc + 0.5 * view_score(sub2)
            best.append((sc, ccx, ccy))
        best.sort(key=lambda t: -t[0])
        if best[0][0] <= 0.01:
            return None
        top = best[:4]
        w = np.array([t[0] for t in top]) ** 2
        sc, cx, cy = top[int(rng.choice(len(top), p=w / w.sum()))]
        scores.append(sc)
        s /= step
        path.append((cx, cy, s))
    return cx, cy, scores, mi, path


# ---------------------------------------------------------------------------
# Animation plan
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    fractal: dict
    target: tuple
    start_center: tuple
    s0: float
    s_end: float
    doublings: float
    fps: int
    frames: int
    zoom_frames: int
    rotation_start: float
    rotation_total: float
    center_alpha: float
    style: dict
    seed: int
    width: int
    height: int
    scales: np.ndarray = field(repr=False, default=None)
    centers: np.ndarray = field(repr=False, default=None)
    angles: np.ndarray = field(repr=False, default=None)


def ease_positions(n, a=0.07, b=0.2):
    t = np.linspace(0.0, 1.0, n)
    v = np.ones(n)
    m = t < a
    x = t[m] / a
    v[m] = x * x * (3 - 2 * x)
    m = t > 1 - b
    x = (1 - t[m]) / b
    v[m] = x * x * (3 - 2 * x)
    v = 0.5 * (v + np.roll(v, 1))
    v[0] = 0.0
    pos = np.cumsum(v)
    return pos / pos[-1]


def build_motion(plan, a=0.07, b=0.2, hold_s=2.0):
    fz = plan.zoom_frames
    pos = ease_positions(fz, a, b)
    hold = int(round(hold_s * plan.fps))
    pos = np.concatenate([pos, np.ones(hold)])
    ls0, ls1 = math.log(plan.s0), math.log(plan.s_end)
    scales = np.exp(ls0 + (ls1 - ls0) * pos)
    tx, ty = plan.target
    sx, sy = plan.start_center
    k = (scales / plan.s0) ** plan.center_alpha
    centers = np.stack([tx + (sx - tx) * k, ty + (sy - ty) * k], axis=1)
    n = len(scales)
    tt = np.arange(n) / max(n - 1, 1)
    # rotation eases in so the first frames are calm
    # rotation follows the zoom so the final composition settles
    angles = plan.rotation_start + plan.rotation_total * (0.85 * pos + 0.15 * tt)
    plan.frames = n
    plan.scales, plan.centers, plan.angles = scales, centers, angles
    return plan


# ---------------------------------------------------------------------------
# Keyframes
# ---------------------------------------------------------------------------

FADE = 0.3


@dataclass
class KeyInfo:
    k: int
    x0: float = 0.0
    y0: float = 0.0
    px: float = 0.0
    nx: int = 0
    ny: int = 0
    max_iter: int = 1000
    lo: float = 0.0
    hi: float = 1.0


def assign_keys(plan):
    lvl = np.log2(plan.s0 / plan.scales)
    K = max(1, int(math.ceil(plan.doublings - 1e-9)))
    kp = np.minimum(np.floor(lvl).astype(int), K - 1)
    kp = np.maximum(kp, 0)
    u = lvl - kp
    x = np.clip((u - (1 - FADE)) / FADE, 0, 1)
    w = x * x * (3 - 2 * x)
    w[kp + 1 > K - 1] = 0.0
    return K, kp, w


_RATE = None


def iteration_rate():
    """Measured iterations/second of this machine (used for time estimates)."""
    global _RATE
    if _RATE is None:
        probe_grid(0, 2, 0.0, 0.0, -1.3, 0.0, 1e-3, 8, 8, 1.0, 0.0, 100, 0.0)  # compile
        t = time.time()
        probe_grid(0, 2, 0.0, 0.0, -1.3, 0.0, 1e-3, 64, 36, 1.0, 0.0, 30000, 0.0)
        _RATE = 0.8 * 64 * 36 * 30000 / max(time.time() - t, 1e-3)
    return _RATE


def plan_keys(plan, spec, W, H, ss=1, budget_s=None, max_key_px=48_000_000):
    """Lay out keyframes and pre-pass them. Returns (keys, kp, wb, est_seconds)."""
    K, kp, wb = assign_keys(plan)
    radius = 0.5 * math.sqrt(1 + (H / W) ** 2)
    keys = []
    for k in range(K):
        prim = (kp == k) & (wb < 1)
        sec = (kp == k - 1) & (wb > 0)
        use = prim | sec
        if not use.any():
            use = kp == k
        s = plan.scales[use]
        c = plan.centers[use]
        R = s * radius
        x_lo = float(np.min(c[:, 0] - R))
        x_hi = float(np.max(c[:, 0] + R))
        y_lo = float(np.min(c[:, 1] - R))
        y_hi = float(np.max(c[:, 1] + R))
        pmask = kp == k
        px = float(np.min(plan.scales[pmask])) / W if pmask.any() else float(np.min(s)) / W
        nx = int(math.ceil((x_hi - x_lo) / px)) + 4
        ny = int(math.ceil((y_hi - y_lo) / px)) + 4
        if nx * ny > max_key_px:
            f = math.sqrt(nx * ny / max_key_px)
            px *= f
            nx = int(math.ceil((x_hi - x_lo) / px)) + 4
            ny = int(math.ceil((y_hi - y_lo) / px)) + 4
        nx += (-nx) % 4
        ny += (-ny) % 4
        xc, yc = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
        keys.append(KeyInfo(k, xc - nx * px / 2, yc + ny * px / 2, px, nx, ny))

    # pre-pass: iteration limits + color normalisation per key
    gw = 192
    gh = max(8, int(round(gw * H / W)))
    mi = min(2000, spec.cap)
    los, his, mis = [], [], []
    rate = iteration_rate()
    est = 0.0
    for key in keys:
        idx = np.flatnonzero(kp == key.k)
        f = int(idx[len(idx) // 2]) if idx.size else 0
        cx, cy = plan.centers[f]
        grid, cost = spec.probe_cost(cx, cy, plan.scales[f], gw, gh, min(spec.cap, mi * 3),
                                     plan.angles[f])
        mi = max(mi, adapt_iters(grid, mi, spec.cap))
        est += key.nx * key.ny * ss * ss * float(np.minimum(cost, mi).mean()) / rate
        if budget_s is not None and est > budget_s:
            return keys, kp, wb, float("inf")
        esc = grid[grid >= 0]
        if esc.size > 10:
            lo, hi = np.percentile(np.log(esc + 1.0), [2, 98])
        else:
            lo, hi = (los[-1], his[-1]) if los else (0.0, 1.0)
        los.append(lo)
        his.append(max(hi, lo + 0.08))
        mis.append(mi)
    los, his = gaussian_smooth(np.array(los)), gaussian_smooth(np.array(his))
    for key, lo, hi, m in zip(keys, los, his, mis):
        key.lo, key.hi, key.max_iter = float(lo), float(hi), int(m)
    return keys, kp, wb, est


def gaussian_smooth(a, sigma=1.3):
    if len(a) < 3:
        return a
    r = int(3 * sigma)
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    p = np.pad(a, r, mode="edge")
    return np.convolve(p, k, mode="valid")


def downsample2(img):
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    v = img[:h, :w]
    return (0.25 * (v[0::2, 0::2] + v[1::2, 0::2] + v[0::2, 1::2] + v[1::2, 1::2])).astype(np.float32)


class KeyCache:
    def __init__(self, spec, keys, style, lut, lut_dark, ss):
        self.spec, self.keys, self.style = spec, keys, style
        self.lut, self.lut_dark, self.ss = lut, lut_dark, ss
        self.cache = {}
        self.render_time = 0.0

    def get(self, k):
        if k not in self.cache:
            t0 = time.time()
            key = self.keys[k]
            cp = color_params(self.style, self.lut_dark, key.lo, key.hi)
            eps2 = PERIOD_EPS2
            m0 = render_key(self.spec.kind, self.spec.power, self.spec.jx, self.spec.jy,
                            key.x0, key.y0, key.px, key.nx, key.ny, self.ss,
                            key.max_iter, eps2, cp, self.lut)
            m1 = downsample2(m0)
            m2 = downsample2(m1)
            self.cache[k] = (m0, m1, m2, key)
            self.render_time += time.time() - t0
        return self.cache[k]

    def evict_below(self, k):
        for kk in [kk for kk in self.cache if kk < k]:
            del self.cache[kk]


def render_frame(f, plan, kc, kp, wb, W, H, out, style):
    k = int(kp[f])
    a0, a1, a2, ka = kc.get(k)
    w = float(wb[f])
    if w > 0:
        b0, b1, b2, kb = kc.get(k + 1)
    else:
        b0, b1, b2, kb = a0, a1, a2, ka
    cx, cy = plan.centers[f]
    ang = plan.angles[f]
    hue = hue_matrix(style.hue_start + style.hue_total * f / max(plan.frames - 1, 1))
    composite(W, H, float(cx), float(cy), float(plan.scales[f]), math.cos(ang), math.sin(ang),
              a0, a1, a2, ka.x0, ka.y0, ka.px,
              b0, b1, b2, kb.x0, kb.y0, kb.px, w,
              hue, style.vignette, style.exposure, out)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

FRACTAL_WEIGHTS = {"mandelbrot": 0.6, "multibrot": 0.2, "julia": 0.2}


def choose_everything(args, seed, W, H, log=print):
    rng = np.random.default_rng(seed)
    aspect = H / W
    names = list(FRACTAL_WEIGHTS)
    probs = np.array([FRACTAL_WEIGHTS[n] for n in names])
    budget = args.budget * 60.0
    fallback = None
    for attempt in range(40):
        fname = args.fractal or str(rng.choice(names, p=probs / probs.sum()))
        spec = make_fractal(fname, rng)
        # deepest zoom double precision can render cleanly at this width
        mag = max(1.0, abs(complex(spec.home_x, spec.home_y)) + 2.0)
        s_floor = W * 40 * EPS * mag * (2 if args.aa > 1 else 1)
        d_max = math.log2(spec.home_s / s_floor)
        if spec.kind == KIND_JULIA:
            d_max = min(d_max, 30.0)
        if args.depth:
            doublings = min(args.depth, d_max)
        else:
            doublings = float(rng.uniform(0.6, 1.0) * d_max)
        s_end = spec.home_s * 2.0 ** -doublings
        mini = spec.kind == KIND_MANDEL and spec.power == 2
        res = find_target(spec, rng, s_floor * 4 if mini else s_end, aspect, log)
        if res is None:
            continue
        tx, ty, scores, _, path = res
        if min(scores) < 0.03 or np.mean(scores) < 0.06:
            continue
        end_angle = None
        if not mini:
            g = spec.probe(tx, ty, s_end, 96, max(6, int(96 * aspect)), spec.cap)
            ins = g < 0
            if 0.01 < ins.mean() < 0.5:
                jj, ii = np.nonzero(ins)
                px = s_end / 96
                tx += (ii.mean() + 0.5 - 48) * px
                ty += (g.shape[0] / 2 - jj.mean() - 0.5) * px
            # back off until the mini is framed rather than filling the screen
            for _ in range(12):
                g = spec.probe(tx, ty, s_end, 64, max(6, int(64 * aspect)), spec.cap)
                if (g < 0).mean() <= 0.3:
                    break
                s_end *= 1.5
        if mini:
            # dive into a real minibrot and end with it framed upright
            mb = find_minibrot(spec, rng, path, s_floor, doublings, log)
            if mb is None:
                continue
            nr, ni, size, period = mb
            frame_c = complex(nr, ni) + size * complex(spec.home_x, spec.home_y)
            tx, ty = frame_c.real, frame_c.imag
            s_end = max(abs(size) * spec.home_s * float(rng.uniform(1.0, 2.6)), s_floor)
            end_angle = math.atan2(size.imag, size.real)
            log(f"  minibrot: period {period}, size {abs(size):.3e}")
        # where does the video start?
        if rng.random() < 0.65 or args.from_home:
            s0 = spec.home_s
            start = (spec.home_x, spec.home_y)
            rot0 = 0.0
        else:
            skip = float(rng.uniform(3, 9))
            s0 = spec.home_s * 2.0 ** -skip
            start = (tx + rng.normal(0, 0.15) * s0, ty + rng.normal(0, 0.15) * s0)
            rot0 = float(rng.uniform(0, 2 * math.pi))
        D = math.log2(s0 / s_end)
        speed = args.speed or float(rng.uniform(0.85, 1.25))
        zoom_frames = args.frames or int(round(D / speed * args.fps / (1 - (0.07 + 0.2) / 2)))
        if args.duration:
            zoom_frames = int(round(args.duration * args.fps))
        if end_angle is not None:
            d = (end_angle - rot0 + math.pi) % (2 * math.pi) - math.pi
            rot_total = d + float(rng.choice([-1, 0, 0, 1])) * 2 * math.pi
        else:
            rot_total = 0.0 if rng.random() < 0.2 else float(rng.choice([-1, 1]) * rng.uniform(0.5, 3.0) * math.pi)
        style = random_style(rng, args.palette)
        plan = Plan(fractal=asdict(spec), target=(tx, ty), start_center=start, s0=s0, s_end=s_end,
                    doublings=D, fps=args.fps, frames=0, zoom_frames=max(zoom_frames, 2),
                    rotation_start=rot0, rotation_total=rot_total,
                    center_alpha=float(rng.uniform(1.3, 1.8)), style=asdict(style), seed=seed,
                    width=W, height=H)
        build_motion(plan)
        # validate the actual camera path: every stage must look good
        ok = True
        checks = np.linspace(0, plan.zoom_frames - 1, 10).astype(int)
        mi = min(2000, spec.cap)
        for f in checks[1:-1]:
            g = spec.probe(*plan.centers[f], plan.scales[f], 64, max(6, int(64 * aspect)),
                           min(spec.cap, mi * 2), plan.angles[f])
            mi = max(mi, adapt_iters(g, mi, spec.cap))
            if view_score(g) < 0.02:
                ok = False
                break
        if not ok:
            continue
        # is it affordable? (deep minibrots near parabolic points can need hours)
        limit = budget if fallback is None else min(budget * 3, fallback[0])
        keys, kp, wb, est = plan_keys(plan, spec, W, H, args.aa, limit)
        if est > budget:
            log(f"  candidate {attempt + 1}: {spec.name} too slow "
                f"({'>' if not math.isfinite(est) else '~'}{min(est, limit) / 60:.0f} min), searching on")
            if math.isfinite(est) and (fallback is None or est < fallback[0]):
                fallback = (est, (spec, style, plan, (keys, kp, wb, est)))
            continue
        log(f"  found target after {attempt + 1} attempt(s), est. render {est / 60:.1f} min")
        return spec, style, plan, rng, (keys, kp, wb, est)
    if fallback is not None:
        log(f"  using cheapest candidate found (est. {fallback[0] / 60:.0f} min)")
        spec, style, plan, keyplan = fallback[1]
        return spec, style, plan, rng, keyplan
    raise RuntimeError("could not find a good zoom target; try another seed or a larger --budget")


def open_ffmpeg(path, W, H, fps, crf, preset):
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def describe(spec, style, plan):
    bits = [spec.name]
    if spec.kind == KIND_JULIA:
        bits.append(f"c={spec.jx:+.5f}{spec.jy:+.5f}i")
    bits.append(f"zoom 2^{plan.doublings:.1f} (x{2 ** plan.doublings:.2e})")
    bits.append(f"palette={style.palette}")
    bits.append(["flat", "lit"][style.lighting])
    bits.append(["", "ink", "glow"][style.edge] or "plain")
    bits.append(f"{plan.frames / plan.fps:.1f}s")
    return ", ".join(b for b in bits if b)


def unique_path(path):
    base, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(path):
        path = f"{base}_{i}{ext}"
        i += 1
    return path


def run_one(args, seed):
    W, H = args.width, args.height
    t_start = time.time()
    print(f"\n=== seed {seed} ===")
    print("Searching for a beautiful zoom target...")
    spec, style_d, plan, rng, (keys, kp, wb, est) = choose_everything(args, seed, W, H)
    style = Style(**plan.style)
    lut, lut_dark, label = build_lut(style.palette, rng)
    print("  " + describe(spec, style, plan))
    print(f"  target = {plan.target[0]:.17g} {plan.target[1]:+.17g}i")

    print(f"  search took {time.time() - t_start:.1f}s")
    kc = KeyCache(spec, keys, style, lut, lut_dark, args.aa)
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"fractal_{spec.name}_{seed}")
    meta = {k: v for k, v in asdict(plan).items() if k not in ("scales", "centers", "angles")}
    meta["palette_label"] = label
    meta["max_iter_final"] = keys[-1].max_iter
    meta["command"] = " ".join(sys.argv)

    if args.contact:
        path = unique_path(stem + "_contact.png")
        contact_sheet(plan, kc, kp, wb, W, H, style, path)
        print(f"Saved contact sheet: {path}  ({time.time() - t_start:.0f}s)")
        return path

    frame = np.empty((H, W, 3), np.uint8)
    video_path = unique_path(stem + ".mp4")
    proc = None
    if not args.no_video:
        if shutil.which("ffmpeg") is None:
            print("ffmpeg not found on PATH - saving PNG frames instead")
            args.png_dir = args.png_dir or stem + "_frames"
        else:
            proc = open_ffmpeg(video_path, W, H, args.fps, args.crf, args.preset)
    if args.png_dir:
        os.makedirs(args.png_dir, exist_ok=True)

    poster_f = int(plan.zoom_frames * 0.8)
    print(f"Rendering {plan.frames} frames at {W}x{H} using {len(keys)} keyframes "
          f"(final max_iter {keys[-1].max_iter})")
    it = range(plan.frames)
    bar = tqdm(it, ncols=90, unit="f") if tqdm else it
    try:
        for f in bar:
            kc.evict_below(int(kp[f]))
            render_frame(f, plan, kc, kp, wb, W, H, frame, style)
            if proc:
                proc.stdin.write(frame.tobytes())
            if args.png_dir:
                Image.fromarray(frame).save(os.path.join(args.png_dir, f"frame_{f:06d}.png"),
                                            compress_level=1)
            if f == poster_f:
                Image.fromarray(frame).save(stem + "_poster.png")
            if tqdm:
                bar.set_postfix(key=f"{int(kp[f]) + 1}/{len(keys)}")
    finally:
        if proc:
            proc.stdin.close()
            proc.wait()
    Image.fromarray(frame).save(stem + "_final.png")
    with open(stem + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    dt = time.time() - t_start
    print(f"Done in {dt / 60:.1f} min (keyframes {kc.render_time / 60:.1f} min)")
    if proc:
        print(f"Saved video:  {video_path}")
    print(f"Saved stills: {stem}_poster.png, {stem}_final.png")
    return video_path


def contact_sheet(plan, kc, kp, wb, W, H, style, path, n=12):
    """Render n frames along the path into one image (quick look)."""
    frames = np.linspace(0, plan.frames - 1, n).astype(int)
    cols = 4
    rows = math.ceil(n / cols)
    tw, th = W // 2, H // 2
    sheet = Image.new("RGB", (cols * tw, rows * th))
    buf = np.empty((H, W, 3), np.uint8)
    for idx, f in enumerate(frames):
        kc.evict_below(int(kp[f]))
        render_frame(f, plan, kc, kp, wb, W, H, buf, style)
        img = Image.fromarray(buf).resize((tw, th), Image.LANCZOS)
        sheet.paste(img, ((idx % cols) * tw, (idx // cols) * th))
    sheet.save(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Random deep-zoom fractal video generator")
    p.add_argument("--seed", type=int, help="reproduce a specific video")
    p.add_argument("--count", type=int, default=1, help="number of videos to make")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--aa", type=int, default=1, choices=[1, 2, 3],
                   help="extra supersampling per keyframe pixel (keyframes already give ~2x)")
    p.add_argument("--fractal", choices=["mandelbrot", "multibrot", "multibrot3", "multibrot4",
                                         "multibrot5", "julia"])
    p.add_argument("--palette", choices=list(PALETTES) + ["cosine"])
    p.add_argument("--depth", type=float, help="zoom depth in doublings (max ~36 at 1080p)")
    p.add_argument("--speed", type=float, help="zoom speed in doublings per second (default ~1)")
    p.add_argument("--duration", type=float, help="zoom duration in seconds (overrides speed)")
    p.add_argument("--frames", type=int, help="number of zoom frames (overrides speed)")
    p.add_argument("--from-home", action="store_true", help="always start fully zoomed out")
    p.add_argument("--max-iter", type=int, default=50000,
                   help="iteration cap (higher = fewer black specks in deep zooms, slower)")
    p.add_argument("--budget", type=float, default=20.0,
                   help="rough render-time budget in minutes; slower zoom targets are skipped")
    p.add_argument("--out", default="renders", help="output folder")
    p.add_argument("--crf", type=int, default=17, help="x264 quality (lower = better/bigger)")
    p.add_argument("--preset", default="slow", help="x264 preset")
    p.add_argument("--png-dir", help="also write PNG frames here")
    p.add_argument("--no-video", action="store_true", help="skip the MP4")
    p.add_argument("--contact", action="store_true",
                   help="only render a contact sheet of 12 frames along the path")
    p.add_argument("--preview", action="store_true", help="fast 640x360 @ 24fps")
    args = p.parse_args(argv)
    if args.preview:
        args.width, args.height, args.fps = 640, 360, 24
    if args.fractal and args.fractal.startswith("multibrot") and len(args.fractal) == 9:
        args.fractal = "multibrot"
    return args


def main(argv=None):
    args = parse_args(argv)
    global ITER_CAP
    ITER_CAP = args.max_iter
    base = args.seed if args.seed is not None else int.from_bytes(os.urandom(3), "little")
    for i in range(args.count):
        run_one(args, base + i)


if __name__ == "__main__":
    main()
