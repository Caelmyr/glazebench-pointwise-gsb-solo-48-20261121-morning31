import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .art import base_rgba, SEQUENTIAL
from .color import quantile_rows
from .defects import gaussian_supported, robust_noise
from .model import InputError
from .photos import luminance


DIRECTIONS = (
    ("horizontal", 0, 1, 0),
    ("rising diagonal", 1, -1, 45),
    ("vertical", 1, 0, 90),
    ("falling diagonal", 1, 1, 135),
)
MIN_TEXTURE_PIXELS = 100

# Local pattern windows. A window has to span several repeats before a
# spacing can be trusted, so these stay deliberately wide.
WINDOW_TARGETS = (72, 92, 116)
MIN_WINDOW = 56
MAX_WINDOW = 148
MAX_PATTERN_CELLS = 240
LOCAL_MIN_PIXELS = 220
MIN_WINDOW_SUPPORT = 0.82
MIN_COHERENCE = 0.80
MIN_FIRST_PEAK = 0.80
MIN_HARMONIC = 0.45
HARMONIC_RATIO = 0.55
MIN_DIFFERENCE_DEPTH = 0.85
MIN_REPEATS = 3
CMND_THRESHOLD = 0.50
PATTERN_SHADOW = (26, 34, 38)


def palette_color(normalized):
    boundaries = [stop[0] for stop in SEQUENTIAL]
    channels = np.asarray([color for _, color in SEQUENTIAL], dtype=np.float32)
    return tuple(
        int(round(float(np.interp(normalized, boundaries, channels[:, channel]))))
        for channel in range(3)
    )


def shift_overlap(array, dy, dx):
    rows, columns = array.shape
    if abs(dy) >= rows or abs(dx) >= columns:
        raise InputError("Texture spacing exceeds the registered image size.")
    if dy >= 0:
        a_rows = slice(0, rows - dy)
        b_rows = slice(dy, rows)
    else:
        a_rows = slice(-dy, rows)
        b_rows = slice(0, rows + dy)
    if dx >= 0:
        a_cols = slice(0, columns - dx)
        b_cols = slice(dx, columns)
    else:
        a_cols = slice(-dx, columns)
        b_cols = slice(0, columns + dx)
    return array[a_rows, a_cols], array[b_rows, b_cols]


def autocorrelation(signal, valid, direction, spacings):
    name, uy, ux, angle = direction
    table = []
    for step in spacings:
        dy, dx = uy * step, ux * step
        if abs(dy) >= signal.shape[0] or abs(dx) >= signal.shape[1]:
            continue
        a_valid, b_valid = shift_overlap(valid, dy, dx)
        common = a_valid & b_valid
        count = int(np.count_nonzero(common))
        if count < MIN_TEXTURE_PIXELS:
            continue
        whole_a, whole_b = shift_overlap(signal, dy, dx)
        a, b = whole_a[common], whole_b[common]
        a, b = a - a.mean(), b - b.mean()
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        result = float(np.dot(a, b) / norm) if norm > 1e-12 else None
        table.append({
            "step": step,
            "spacing_px": round(math.hypot(dy, dx), 3),
            "sample_pairs": count,
            "correlation": round(result, 5) if result is not None else None,
        })
    return {"direction": name, "angle_degrees": angle, "points": table}


def decorated_periods(curve, scale, direction):
    _, uy, ux, _ = direction
    if scale:
        for point in curve["points"]:
            step = point["step"]
            point["spacing_mm"] = round(math.hypot(scale[0] * ux, scale[1] * uy) * step, 4)
    peaks = []
    for before, point, after in zip(curve["points"], curve["points"][1:], curve["points"][2:]):
        scores = [entry["correlation"] for entry in (before, point, after)]
        if any(score is None for score in scores):
            continue
        if point["step"] != before["step"] + 1 or after["step"] != point["step"] + 1:
            continue
        if scores[1] > max(scores[0], scores[2], 0.15):
            peaks.append(point)
    curve["local_highs"] = sorted(peaks, key=lambda item: -item["correlation"])[:3]
    return curve


def lag_curves(signal, support, requested_max, scale):
    max_step = min(requested_max, min(signal.shape) // 3)
    diagonal = math.ceil(max_step / math.sqrt(2))
    curves = []
    for direction in DIRECTIONS:
        spacing = diagonal if direction[1] and direction[2] else max_step
        curve = autocorrelation(signal, support, direction, range(1, spacing + 1))
        curves.append(decorated_periods(curve, scale, direction))
    return curves


def gradient_pairs(field, mask, sigma):
    fill, _ = gaussian_supported(field, mask, sigma)
    core = ndimage.binary_erosion(mask, iterations=max(1, round(sigma)))
    dx = ndimage.sobel(fill, axis=1, mode="nearest") / 8
    dy = ndimage.sobel(fill, axis=0, mode="nearest") / 8
    return dx, dy, core


def orientation_strength(dx, dy, core):
    if np.count_nonzero(core) < MIN_TEXTURE_PIXELS:
        return None
    gx = dx[core]
    gy = dy[core]
    ixx = float(np.mean(gx * gx))
    iyy = float(np.mean(gy * gy))
    ixy = float(np.mean(gx * gy))
    scale = ixx + iyy
    if scale < 1e-8:
        return None
    angle = math.degrees(0.5 * math.atan2(2 * ixy, ixx - iyy))
    strength = min(math.hypot(ixx - iyy, 2 * ixy) / scale, 1)
    return {
        "ridge_angle_clockwise": round((angle + 90) % 180, 2),
        "coherence": round(strength, 4),
        "gradient_energy": round(scale, 7),
    }


def gradient_histogram(dx, dy, core):
    angle = (np.arctan2(dy[core], dx[core]) + np.pi / 2) % np.pi
    strength = np.hypot(dx[core], dy[core])
    hist, bounds = np.histogram(angle, bins=12, range=(0, np.pi), weights=strength)
    norm = max(float(np.sum(hist)), 1e-8)
    return [
        {
            "start_deg": round(math.degrees(bounds[index])),
            "end_deg": round(math.degrees(bounds[index + 1])),
            "fraction": round(float(value) / norm, 4),
        }
        for index, value in enumerate(hist)
    ]


def multiscale_detail(values, supported, sigma):
    scales = (1.0, max(2.0, sigma / 2), sigma)
    rows = []
    for scale in scales:
        mean, weight = gaussian_supported(values, supported, scale)
        core = supported & (weight >= 0.77)
        residual = values[core] - mean[core]
        if len(residual) < MIN_TEXTURE_PIXELS:
            continue
        rows.append({
            "radius_px": scale,
            "samples": len(residual),
            "roughness_rms": round(float(np.sqrt(np.mean(residual**2))), 5),
            "robust_range": quantile_rows(np.abs(residual), (0.50, 0.90, 0.95), 5),
        })
    return rows


def local_rms(signal, support, radius):
    squared, fraction = gaussian_supported(signal**2, support, radius)
    quality = support & (fraction >= 0.70)
    return np.sqrt(squared), quality


def window_layout(extent):
    width, height = extent
    short_side = min(extent)
    window = min(MAX_WINDOW, max(MIN_WINDOW, round(short_side * 0.20)))
    window = min(WINDOW_TARGETS, key=lambda candidate: abs(candidate - window))
    step = max(28, round(window * 0.62))
    columns = max(1, math.floor((width - window) / step) + 1)
    rows = max(1, math.floor((height - window) / step) + 1)
    while columns * rows > MAX_PATTERN_CELLS and step < window:
        step += 4
        columns = max(1, math.floor((width - window) / step) + 1)
        rows = max(1, math.floor((height - window) / step) + 1)
    return window, step, columns, rows


def shifted_pair(signal, support, dy, dx):
    a, b = shift_overlap(signal, dy, dx)
    va, vb = shift_overlap(support.astype(np.float32), dy, dx)
    common = (va >= 0.999) & (vb >= 0.999)
    count = int(np.count_nonzero(common))
    if count < LOCAL_MIN_PIXELS:
        return None, None, count
    left, right = a[common], b[common]
    left, right = left - left.mean(), right - right.mean()
    norm = float(np.linalg.norm(left) * np.linalg.norm(right))
    if norm <= 1e-12:
        return None, None, count
    correlation = float(np.dot(left, right) / norm)
    difference = float(np.mean((left - right) ** 2))
    return correlation, difference, count


def local_orientation(patch, interior):
    if np.count_nonzero(interior) < 64:
        return None
    smoothed = ndimage.gaussian_filter(patch, sigma=0.8, mode="nearest")
    gx = ndimage.sobel(smoothed, axis=1, mode="nearest") / 8
    gy = ndimage.sobel(smoothed, axis=0, mode="nearest") / 8
    gx, gy = gx[interior], gy[interior]
    ixx = float(np.mean(gx * gx))
    iyy = float(np.mean(gy * gy))
    ixy = float(np.mean(gx * gy))
    energy = ixx + iyy
    if energy < 1e-9 or not math.isfinite(energy):
        return None
    angle = math.degrees(0.5 * math.atan2(2 * ixy, ixx - iyy))
    ridge = (angle + 90) % 180
    coherence = min(math.hypot(ixx - iyy, 2 * ixy) / energy, 1)
    return ridge, coherence, energy


def lag_correlations(signal, support, nx, ny, max_lag):
    lags = np.arange(2, max_lag + 1, dtype=np.float64)
    correlations = np.full(lags.shape, np.nan)
    differences = np.full(lags.shape, np.nan)
    pairs = np.zeros(lags.shape, dtype=np.int32)
    for index, lag in enumerate(lags):
        dx = int(round(nx * lag))
        dy = int(round(ny * lag))
        if dx == 0 and dy == 0:
            continue
        correlation, difference, count = shifted_pair(signal, support, dy, dx)
        pairs[index] = count
        if correlation is not None:
            correlations[index] = correlation
            differences[index] = difference
    return lags, correlations, differences, pairs


def spacing_from_lag(signal, support, theta, max_lag):
    # Spacing is measured normal to the ridges. The ridge angle is measured
    # clockwise from the image x axis, so its normal is ridge minus 90.
    radians_value = math.radians(theta - 90)
    nx, ny = math.cos(radians_value), math.sin(radians_value)
    # The fundamental repeat is bounded by the user's maximum spacing, but the
    # second-cycle check has to reach roughly twice that lag, otherwise a wide
    # legitimate spacing could never be confirmed inside one window.
    verification_lag = min(int(max_lag * 2.4), min(signal.shape) - 4)
    lags, correlations, differences, pairs = lag_correlations(signal, support, nx, ny, verification_lag)
    usable = np.isfinite(differences)
    if np.count_nonzero(usable) < 3:
        return None
    # Cumulative mean normalized difference (YIN-style): the first dip under
    # the threshold is the fundamental repeat; higher harmonics are rejected
    # because earlier dips are always examined first.
    valid_diffs = np.where(usable, differences, 0.0)
    running = np.cumsum(valid_diffs)
    positions = np.cumsum(usable.astype(np.int32))
    cmnd = np.full(lags.shape, np.inf)
    safe = usable & (running > 1e-12) & (positions > 0)
    cmnd[safe] = valid_diffs[safe] * positions[safe] / running[safe]
    candidate = safe & (lags <= max_lag)
    below = np.nonzero(candidate & (cmnd <= CMND_THRESHOLD))[0]
    if len(below) == 0:
        return None
    first = int(below[0])
    # Require a genuine local dip rather than the shoulder of a deeper valley.
    while first + 1 < len(cmnd) and np.isfinite(cmnd[first + 1]) and cmnd[first + 1] < cmnd[first]:
        first += 1
    first_lag = float(lags[first])
    first_score = float(correlations[first])
    if first_score < MIN_FIRST_PEAK or first_lag < 2.5:
        return None
    harmonic_target = 2 * first_lag
    near = np.nonzero(usable & (np.abs(lags - harmonic_target) <= max(2.0, 0.22 * first_lag)))[0]
    harmonic = float(np.nanmax(correlations[near])) if len(near) else None
    repeats = min(signal.shape) / first_lag
    return {
        "spacing_px": round(first_lag, 3),
        "first_correlation": round(first_score, 4),
        "harmonic_correlation": round(harmonic, 4) if harmonic is not None else None,
        "visible_repeats": round(repeats, 2),
        "sample_pairs": int(pairs[first]),
        "difference_depth": round(1 - float(min(cmnd[first], 1.0)), 4),
    }


def local_pattern(signal, support, maximum_lag, scale, extent):
    window, step, columns, rows = window_layout(extent)
    # Fundamental spacing cannot exceed the user bound; harmonic verification
    # is extended inside spacing_from_lag up to what the window can support.
    max_lag = min(maximum_lag, window - 4)
    edge_margin = max(3, round(window * 0.12))
    cells = []
    uncertain = []
    for row_index in range(rows):
        top = min(row_index * step, signal.shape[0] - window)
        for column_index in range(columns):
            left = min(column_index * step, signal.shape[1] - window)
            region = (slice(top, top + window), slice(left, left + window))
            local_support = support[region]
            supported = int(np.count_nonzero(local_support))
            center_x = left + window / 2
            center_y = top + window / 2
            entry = {
                "row": row_index,
                "column": column_index,
                "bounds": [left, top, left + window, top + window],
                "normalized_bounds": [
                    round(left / extent[0], 6),
                    round(top / extent[1], 6),
                    round((left + window) / extent[0], 6),
                    round((top + window) / extent[1], 6),
                ],
                "normalized_center": [
                    round((center_x + 0.5) / extent[0], 6),
                    round((center_y + 0.5) / extent[1], 6),
                ],
                "supported_pixels": supported,
                "window_px": window,
                "periodic": False,
            }
            patch = signal[region]
            if supported < LOCAL_MIN_PIXELS or supported / window**2 < MIN_WINDOW_SUPPORT:
                entry["reason"] = "not enough supported pixels"
                uncertain.append(entry)
                cells.append(entry)
                continue
            # Keep occlusion and region edges out of the local statistics so a
            # sharp boundary cannot masquerade as a ridge direction.
            interior = ndimage.binary_erosion(local_support, iterations=edge_margin)
            if np.count_nonzero(interior) < LOCAL_MIN_PIXELS:
                entry["reason"] = "window dominated by an occlusion edge"
                uncertain.append(entry)
                cells.append(entry)
                continue
            orientation = local_orientation(patch, interior)
            if orientation is None:
                entry["reason"] = "no directional gradient"
                uncertain.append(entry)
                cells.append(entry)
                continue
            ridge, coherence, energy = orientation
            entry["ridge_angle_clockwise"] = round(ridge, 2)
            entry["coherence"] = round(coherence, 4)
            entry["gradient_energy"] = round(energy, 7)
            if coherence < MIN_COHERENCE:
                entry["reason"] = "mixed or weak direction"
                uncertain.append(entry)
                cells.append(entry)
                continue
            spacing = spacing_from_lag(patch, interior, ridge, max_lag)
            if spacing is None:
                entry["reason"] = "no repeatable spacing"
                uncertain.append(entry)
                cells.append(entry)
                continue
            entry.update(spacing)
            if spacing["difference_depth"] < MIN_DIFFERENCE_DEPTH:
                entry["reason"] = "repeat contrast is too shallow"
                uncertain.append(entry)
                cells.append(entry)
                continue
            if spacing["harmonic_correlation"] is None or spacing["harmonic_correlation"] < MIN_HARMONIC:
                entry["reason"] = "second repeat cycle not visible"
                uncertain.append(entry)
                cells.append(entry)
                continue
            if spacing["harmonic_correlation"] < HARMONIC_RATIO * spacing["first_correlation"]:
                entry["reason"] = "repeat weakens after the first cycle"
                uncertain.append(entry)
                cells.append(entry)
                continue
            if spacing["visible_repeats"] < MIN_REPEATS:
                entry["reason"] = "fewer than three visible repeats"
                uncertain.append(entry)
                cells.append(entry)
                continue
            entry["periodic"] = True
            entry.pop("reason", None)
            if scale:
                entry["spacing_mm"] = round(spacing["spacing_px"] * math.hypot(scale[0], scale[1]), 4)
            cells.append(entry)
    periodic = [entry for entry in cells if entry["periodic"]]
    summary = {
        "window_px": window,
        "step_px": step,
        "grid": [columns, rows],
        "periodic_cells": len(periodic),
        "uncertain_cells": len(uncertain),
        "minimum_visible_repeats": MIN_REPEATS,
        "coherence_floor": MIN_COHERENCE,
        "difference_depth_floor": MIN_DIFFERENCE_DEPTH,
    }
    if periodic:
        spacing_key = "spacing_mm" if scale else "spacing_px"
        values = sorted(entry[spacing_key] for entry in periodic)
        angles = np.array([entry["ridge_angle_clockwise"] for entry in periodic])
        summary.update({
            "spacing_unit": "mm" if scale else "px",
            "spacing_median": round(float(np.median(values)), 4),
            "spacing_range": [round(values[0], 4), round(values[-1], 4)],
            "angle_median": round(float(np.median(angles)), 2),
            "angle_range": [round(float(angles.min()), 2), round(float(angles.max()), 2)],
        })
    return {"summary": summary, "cells": cells}


def ridge_segment(center_x, center_y, angle, length):
    radians_value = math.radians(angle)
    run = math.cos(radians_value) * length / 2
    rise = math.sin(radians_value) * length / 2
    return (
        (center_x - run, center_y - rise),
        (center_x + run, center_y + rise),
    )


def pattern_overlay(frame, source_linear, pattern, spacing_scale):
    image = base_rgba(frame, source_linear)
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    short_side = min(width, height)
    glyph = max(13, round(short_side * 0.045))
    line_width = max(1, round(short_side / 420))
    # A None scale means every confirmed window shares one spacing: render the
    # glyphs in the middle of the ramp rather than pinning them to the low end.
    bounds = spacing_scale if spacing_scale and spacing_scale[0] > 0 else (0.0, 1.0)
    for cell in pattern["cells"]:
        cx = cell["normalized_center"][0] * width
        cy = cell["normalized_center"][1] * height
        if not cell["periodic"]:
            arm = glyph * 0.42
            draw.line((cx - arm, cy - arm, cx + arm, cy + arm), fill=PATTERN_SHADOW, width=line_width + 2)
            draw.line((cx + arm, cy - arm, cx - arm, cy + arm), fill=PATTERN_SHADOW, width=line_width + 2)
            draw.line((cx - arm, cy - arm, cx + arm, cy + arm), fill=(176, 178, 168, 190), width=line_width)
            draw.line((cx + arm, cy - arm, cx - arm, cy + arm), fill=(176, 178, 168, 190), width=line_width)
            continue
        value = cell.get("spacing_mm", cell["spacing_px"])
        if spacing_scale and spacing_scale[0] > 0:
            normalized = float(np.clip(
                (value - bounds[0]) / max(bounds[1] - bounds[0], 1e-9), 0, 1,
            ))
        else:
            normalized = 0.5
        color = palette_color(normalized)
        start, end = ridge_segment(cx, cy, cell["ridge_angle_clockwise"], glyph)
        draw.line((start[0], start[1], end[0], end[1]), fill=PATTERN_SHADOW, width=line_width + 2)
        draw.line((start[0], start[1], end[0], end[1]), fill=color, width=line_width)
    # Dim everything lightly so the local glyphs stay legible on bright glazes.
    veil = Image.new("RGBA", image.size, (18, 26, 30, 46))
    return Image.alpha_composite(image, veil)


def measure_texture(frame, rgb_linear, domain, parameters):
    valid = frame.valid & ~frame.clipped & domain
    values = luminance(rgb_linear)
    radius = parameters["background_radius"]
    background, weight = gaussian_supported(values, valid, radius)
    support = valid & (weight >= 0.78)
    if np.count_nonzero(support) < MIN_TEXTURE_PIXELS:
        raise InputError("The region needs a wider uninterrupted area at this texture scale.")
    high = values - background
    dx, dy, core = gradient_pairs(high, support, parameters["edge_smooth"])
    amplitudes, display_support = local_rms(high, support, max(radius * 0.45, 1))
    pattern = local_pattern(high, support, parameters["maximum_lag"], frame.scale, frame.extent)
    periodic = [cell for cell in pattern["cells"] if cell["periodic"]]
    spacing_scale = None
    if periodic:
        key = "spacing_mm" if frame.scale else "spacing_px"
        span = [min(cell[key] for cell in periodic), max(cell[key] for cell in periodic)]
        if span[1] > span[0]:
            spacing_scale = span
    result = {
        "supported_pixels": int(np.count_nonzero(support)),
        "grid_roughness": quantile_rows(amplitudes[display_support], (0.10, 0.50, 0.90, 0.99), 6),
        "detail_noise": round(robust_noise(high[support]), 6),
        "texture_orientation": orientation_strength(dx, dy, core),
        "orientation_histogram": gradient_histogram(dx, dy, core),
        "lag_curves": lag_curves(high, support, parameters["maximum_lag"], frame.scale),
        "scales": multiscale_detail(values, valid, radius),
        "local_pattern": pattern,
        "note": "Texture comes from visible image contrast under the selected capture and analysis scale.",
        "pattern_note": (
            "Local glyphs show ridge direction and repeat spacing only where several cycles repeat; "
            "crossed marks near occlusion or mixed texture stay marked uncertain."
        ),
    }
    return result, amplitudes, display_support, spacing_scale
