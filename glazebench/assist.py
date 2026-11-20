"""Point-pair assisted registration.

The regular workflow asks the operator to find the same four tile corners in
every capture. When a raking-light capture is shifted or has a corner outside
the frame, the operator instead marks a few visible common locations. This
module fits the homography from those pairs and, importantly, reports where
the suggestion has no real evidence: extrapolated areas, areas the side photo
does not cover, and areas whose local pattern is repetitive enough that a
wrong match would look just as plausible.
"""
import math

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .art import base_rgba
from .geometry import (
    DEFAULT_CORNERS,
    SampleFrame,
    apply_homography,
    fit_homography,
    homography,
    point_pairs_parameter,
    specimen_frame,
    warp_capture,
)
from .inspection import cross_edge_mismatch
from .model import InputError, identity
from .photos import luminance, open_working, uint8_image

MAX_POINT_PAIRS = 12
ASSIST_GRID_LIMIT = 860
BLOCK_RADIUS = 7
SEARCH_RADIUS = 26
# Distinctiveness below this means another location matches almost as well.
WEAK_DISTINCTIVENESS = 0.12
RESIDUAL_WARN_NORM = 0.012
MIN_ANCHOR_SPREAD = 0.12
EDGE_WARN_QUANTILE = 0.55
AMBER = (236, 204, 114)
ANCHOR_COLOR = (234, 193, 129)
SUPPORTED_COLOR = (91, 168, 143)
UNSUPPORTED_COLOR = (142, 151, 145)
WARP_BLUE = (150, 214, 235)


def _downscaled_gray(linear):
    height, width = linear.shape[:2]
    step = max(1, math.ceil(max(height, width) / ASSIST_GRID_LIMIT))
    if step > 1:
        linear = linear[::step, ::step]
    return luminance(linear).astype(np.float32)


def _block_sad(image, center, block_radius, search_radius):
    """Mean absolute difference of a template against every search offset.

    Returns a score square centered on the marked location. Border pixels are
    edge-padded so anchors near the frame edge still get a complete map.
    """
    cy, cx = center
    pad = block_radius + search_radius
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="edge")
    origin_y, origin_x = cy + search_radius, cx + search_radius
    span = 2 * block_radius + 1
    block = padded[origin_y:origin_y + span, origin_x:origin_x + span]
    side = 2 * search_radius + 1
    scores = np.empty((side, side), dtype=np.float32)
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            candidate = padded[origin_y + dy:origin_y + dy + span, origin_x + dx:origin_x + dx + span]
            scores[dy + search_radius, dx + search_radius] = np.mean(np.abs(candidate - block))
    return scores


def local_distinctiveness(base_gray, side_gray, positions):
    """How uniquely each anchor location is explained by the side capture.

    The score is the normalized gap between the best template offset and the
    second best: ``(second - best) / (median - best)``. A distinctive mark has
    one clear winner and a score near 1. On repeated stripes many offsets tie
    (the gap is near zero, and on a perfectly periodic direction it is exactly
    zero), so a low score flags a correspondence that image content cannot
    disambiguate on its own.
    """
    height, width = base_gray.shape
    scores = []
    mark_errors = []
    for x_norm, y_norm in positions:
        cx = int(round(x_norm * (width - 1)))
        cy = int(round(y_norm * (height - 1)))
        if (
            cx - BLOCK_RADIUS - SEARCH_RADIUS < 0 or cy - BLOCK_RADIUS - SEARCH_RADIUS < 0
            or cx + BLOCK_RADIUS + SEARCH_RADIUS >= width or cy + BLOCK_RADIUS + SEARCH_RADIUS >= height
        ):
            continue
        sad = _block_sad(side_gray, (cy, cx), BLOCK_RADIUS, SEARCH_RADIUS)
        flat = np.sort(sad.ravel())
        best = float(flat[0])
        second = float(flat[1])
        median = float(np.median(flat))
        gap = (second - best) / max(median - best, 1e-6)
        scores.append(float(np.clip(gap, 0.0, 1.0)))
        mark_errors.append(float(sad[SEARCH_RADIUS, SEARCH_RADIUS]))
    if not scores:
        return None
    score_array = np.asarray(scores, dtype=np.float32)
    return {
        "scores": score_array,
        "median_score": float(np.median(score_array)),
        "weak_fraction": float(np.mean(score_array < WEAK_DISTINCTIVENESS)),
        "mark_sad_median": float(np.median(mark_errors)),
    }


def convex_hull(points):
    ordered = sorted(set(map(tuple, points)))
    if len(ordered) <= 1:
        return ordered

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def points_in_polygon(points, polygon):
    """Vectorized ray-crossing containment test for an (N, 2) point array."""
    vertices = np.asarray(polygon, dtype=np.float64)
    count = len(vertices)
    inside = np.zeros(len(points), dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    for index in range(count):
        x1, y1 = vertices[index]
        x2, y2 = vertices[(index + 1) % count]
        crosses = (y1 > y) != (y2 > y)
        hit_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= crosses & (x < hit_x)
    return inside


def anchor_distance_field(shape, anchors):
    height, width = shape
    hull = convex_hull([[x * (width - 1), y * (height - 1)] for x, y in anchors])
    if len(hull) < 3:
        center = np.mean(hull or [[width / 2, height / 2]], axis=0)
        yy, xx = np.mgrid[0:height, 0:width]
        return np.hypot(xx - center[0], yy - center[1]).astype(np.float32), hull
    yy, xx = np.mgrid[0:height, 0:width]
    points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    inside = points_in_polygon(points, hull).reshape(shape)
    if inside.all():
        return np.zeros(shape, dtype=np.float32), hull
    return ndimage.distance_transform_edt(~inside).astype(np.float32), hull


def suggest_registration(store, tile, photo_id, payload):
    photo_id = identity(photo_id, "Photo ID")
    plane = tile.get("plane")
    if plane is None:
        raise InputError("Set a base photo and mark the tile plane first.")
    if plane["photo_id"] == photo_id:
        raise InputError("Point-pair assistance aligns another capture to the base photo.")
    base_photo = next((item for item in tile["photos"] if item["id"] == plane["photo_id"]), None)
    side_photo = next((item for item in tile["photos"] if item["id"] == photo_id), None)
    if side_photo is None:
        raise InputError("The selected capture was removed from this specimen.")

    base_points, side_points = point_pairs_parameter(payload.get("pairs"), maximum=MAX_POINT_PAIRS)
    base_array = np.asarray(base_points, dtype=np.float64)
    side_array = np.asarray(side_points, dtype=np.float64)
    matrix = fit_homography(base_array, side_array)
    predicted_side = apply_homography(matrix, base_array)
    pixel_scale = float(np.hypot(side_photo["width"], side_photo["height"]) / math.sqrt(2))
    residuals = np.linalg.norm(predicted_side - side_array, axis=1)
    residual_px = residuals * pixel_scale

    spread = float(np.mean(np.linalg.norm(base_array - base_array.mean(axis=0), axis=1)))
    corners = apply_homography(matrix, np.asarray(DEFAULT_CORNERS, dtype=np.float64))

    destination = tuple(plane["size"])
    base_frame = specimen_frame(store.blobs, tile, base_photo["id"])
    rgb, alpha = open_working(store.blobs, side_photo)
    side_linear, side_valid, side_clipped = warp_capture(rgb, alpha, destination, matrix)
    side_frame = SampleFrame(side_linear, side_valid, side_photo, tuple(matrix.ravel()), None, side_clipped)

    base_gray = _downscaled_gray(base_frame.linear)
    side_gray = _downscaled_gray(side_frame.linear)
    distinct = local_distinctiveness(base_gray, side_gray, base_points)

    common = base_frame.valid & side_frame.valid
    edge_error, edge_confident = cross_edge_mismatch(base_frame, side_frame, common)
    edge_quantiles = (
        np.quantile(edge_error[edge_confident], (0.5, 0.9)).tolist() if edge_confident.any() else None
    )

    distance, hull = anchor_distance_field(base_frame.valid.shape, base_points)
    diagonal = float(np.hypot(*base_frame.extent))
    extrapolation = distance / max(diagonal, 1.0)
    anchor_mask = extrapolation <= 0.06

    warnings = []
    if spread < MIN_ANCHOR_SPREAD:
        warnings.append(
            "The marked pairs are clustered in one area. The transform is guessed by extrapolation "
            "farther away; spread pairs toward different parts of the tile."
        )
    if float(residuals.max()) > RESIDUAL_WARN_NORM:
        worst = int(np.argmax(residuals)) + 1
        warnings.append(
            f"Point pair {worst} does not fit the same geometry as the others "
            f"({residual_px[worst - 1]:.1f} px discrepancy). Check that the two marks hit the same spot."
        )
    if distinct is None:
        warnings.append("The pairs are too close to the photo edge to test the local pattern for repeats.")
    else:
        if distinct["weak_fraction"] >= 0.34:
            warnings.append(
                "Several marked spots sit on repeated stripes: a nearby offset matches almost equally well. "
                "Add pairs on distinctive marks (bubbles, specks, edges) before trusting the alignment."
            )
        if distinct["median_score"] < WEAK_DISTINCTIVENESS:
            warnings.append(
                "The two photos look locally interchangeable around the marks, so image content alone cannot "
                "confirm the shift. Treat striped areas as unsupported evidence."
            )
    if edge_quantiles is not None and edge_quantiles[0] > EDGE_WARN_QUANTILE:
        warnings.append(
            "Edge directions disagree across much of the overlap. The suggested alignment may be off by a "
            "stripe period; compare the checkerboard before saving."
        )
    if float(side_valid.mean()) < 0.15:
        warnings.append(
            "Less than 15% of the tile plane is covered by this photo. Most of the plane will stay unsupported."
        )
    off_frame = [
        index for index, point in enumerate(corners)
        if not (0 <= point[0] <= 1 and 0 <= point[1] <= 1)
    ]
    if off_frame:
        names = ", ".join(str(index + 1) for index in off_frame)
        warnings.append(
            f"Suggested corner(s) {names} fall outside the photo frame because that corner was not captured. "
            "Their position is extrapolated from the marked pairs and has no direct evidence."
        )

    checks = {
        "pair_count": len(base_points),
        "anchor_spread": round(spread, 4),
        "max_residual_norm": round(float(residuals.max()), 5),
        "max_residual_px": round(float(residual_px.max()), 2),
        "residual_px": [round(float(value), 2) for value in residual_px],
        "distinctiveness": None
        if distinct is None
        else {
            "median_second_best_gap": round(distinct["median_score"], 4),
            "weak_pair_fraction": round(distinct["weak_fraction"], 3),
        },
        "edge_agreement_quantiles": None if edge_quantiles is None else [round(value, 4) for value in edge_quantiles],
        "side_support_percent": round(100.0 * float(side_valid.mean()), 2),
        "common_support_percent": round(100.0 * float(common.mean()), 2),
        "extrapolated_percent": round(100.0 * float((~anchor_mask & base_frame.valid).mean()), 2),
        "off_frame_corners": off_frame,
    }
    evidence = evidence_overlay(
        base_frame, side_frame, base_points, side_points, anchor_mask, hull, matrix,
    )
    pairs = [
        {"base": base_points[index], "side": side_points[index]}
        for index in range(len(base_points))
    ]
    return {
        "corners": [[round(float(x), 6), round(float(y), 6)] for x, y in corners],
        "pairs": pairs,
        "residuals_px": checks["residual_px"],
        "warnings": warnings,
        "checks": checks,
        "evidence": evidence,
    }


def evidence_overlay(base_frame, side_frame, base_points, side_points, anchor_mask, hull, matrix):
    image = base_rgba(base_frame).convert("RGB")
    unsupported = (~side_frame.valid) & base_frame.valid
    extrapolated = (~anchor_mask) & side_frame.valid & base_frame.valid
    array = np.asarray(image).astype(np.float32)
    array[unsupported] = array[unsupported] * 0.45 + np.asarray(UNSUPPORTED_COLOR) * 0.55
    array[extrapolated] = array[extrapolated] * 0.55 + np.asarray(AMBER) * 0.45
    image = uint8_image(np.clip(array / 255.0, 0, 1), base_frame.valid.astype(np.float32)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size

    if len(hull) >= 3:
        draw.polygon([(x, y) for x, y in hull], outline=(255, 244, 214, 230), width=2)

    inverse = np.linalg.inv(matrix)
    radius = max(5, round(min(width, height) / 110))
    for index, (base, side) in enumerate(zip(base_points, side_points), start=1):
        warped_side = apply_homography(inverse, np.asarray([side], dtype=np.float64))[0]
        bx, by = base[0] * width, base[1] * height
        sx, sy = warped_side[0] * width, warped_side[1] * height
        draw.ellipse((bx - radius, by - radius, bx + radius, by + radius),
                     fill=ANCHOR_COLOR, outline=(31, 49, 39), width=2)
        draw.ellipse((sx - radius, sy - radius, sx + radius, sy + radius),
                     outline=WARP_BLUE, width=2)
        draw.line((bx, by, sx, sy), fill=(236, 121, 125, 220), width=1)
        draw.text((bx + radius + 3, by - radius - 2), str(index),
                  fill=(255, 247, 215), stroke_width=2, stroke_fill=(24, 36, 30))

    legend_y = height - 18
    draw.rectangle((10, legend_y - 12, width - 10, height - 4), fill=(24, 36, 30, 215))
    swatches = [
        (SUPPORTED_COLOR, "supported overlap"),
        (AMBER, "extrapolated, no anchor evidence"),
        (UNSUPPORTED_COLOR, "not covered by side photo"),
        (WARP_BLUE, "side point after warp"),
    ]
    x = 18
    for color, label in swatches:
        draw.rectangle((x, legend_y - 6, x + 11, legend_y + 5), fill=color)
        draw.text((x + 16, legend_y - 7), label, fill=(245, 241, 223))
        x += 16 + int(draw.textlength(label)) + 28
    return image.convert("RGBA")
