import numpy as np
from scipy import ndimage

from .color import float_list, quantile_rows
from .defects import gaussian_supported, robust_noise
from .model import InputError
from .photos import luminance


MIN_COMMON_PIXELS = 140
MIN_OVERLAP_FRACTION = 0.12


def captured_pair(primary, comparison, area):
    common = area & primary.valid & comparison.valid
    common_count = int(np.count_nonzero(common))
    available = int(np.count_nonzero(area))
    if common_count < MIN_COMMON_PIXELS or common_count < available * MIN_OVERLAP_FRACTION:
        raise InputError("The photos need a larger common area inside the registered specimen plane.")
    good = common & ~primary.clipped & ~comparison.clipped
    good_count = int(np.count_nonzero(good))
    if good_count < MIN_COMMON_PIXELS:
        raise InputError("Too little shared, unclipped detail remains in the selected area.")
    return common, good, {
        "requested_pixels": available,
        "common_pixels": common_count,
        "unclipped_pixels": good_count,
        "clipped_pairs": common_count - good_count,
        "overlap_percent": round(100 * common_count / available, 2),
        "pair_clipped_percent": round(100 * (common_count - good_count) / common_count, 2),
    }


def registration_pair(primary, comparison, darker=None):
    if primary.extent != comparison.extent:
        raise InputError("Both measurements require the same registered specimen plane.")
    if primary.photo["id"] == comparison.photo["id"]:
        raise InputError("Choose two different photos.")
    pairing = {
        "primary_id": primary.photo["id"],
        "primary_role": primary.photo["role"],
        "comparison_id": comparison.photo["id"],
        "comparison_role": comparison.photo["role"],
    }
    if darker is None:
        return pairing
    if darker.extent != primary.extent:
        raise InputError("The darker parallel capture needs the same registered specimen plane.")
    if darker.photo["id"] in (primary.photo["id"], comparison.photo["id"]):
        raise InputError("The darker parallel capture must be a third, distinct photo.")
    if darker.photo["role"] != "parallel-dark":
        raise InputError("The highlight-recovery capture needs the darker parallel-polarized role.")
    pairing["darker_id"] = darker.photo["id"]
    pairing["darker_role"] = darker.photo["role"]
    return pairing


def robust_exposure_ratio(primary, secondary, candidate):
    eligible = candidate & (primary >= 0.025) & (secondary >= 0.025) & (primary <= 0.91) & (secondary <= 0.91)
    ratios = primary[eligible] / secondary[eligible]
    if len(ratios) < 100:
        raise InputError("Too little middle-range shared detail to estimate the exposure ratio.")
    quantiles = np.quantile(ratios, (0.10, 0.25, 0.50, 0.75, 0.90))
    estimate = float(quantiles[2])
    if not 0.2 <= estimate <= 5.0:
        raise InputError("Shared intensity differs by too much to normalize these captures.")
    return estimate, {
        "estimated": True,
        "candidates": len(ratios),
        "ratio_quantiles_10_25_50_75_90": float_list(quantiles, 5),
    }


def darker_exposure_ratio(primary, darker, candidate):
    ratio, method = robust_exposure_ratio(primary, darker, candidate)
    if ratio < 1.05:
        raise InputError(
            "The darker parallel capture is not actually darker. Shoot it at a lower exposure so the bright spots stay below white."
        )
    return ratio, method


def blend_parallel_exposures(primary, darker, area):
    """Merge a normal and a darker parallel-polarized frame into one luminance field.

    The darker frame is normalized to the primary exposure with a single global
    gain estimated from shared, unclipped detail. Each output pixel is taken
    whole from one source (never averaged across the join), so a real bright
    spot cannot gain an artificial edge where the two exposures meet.
    """
    direct = luminance(primary.linear)
    dim = luminance(darker.linear)
    shared = area & primary.valid & darker.valid & ~primary.clipped & ~darker.clipped
    factor, method = darker_exposure_ratio(direct, dim, shared)
    scaled = dim * factor
    # The darker frame only takes over where the primary has reached its ceiling
    # and the darker frame still holds usable, non-black, non-white detail.
    recoverable = area & primary.valid & darker.valid & primary.clipped & ~darker.clipped
    merged = direct.copy()
    merged[recoverable] = scaled[recoverable]
    dark_ceiling = area & primary.valid & darker.valid & primary.clipped & darker.clipped
    return merged, recoverable, dark_ceiling, factor, method


def fixed_gain(value):
    return value, {"estimated": False, "ratio": value}


def glare_scale(parallel, cross, estimated_area, strategy, commanded_gain):
    if strategy == "fixed":
        return fixed_gain(commanded_gain)
    eligible = estimated_area & (parallel >= 0.028) & (cross >= 0.028) & (parallel < 0.94) & (cross < 0.94)
    ratio = parallel[eligible] / cross[eligible]
    if len(ratio) < 100:
        raise InputError("No stable middle-intensity tie area is available to balance the two photos.")
    lo, mid, hi = np.quantile(ratio, (0.20, 0.50, 0.82))
    gain = float(lo)
    if not 0.2 <= gain <= 5:
        raise InputError("Polarized captures require a more consistent exposure.")
    return gain, {
        "estimated": True,
        "candidates": len(ratio),
        "reason": "Lower-ratio samples reduce the influence of bright specular pixels.",
        "ratio_quantiles_20_50_82": float_list([lo, mid, hi], 5),
    }


def hotspot_summaries(hot, frame, minimum):
    labels, _ = ndimage.label(hot)
    result = []
    width, height = frame.extent
    for index, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None:
            continue
        rows, cols = region
        count = int(np.count_nonzero(labels[region] == index))
        if count < minimum:
            continue
        result.append({
            "pixels": count,
            "bounds": [cols.start / width, rows.start / height, cols.stop / width, rows.stop / height],
            "area_mm2": round(count * frame.scale[0] * frame.scale[1], 3) if frame.scale else None,
        })
    result.sort(key=lambda item: -item["pixels"])
    return result[:24], len(result)


def measure_glare(parallel, cross, area, calibration_area, parameters, darker=None):
    pairing = registration_pair(parallel, cross, darker)
    if parallel.photo["role"] != "parallel" or cross.photo["role"] != "cross":
        raise InputError("Polarized glare needs one parallel-polarized and one cross-polarized capture.")
    common, measure, overlap = captured_pair(parallel, cross, area)
    subdued = luminance(cross.linear)
    dark_exposure = None
    recovered = np.zeros(area.shape, dtype=bool)
    if darker is None:
        direct = luminance(parallel.linear)
        valid_pixels = measure
    else:
        merged, recovered, still_clipped, dark_factor, dark_method = blend_parallel_exposures(parallel, darker, area)
        direct = merged
        # A pixel is measurable when the primary holds detail, or when the darker
        # capture genuinely recovers it. Anything clipped in both, or outside the
        # darker frame's support, stays unmeasured.
        valid_pixels = common & (
            (~parallel.clipped & ~cross.clipped)
            | (recovered & ~cross.clipped)
        )
        dark_supported = area & parallel.valid & darker.valid
        dark_unmeasured = area & parallel.valid & ~darker.valid
        dark_exposure = {
            "capture_id": darker.photo["id"],
            "capture_name": darker.photo["name"],
            "normalization_gain": round(dark_factor, 5),
            "balance": dark_method,
            "shared_pixels": int(np.count_nonzero(dark_supported)),
            "recovered_pixels": int(np.count_nonzero(recovered)),
            "recovered_percent_of_measured": round(
                float(100 * np.count_nonzero(recovered) / max(np.count_nonzero(valid_pixels), 1)), 3
            ),
            "still_clipped_pixels": int(np.count_nonzero(still_clipped)),
            "outside_darker_frame_pixels": int(np.count_nonzero(dark_unmeasured)),
        }
        overlap["darker_support_percent"] = round(
            float(100 * np.count_nonzero(dark_supported) / max(int(np.count_nonzero(area)), 1)), 2
        )
        overlap["recovered_pixels"] = int(np.count_nonzero(recovered))
        overlap["unmeasured_after_blend_pixels"] = int(np.count_nonzero(common & ~valid_pixels))
        if np.count_nonzero(valid_pixels) < MIN_COMMON_PIXELS:
            raise InputError("Too little shared, unclipped detail remains after combining the two parallel exposures.")
    ties = calibration_area & parallel.valid & cross.valid & ~parallel.clipped & ~cross.clipped
    if darker is not None:
        # Primary-clipped tie pixels are usable only if the darker frame replaces them.
        ties &= recovered | ~parallel.clipped
        ties &= darker.valid & ~darker.clipped
    factor, method = glare_scale(direct, subdued, ties, parameters["balance"], parameters["fixed_gain"])
    fitted = subdued * factor
    remaining = np.maximum(direct - fitted, 0)
    percent = remaining / np.maximum(direct, 0.025)
    hot = valid_pixels & (percent >= parameters["glare_fraction"])
    entries, count = hotspot_summaries(hot, parallel, parameters["minimum_hotspot"])
    measured_count = int(np.count_nonzero(valid_pixels))
    density = int(np.count_nonzero(hot)) / measured_count
    below_zero = valid_pixels & (direct < fitted)
    result = {
        "pair": pairing,
        "support": overlap,
        "balance": method,
        "gain": round(factor, 5),
        "dark_exposure": dark_exposure,
        "signed_signal_below_zero_percent": round(100 * np.count_nonzero(below_zero) / measured_count, 2),
        "strong_glare_percent": round(100 * density, 3),
        "glare_quantiles": quantile_rows(percent[valid_pixels]),
        "residual_light_quantiles": quantile_rows(remaining[valid_pixels]),
        "hotspots": entries,
        "hotspot_count": count,
        "note": "Relative glare under this recorded capture pair, in linear image intensity.",
    }
    if dark_exposure is not None:
        result["note"] = (
            "Relative glare in linear image intensity. Bright spots clipped in the normal parallel capture are "
            "filled from the normalized darker parallel capture where it still holds detail; outlined regions show "
            "where the darker exposure was used. Pixels clipped in both captures or outside the darker frame stay unmeasured."
        )
    return result, percent, valid_pixels, common & ~valid_pixels, recovered


def directional_grid(signal, support, active, cells=5):
    height, width = support.shape
    ys = np.rint(np.linspace(0, height, cells + 1)).astype(int)
    xs = np.rint(np.linspace(0, width, cells + 1)).astype(int)
    items = []
    for row, (top, bottom) in enumerate(zip(ys, ys[1:])):
        for col, (left, right) in enumerate(zip(xs, xs[1:])):
            region = np.s_[top:bottom, left:right]
            valid = support[region]
            if np.count_nonzero(valid) < 20:
                continue
            observed = signal[region][valid]
            localized = active[region][valid]
            items.append({
                "cell": [col, row],
                "median_signed": round(float(np.median(observed)), 5),
                "strong_percent": round(100 * np.mean(localized), 2),
                "positive_percent": round(100 * np.mean(observed > 0), 2),
            })
    return items


def measure_direction(first, second, area, parameters):
    pairing = registration_pair(first, second)
    if first.photo["role"] != "rake-left" or second.photo["role"] != "rake-right":
        raise InputError("Direction comparison needs one left and one right raking-light photo.")
    common, fit_area, overlap = captured_pair(first, second, area)
    a = luminance(first.linear)
    b = luminance(second.linear)
    if parameters["balance"] == "auto":
        gain, method = robust_exposure_ratio(a, b, fit_area)
    else:
        gain, method = fixed_gain(parameters["fixed_gain"])
    b = b * gain
    ratio = (a - b) / np.maximum(a + b, 0.045)
    baseline, smooth_coverage = gaussian_supported(ratio, fit_area, parameters["scale_px"])
    residual = ratio - baseline if parameters["remove_broad"] else ratio
    support = fit_area & (smooth_coverage >= 0.68) if parameters["remove_broad"] else fit_area
    if np.count_nonzero(support) < MIN_COMMON_PIXELS:
        raise InputError("The common area is too fragmented for the selected shading scale.")
    values = residual[support]
    floor = robust_noise(values)
    limit = max(parameters["strength_floor"], 2.8 * floor)
    active = support & (np.abs(residual) >= limit)
    result = {
        "pair": pairing,
        "support": overlap,
        "balance": method,
        "gain": round(gain, 5),
        "local_threshold": round(limit, 5),
        "left_strong_percent": round(100 * np.mean(values >= limit), 2),
        "right_strong_percent": round(100 * np.mean(values <= -limit), 2),
        "signed_quantiles": quantile_rows(values),
        "absolute_quantiles": quantile_rows(np.abs(values)),
        "cells": directional_grid(residual, support, active),
        "note": "A registered signed contrast field. Highlights and shadows are relative to these two captures.",
    }
    return result, residual, support, common & ~support
