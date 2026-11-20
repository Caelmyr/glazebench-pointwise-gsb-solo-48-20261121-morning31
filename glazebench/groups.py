"""Approved reference groups and group color-support measurement.

A group gathers several approved visual references. Measuring a specimen
against a group marks which of its colors are covered by the photographed
samples and which fall in the unsampled space between them. Coverage is the
union of the per-sample color neighborhoods; it is never interpolated across
colors the group did not actually record.
"""

import numpy as np

from .color import delta_e, dominant_swatches, linear_lab
from .model import InputError, group_fields


MIN_GAP_PALETTE_PIXELS = 40


def dedupe_members(references):
    """Collapse repeated records of the same physical specimen.

    Several approved references can point at one specimen. They describe the
    same fired tile, so the group counts that tile once: the earliest record
    stands as its sample and the rest are reported as collapsed duplicates.
    """
    kept = []
    collapsed = []
    seen = {}
    ordered = sorted(references, key=lambda item: (item["created_at"], item["id"]))
    for ref in ordered:
        specimen_id = ref["specimen"]["id"]
        if specimen_id in seen:
            collapsed.append({
                "id": ref["id"],
                "title": ref["title"],
                "specimen_name": ref["specimen"]["name"],
                "kept_under": seen[specimen_id]["title"],
            })
            continue
        seen[specimen_id] = ref
        kept.append(ref)
    return kept, collapsed


def resolve_group(store, group_id):
    group = store.group(group_id)
    fetched = [store.reference(ref_id) for ref_id in group["member_ids"]]
    active = [ref for ref in fetched if not ref["retired"]]
    retired = [ref for ref in fetched if ref["retired"]]
    members, collapsed = dedupe_members(active)
    if not members:
        raise InputError("Every member of this reference group has been retired.")
    return {
        "group": group,
        "members": members,
        "duplicates": collapsed,
        "retired": retired,
        "all_members": fetched,
    }


def create_group(store, payload):
    fields = group_fields(payload)
    for ref_id in fields["member_ids"]:
        ref = store.reference(ref_id)
        if ref["retired"]:
            raise InputError(f"Reference '{ref['title']}' has been retired and cannot join a new group.")
    return store.save_group(fields)


def present_group(store, group):
    members = []
    for ref_id in group["member_ids"]:
        ref = store.reference(ref_id)
        members.append({
            "id": ref["id"],
            "title": ref["title"],
            "retired": ref["retired"],
            "swatch": ref["fingerprint"]["swatch"],
            "specimen_name": ref["specimen"]["name"],
            "tile_id": ref["tile_id"],
        })
    active = [item for item in members if not item["retired"]]
    return {
        **group,
        "members": members,
        "active_members": len(active),
        "distinct_tiles": len({item["tile_id"] for item in active}),
    }


def basis_alignment(this_basis, member_basis):
    """Return None when both sides share a capture basis, else a note."""
    if this_basis["synthetic"] != member_basis["synthetic"]:
        return "This sample mixes synthetic and photographed material with the measurement."
    if this_basis["photo_role"] != member_basis["photo_role"]:
        return "The illumination labels differ. Interpret these color distances accordingly."
    if this_basis["neutral_target"] != member_basis["neutral_target"]:
        return "The pair uses different neutral-card targets or one side uses uncorrected color."
    return None


def member_summary(ref, radius, covered, measured, distance, support, note):
    fingerprint = ref["fingerprint"]
    return {
        "id": ref["id"],
        "title": ref["title"],
        "specimen_name": ref["specimen"]["name"],
        "tile_id": ref["specimen"]["id"],
        "swatch": fingerprint["swatch"],
        "median_lab": fingerprint["median_lab"],
        "dispersion": fingerprint["dispersion"],
        "radius": round(radius, 3),
        "supported_percent": round(100 * int(np.count_nonzero(covered)) / measured, 2),
        "nearest_delta_median": round(float(np.median(distance[support])), 3),
        "basis_note": note,
    }


def classify_palette(palette, centers, radii, member_rows):
    classified = []
    for item in palette:
        center = np.asarray(item["lab"], dtype=np.float32)
        margins = delta_e(centers, center) - radii
        nearest = int(np.argmin(margins))
        classified.append({
            **item,
            "status": "supported" if margins[nearest] <= 0 else "gap",
            "nearest_member": member_rows[nearest]["title"],
            "margin": round(float(margins[nearest]), 3),
        })
    return classified


def measure_group_support(linear, support, resolved, tolerance, basis):
    """Mark measured pixels covered by at least one photographed group sample.

    Each member contributes the neighborhood it actually recorded: its median
    color plus its own within-sample dispersion and the match tolerance. A
    pixel is supported only inside that union, so colors between samples —
    never photographed in an approved tile — remain unsampled gaps.
    """
    lab_image = linear_lab(linear)
    members = resolved["members"]
    measured = int(np.count_nonzero(support))
    states = np.full(support.shape, -1, dtype=np.int8)
    states[support] = 0
    member_rows = []
    alerts = []
    centers = []
    radii = []
    for ref in members:
        fingerprint = ref["fingerprint"]
        center = np.asarray(fingerprint["median_lab"], dtype=np.float32)
        radius = float(fingerprint["dispersion"]) + tolerance
        distance = delta_e(lab_image, center)
        covered = support & (distance <= radius)
        states[covered] = 1
        note = basis_alignment(basis, ref["basis"])
        if note:
            alerts.append({"member_id": ref["id"], "title": ref["title"], "note": note})
        member_rows.append(member_summary(ref, radius, covered, measured, distance, support, note))
        centers.append(center)
        radii.append(radius)
    centers = np.stack(centers)
    radii = np.asarray(radii, dtype=np.float32)
    palette = classify_palette(dominant_swatches(lab_image, support), centers, radii, member_rows)
    gap_mask = support & (states == 0)
    gap_pixels = int(np.count_nonzero(gap_mask))
    gap_palette = dominant_swatches(lab_image, gap_mask) if gap_pixels >= MIN_GAP_PALETTE_PIXELS else []
    supported_pixels = int(np.count_nonzero(states == 1))
    metrics = {
        "group_id": resolved["group"]["id"],
        "group_title": resolved["group"]["title"],
        "group_version": resolved["group"]["version"],
        "tolerance": tolerance,
        "member_count": len(members),
        "distinct_tiles": len(members),
        "duplicates_collapsed": resolved["duplicates"],
        "retired_excluded": [
            {"id": ref["id"], "title": ref["title"], "specimen_name": ref["specimen"]["name"]}
            for ref in resolved["retired"]
        ],
        "members": member_rows,
        "measured_pixels": measured,
        "supported_percent": round(100 * supported_pixels / measured, 2),
        "gap_percent": round(100 * gap_pixels / measured, 2),
        "palette": palette,
        "gap_palette": gap_palette,
        "basis_status": "aligned" if not alerts else "attention",
        "basis_alerts": alerts,
    }
    return metrics, states
