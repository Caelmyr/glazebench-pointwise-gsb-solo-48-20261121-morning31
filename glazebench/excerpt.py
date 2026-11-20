"""Partial evidence packets: crop pinned measurements to one rectangle.

A normal evidence packet keeps the whole registered tile plane. A partial
packet pins the same measurements but exports only the pixels inside a
rectangle drawn in registered-plane coordinates. The rectangle, the plane
size at pin time, unsupported (never photographed) gaps, and annotations cut
by the rectangle travel with the packet so a recipient can locate the excerpt
on the original tile and never mistakes it for a new, complete measurement.
"""

import numpy as np
from PIL import ImageDraw

from .art import font, load_artifact_picture, stored_preview
from .model import InputError, dictionary, scalar


MIN_CROP_SPAN = 0.02
MIN_CROP_PIXELS = 16
MAX_PLANE_EDGE = 1800
BORDER_COLOR = (246, 139, 96, 255)
RIBBON_INK = (255, 245, 225, 255)
RIBBON_BAND = (30, 38, 43, 232)
RIBBON_TEXT = "EVIDENCE EXCERPT"
EDGE_NAMES = ("top", "right", "bottom", "left")
MAX_LISTED_MARKERS = 80


def crop_rectangle(payload):
    """Validate the pin request's optional crop rectangle.

    Coordinates are normalized against the registered tile plane:
    ``left, top, right, bottom`` each in the range 0..1 with a minimum span.
    Accepts either a four-item list or an object with those four keys.
    Returns ``None`` when the request is for an ordinary full packet.
    """
    data = dictionary(payload, "Evidence packet")
    raw = data.get("crop")
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        values_in = list(raw)
    elif isinstance(raw, dict):
        values_in = [raw.get(label) for label in ("left", "top", "right", "bottom")]
    else:
        raise InputError("The excerpt rectangle must be a list of four coordinates or an object with left, top, right, and bottom.")
    values = []
    for label, number in zip(("left", "top", "right", "bottom"), values_in):
        values.append(round(scalar(number, f"Excerpt {label}", lower=0, upper=1), 6))
    left, top, right, bottom = values
    if right <= left or bottom <= top:
        raise InputError("The excerpt rectangle must have a positive width and height.")
    if right - left < MIN_CROP_SPAN or bottom - top < MIN_CROP_SPAN:
        raise InputError(f"The excerpt rectangle must cover at least {MIN_CROP_SPAN:.0%} of each plane edge.")
    return [left, top, right, bottom]


def plane_extent(record):
    extent = record.get("support", {}).get("registered_image_extent")
    if not isinstance(extent, list) or len(extent) != 2:
        raise InputError("A pinned measurement is missing its registered plane extent.")
    width, height = extent
    if not isinstance(width, int) or not isinstance(height, int):
        raise InputError("The registered plane extent must use integer pixels.")
    if not MIN_CROP_PIXELS <= width <= MAX_PLANE_EDGE or not MIN_CROP_PIXELS <= height <= MAX_PLANE_EDGE:
        raise InputError("The registered plane extent is outside the supported tile plane size.")
    return width, height


def pixel_box(extent, rect):
    """Convert normalized rectangle to integer pixel box ``(x0, y0, x1, y1)``."""
    width, height = extent
    left, top, right, bottom = rect
    x0 = int(round(left * width))
    y0 = int(round(top * height))
    x1 = int(round(right * width))
    y1 = int(round(bottom * height))
    x0, x1 = sorted((min(max(x0, 0), width - 1), min(max(x1, 1), width)))
    y0, y1 = sorted((min(max(y0, 0), height - 1), min(max(y1, 1), height)))
    if x1 - x0 < MIN_CROP_PIXELS or y1 - y0 < MIN_CROP_PIXELS:
        raise InputError("The excerpt rectangle covers too few pixels to remain readable.")
    return [x0, y0, x1, y1]


def gap_report(image, rect):
    """Summarize unsupported pixels inside the cropped picture.

    Transparent pixels are positions the capture never photographed. Edge
    fractions show where the blank area reaches the excerpt boundary, which
    keeps a gap from being read as a complete, fully supported measurement.
    """
    alpha = np.asarray(image.convert("RGBA").getchannel("A"))
    unsupported = alpha < 254
    gap = round(float(unsupported.mean()) * 100, 2)
    edges = {
        "top": round(float(unsupported[0, :].mean()) * 100, 2),
        "bottom": round(float(unsupported[-1, :].mean()) * 100, 2),
        "left": round(float(unsupported[:, 0].mean()) * 100, 2),
        "right": round(float(unsupported[:, -1].mean()) * 100, 2),
    }
    plane_edges = {
        "top": rect[1] <= 0,
        "left": rect[0] <= 0,
        "right": rect[2] >= 1,
        "bottom": rect[3] >= 1,
    }
    touched = [name for name in EDGE_NAMES if edges[name] > 0]
    return {"gap_percent": gap, "edge_gap_percent": edges, "plane_boundary": plane_edges, "gap_edges": touched}


def mark_excerpt(image):
    """Frame the crop so it can never pass for a full-plane picture."""
    image = image.convert("RGBA").copy()
    width, height = image.size
    edge = max(3, round(min(width, height) / 200))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline=BORDER_COLOR, width=edge)
    typeface = font(max(11, round(min(width, height) / 26)))
    text_width = draw.textlength(RIBBON_TEXT, font=typeface)
    pad = max(5, round(min(width, height) / 90))
    band_height = typeface.size + pad * 2
    band_width = int(text_width) + pad * 3
    band_width = min(band_width, width - edge * 2)
    draw.rectangle((edge, edge, edge + band_width, edge + band_height), fill=RIBBON_BAND)
    draw.text((edge + pad * 1.5, edge + pad), RIBBON_TEXT, font=typeface, fill=RIBBON_INK)
    return image


def _intersects(box, rect):
    x0, y0, x1, y1 = box
    left, top, right, bottom = rect
    return x0 < right and x1 > left and y0 < bottom and y1 > top


def _contains(box, rect):
    x0, y0, x1, y1 = box
    left, top, right, bottom = rect
    return x0 >= left and y0 >= top and x1 <= right and y1 <= bottom


def truncated_annotations(record, rect):
    """List discrete markings that the rectangle cuts or leaves outside."""
    kind = record["kind"]
    truncated = []
    outside = 0
    profile_cut = False
    if kind == "defects":
        for observation in record["metrics"].get("observations", []):
            bounds = observation.get("bounds")
            if not isinstance(bounds, list) or len(bounds) != 4:
                continue
            if not _intersects(bounds, rect):
                outside += 1
            elif not _contains(bounds, rect):
                truncated.append(observation["key"])
    elif kind == "profile":
        endpoints = record["metrics"].get("endpoints", [])
        for point in endpoints:
            if not isinstance(point, list) or len(point) != 2:
                continue
            x, y = point
            if not (rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]):
                profile_cut = True
    return {
        "markers_truncated": truncated[:MAX_LISTED_MARKERS],
        "markers_truncated_count": len(truncated),
        "markers_outside_count": outside,
        "profile_line_cut": profile_cut,
    }


def build_excerpt(store, records, rect):
    """Build cropped, marked replacement records plus the packet descriptor."""
    descriptor = None
    variants = []
    for record in records:
        extent = plane_extent(record)
        if descriptor is not None and list(extent) != descriptor["plane_size"]:
            raise InputError("Pinned measurements were rendered on different registered plane sizes; pin them separately.")
        box = pixel_box(extent, rect)
        pictures = {}
        picture_gaps = {}
        for view, digest in record["pictures"].items():
            if not digest:
                pictures[view] = None
                continue
            cropped = load_artifact_picture(store.blobs, digest).crop(tuple(box))
            picture_gaps[view] = gap_report(cropped, rect)
            pictures[view] = stored_preview(mark_excerpt(cropped), store.blobs)
        summary = {
            "rect": rect,
            "plane_size": list(extent),
            "pixel_box": box,
            "gaps": picture_gaps,
            "truncated_annotations": truncated_annotations(record, rect),
        }
        variants.append({**record, "pictures": pictures, "excerpt": summary})
        if descriptor is None:
            descriptor = {
                "limited": True,
                "rect": rect,
                "plane_size": list(extent),
                "pixel_box": box,
            }
    return descriptor, variants


def imported_excerpt(manifest, measurements):
    """Validate the excerpt descriptor on an incoming evidence archive."""
    raw = manifest.get("excerpt")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise InputError("The partial evidence descriptor must be an object.")
    rect_raw = raw.get("rect")
    if not isinstance(rect_raw, list) or len(rect_raw) != 4:
        raise InputError("The partial evidence descriptor needs four rectangle coordinates.")
    rect = [scalar(value, "Excerpt coordinate", lower=0, upper=1) for value in rect_raw]
    left, top, right, bottom = rect
    if right <= left or bottom <= top or right - left < MIN_CROP_SPAN or bottom - top < MIN_CROP_SPAN:
        raise InputError("The partial evidence rectangle is invalid.")
    plane_size = raw.get("plane_size")
    if (
        not isinstance(plane_size, list) or len(plane_size) != 2
        or not all(isinstance(value, int) and MIN_CROP_PIXELS <= value <= MAX_PLANE_EDGE for value in plane_size)
    ):
        raise InputError("The partial evidence descriptor has an invalid plane size.")
    box = raw.get("pixel_box")
    if (
        not isinstance(box, list) or len(box) != 4
        or not all(isinstance(value, int) for value in box)
    ):
        raise InputError("The partial evidence descriptor has an invalid pixel box.")
    expected = pixel_box(plane_size, rect)
    if box != expected:
        raise InputError("The partial evidence pixel box does not match its rectangle and plane size.")
    if raw.get("limited") is not True:
        raise InputError("A partial evidence descriptor must be marked as limited.")
    for record in measurements:
        summary = record.get("excerpt")
        if summary is None:
            raise InputError("A partial evidence packet needs an excerpt summary on every measurement.")
        if record.get("capture", {}).get("plane_to_capture") is None:
            raise InputError("A partial evidence packet needs the plane-to-capture geometry to re-locate its rectangle.")
        _validate_summary(summary, rect, plane_size)
    return {
        "limited": True,
        "rect": rect,
        "plane_size": plane_size,
        "pixel_box": box,
    }


def _validate_summary(summary, rect, plane_size):
    if not isinstance(summary, dict):
        raise InputError("A measurement excerpt summary must be an object.")
    if summary.get("rect") != rect:
        raise InputError("A measurement excerpt rectangle does not match the packet rectangle.")
    if summary.get("plane_size") != plane_size:
        raise InputError("A measurement excerpt plane size does not match the packet plane.")
    box = summary.get("pixel_box")
    if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, int) for value in box):
        raise InputError("A measurement excerpt pixel box is invalid.")
    if box != pixel_box(plane_size, rect):
        raise InputError("A measurement excerpt pixel box does not match the packet rectangle.")
    gaps = summary.get("gaps")
    if not isinstance(gaps, dict):
        raise InputError("A measurement excerpt is missing its support gap report.")
    for view, report in gaps.items():
        if view not in ("source", "overlay", "paired") or not isinstance(report, dict):
            raise InputError("An excerpt gap report is attached to an unknown image.")
        value = report.get("gap_percent")
        if not isinstance(value, (int, float)) or not 0 <= value <= 100:
            raise InputError("An excerpt gap percentage is outside the valid range.")
        edges = report.get("edge_gap_percent")
        if not isinstance(edges, dict) or any(
            not isinstance(edges.get(name), (int, float)) or not 0 <= edges[name] <= 100
            for name in EDGE_NAMES
        ):
            raise InputError("An excerpt edge gap report is invalid.")
        boundary = report.get("plane_boundary")
        if not isinstance(boundary, dict) or any(not isinstance(boundary.get(name), bool) for name in EDGE_NAMES):
            raise InputError("An excerpt plane boundary report is invalid.")
        if not isinstance(report.get("gap_edges"), list) or any(item not in EDGE_NAMES for item in report["gap_edges"]):
            raise InputError("An excerpt gap edge list is invalid.")
    markers = summary.get("truncated_annotations")
    if not isinstance(markers, dict):
        raise InputError("An excerpt truncation report must be an object.")
    listed = markers.get("markers_truncated")
    if not isinstance(listed, list) or len(listed) > MAX_LISTED_MARKERS or any(not isinstance(item, str) for item in listed):
        raise InputError("An excerpt lists invalid truncated markings.")
    for name in ("markers_truncated_count", "markers_outside_count"):
        if not isinstance(markers.get(name), int) or markers[name] < 0:
            raise InputError("An excerpt truncation count is invalid.")
    if not isinstance(markers.get("profile_line_cut"), bool):
        raise InputError("An excerpt line-scan flag is invalid.")
