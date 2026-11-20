from collections import defaultdict
import math

from .art import comparison_sheet, load_artifact_picture, stored_preview
from .model import InputError, dictionary, enumeration, string, unique_ids


STUDY_CHOICES = {
    "color": {
        "median_lightness": ("Tile median L*", lambda result: result["metrics"]["median_lab"][0]),
        "variation_p90": ("Color variation p90 ΔE76", lambda result: result["metrics"]["delta_e_from_tile_median"]["90"]),
    },
    "coverage": {
        "glazed": ("Fully glazed % of measured area", lambda result: result["metrics"]["percent"]["glazed"]),
        "mixed": ("Mixed % of measured area", lambda result: result["metrics"]["percent"]["mixed"]),
    },
    "glare": {
        "glare": ("Strong glare %", lambda result: result["metrics"]["strong_glare_percent"]),
    },
    "texture": {
        "median_detail": ("Median roughness in image intensity", lambda result: result["metrics"]["grid_roughness"]["50"]),
    },
}


def comparable_signature(result):
    basis = result["basis"]
    metrics = result["metrics"]
    options = result["parameters"]["options"]
    mode = result["kind"]
    schema = {
        "photo_role": basis["photo_role"],
        "synthetic": basis["synthetic"],
        "neutral_target": basis["neutral_target"],
        "source_space": basis["source_space"],
    }
    if mode == "coverage":
        schema["ambiguity"] = options["ambiguity"]
        schema["off_axis_limit"] = options["max_residual_fraction"]
    if mode == "glare":
        schema["balance"] = options["balance"]
        schema["glare_fraction"] = options["glare_fraction"]
        schema["comparison_role"] = metrics["pair"]["comparison_role"]
    if mode == "texture":
        schema["radius"] = options["background_radius"]
        schema["edge_scale"] = options["edge_smooth"]
    return schema


def group_label(result, group_by):
    specimen = result["provenance"]["specimen"]
    chosen = specimen[group_by]
    return chosen.strip() or "Unlabeled"


def capture_note(differences):
    if not differences:
        return "Selected measurements use the same lighting labels and method settings."
    return "Groups mix " + ", ".join(differences) + ". Use comparisons as descriptive image observations."


def trial_entry(record, selector, group_by):
    source = record["provenance"]["specimen"]
    value = selector(record)
    if value is None or not math.isfinite(value):
        raise InputError("One selected run does not have a numeric value for this comparison.")
    return {
        "measurement_id": record["id"],
        "tile_id": source["id"],
        "tile_name": source["name"],
        "group": group_label(record, group_by),
        "body": source["body"],
        "glaze": source["glaze"],
        "version": source["version"],
        "value": round(float(value), 6),
        "capture": record["provenance"]["photo"],
        "overlay": record["pictures"]["overlay"],
        "support": record["support"]["available_percent"],
    }


def study_stats(entries):
    statistics = []
    grouped = defaultdict(list)
    for item in entries:
        grouped[item["group"]].append(item)
    for label, group in sorted(grouped.items(), key=lambda pair: pair[0].casefold()):
        points = [item["value"] for item in group]
        average = sum(points) / len(points)
        sum_sq = sum((point - average) ** 2 for point in points)
        stats = {
            "group": label,
            "n": len(points),
            "mean": round(average, 6),
            "minimum": round(min(points), 6),
            "maximum": round(max(points), 6),
            "spread": round(math.sqrt(sum_sq / (len(points) - 1)), 6) if len(points) > 1 else None,
        }
        statistics.append(stats)
    return statistics


def assess_sampling_groups(entries, group_by):
    grouped = defaultdict(list)
    group_frame = {}
    for sample in entries:
        series = sample["group"]
        grouped[series].append(sample["support"])
    for label, points in grouped.items():
        group_frame[label] = {
            "lower_support_percent": min(points),
            "upper_support_percent": max(points),
        }
    only_one = [label for label, points in grouped.items() if len(points) == 1]
    caveats = []
    if only_one:
        caveats.append("One measurement in " + ", ".join(only_one) + ".")
    if any(float(item["support"]) < 92 for item in entries):
        caveats.append("Image support differs from the requested region for some specimens.")
    if len(grouped) == 1:
        caveats.append(f"Only one '{group_by}' group appears in this selection.")
    return group_frame, caveats


def pairing_key(entry):
    """Same body and glaze marks the matched recipe across two firing runs."""
    return (entry["body"].strip().casefold(), entry["glaze"].strip().casefold())


def recipe_label(entry):
    body = entry["body"].strip() or "Unlabeled body"
    glaze = entry["glaze"].strip() or "Unlabeled glaze"
    return f"{body} · {glaze}"


def mean_of(items):
    return sum(item["value"] for item in items) / len(items)


# Two-sided 95% Student's t critical values for small paired samples.
T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def paired_summary(pairs):
    changes = [pair["change"] for pair in pairs]
    count = len(changes)
    mean_change = sum(changes) / count
    variance = sum((change - mean_change) ** 2 for change in changes) / (count - 1) if count > 1 else None
    standard_error = math.sqrt(variance / count) if variance is not None else None
    t_critical = T_CRITICAL_95.get(count, 1.96)
    interval = (
        [round(mean_change - t_critical * standard_error, 6),
         round(mean_change + t_critical * standard_error, 6)]
        if standard_error is not None else None
    )
    increases = sum(1 for change in changes if change > 0)
    decreases = sum(1 for change in changes if change < 0)
    return {
        "pair_count": count,
        "mean_change": round(mean_change, 6),
        "mean_absolute_change": round(sum(abs(change) for change in changes) / count, 6),
        "change_spread": round(math.sqrt(variance), 6) if variance is not None else None,
        "increases": increases,
        "decreases": decreases,
        "unchanged": count - increases - decreases,
        "confidence_interval": interval,
    }


def build_paired_comparison(entries, firing_groups):
    """Pair the same body+glaze recipe across exactly two firing runs.

    Each recipe contributes one equally weighted difference, so a run with
    many extra tiles cannot dominate runs with a few tiles. Recipes that were
    photographed in only one run are reported separately and never averaged in.
    """
    run_names = sorted(firing_groups, key=lambda label: label.casefold())
    baseline_name, comparison_name = run_names
    by_run = {name: defaultdict(list) for name in run_names}
    for entry in entries:
        by_run[entry["group"]][pairing_key(entry)].append(entry)

    pairs = []
    excluded = []
    for run_name in run_names:
        for key, members in by_run[run_name].items():
            other = by_run[comparison_name if run_name == baseline_name else baseline_name].get(key)
            label = recipe_label(members[0])
            if other is None:
                excluded.append({
                    "recipe": label,
                    "body": members[0]["body"].strip(),
                    "glaze": members[0]["glaze"].strip(),
                    "available_run": run_name,
                    "missing_run": comparison_name if run_name == baseline_name else baseline_name,
                    "n": len(members),
                    "measurement_ids": [item["measurement_id"] for item in members],
                    "tile_names": [item["tile_name"] for item in members],
                })

    for key in sorted(set(by_run[baseline_name]) & set(by_run[comparison_name]), key=lambda item: item[1]):
        first = sorted(by_run[baseline_name][key], key=lambda item: item["tile_name"].casefold())
        second = sorted(by_run[comparison_name][key], key=lambda item: item["tile_name"].casefold())
        baseline_mean = mean_of(first)
        comparison_mean = mean_of(second)
        pairs.append({
            "recipe": recipe_label(first[0]),
            "body": first[0]["body"].strip(),
            "glaze": first[0]["glaze"].strip(),
            "baseline_run": baseline_name,
            "comparison_run": comparison_name,
            "baseline_mean": round(baseline_mean, 6),
            "comparison_mean": round(comparison_mean, 6),
            "change": round(comparison_mean - baseline_mean, 6),
            "baseline_n": len(first),
            "comparison_n": len(second),
            "baseline_measurement_ids": [item["measurement_id"] for item in first],
            "comparison_measurement_ids": [item["measurement_id"] for item in second],
            "baseline_tile_names": [item["tile_name"] for item in first],
            "comparison_tile_names": [item["tile_name"] for item in second],
        })

    caveats = []
    if not pairs:
        caveats.append(
            "No recipe shares a body and glaze label across both firing runs, so no paired comparison was possible."
        )
    else:
        imbalanced = [pair for pair in pairs if pair["baseline_n"] != pair["comparison_n"]]
        if imbalanced:
            labels = ", ".join(pair["recipe"] for pair in imbalanced[:4])
            suffix = " and others" if len(imbalanced) > 4 else ""
            caveats.append(
                "Unequal tile counts in " + labels + suffix + "; each recipe still counts once via its group mean."
            )
        split = [pair for pair in pairs if pair["change"] > 0] and [pair for pair in pairs if pair["change"] < 0]
        if split:
            caveats.append("Recipes change in opposite directions; review the per-recipe table before generalizing.")
        if len(pairs) == 1:
            caveats.append("Only one matched recipe, so the paired change is a single observation, not a trend.")
        if excluded:
            caveats.append(
                f"{len(excluded)} recipe group(s) appear in only one firing run and are excluded from pairing."
            )
    return {
        "available": bool(pairs),
        "pairing_basis": "Same clay body and glaze label, paired across the two firing runs.",
        "baseline_run": baseline_name,
        "comparison_run": comparison_name,
        "summary": paired_summary(pairs) if pairs else None,
        "pairs": pairs,
        "excluded_groups": sorted(excluded, key=lambda item: item["recipe"].casefold()),
        "caveats": caveats,
    }


def sheets_for_study(blobs, title, entries, unit):
    digests = []
    for offset in range(0, len(entries), 9):
        group = entries[offset:offset + 9]
        cards = []
        for item in group:
            source = load_artifact_picture(blobs, item["overlay"])
            cards.append({
                "image": source,
                "heading": item["tile_name"],
                "description": f"{item['group']} · {item['value']:.3f} · {unit}",
            })
        label = title if len(entries) <= 9 else f"{title} · {offset + 1}–{offset + len(cards)}"
        digests.append(stored_preview(comparison_sheet(cards, label), blobs))
    return digests


def unavailable_paired(reason):
    return {
        "available": False,
        "pairing_basis": "Same clay body and glaze label, paired across the two firing runs.",
        "reason": reason,
        "baseline_run": None,
        "comparison_run": None,
        "summary": None,
        "pairs": [],
        "excluded_groups": [],
        "caveats": [],
    }


def create_study(store, payload):
    data = dictionary(payload, "Firing comparison")
    title = string(data.get("title"), "Study title", maximum=100)
    group_by = enumeration(data.get("group_by", "firing"), ("firing", "body", "glaze", "batch"), "Group by")
    identifiers = unique_ids(data.get("result_ids"), "Measurements", minimum=2, maximum=18)
    records = [store.result(result_id) for result_id in identifiers]
    kind = records[0]["kind"]
    if kind not in STUDY_CHOICES or any(record["kind"] != kind for record in records):
        raise InputError("Select 2 to 18 runs of the same supported measurement type.")
    if len({record["tile_id"] for record in records}) != len(records):
        raise InputError("Select one measurement per specimen for a firing comparison.")
    option = enumeration(data.get("metric"), STUDY_CHOICES[kind], "Metric")
    unit, reducer = STUDY_CHOICES[kind][option]
    signatures = [comparable_signature(record) for record in records]
    differences = sorted(key.replace("_", " ") for key in signatures[0] if len({item[key] for item in signatures}) > 1)
    entries = sorted(
        [trial_entry(record, reducer, group_by) for record in records],
        key=lambda item: (item["group"].casefold(), item["tile_name"].casefold(), item["measurement_id"]),
    )
    sampling, caveats = assess_sampling_groups(entries, group_by)
    firing_groups = sorted({item["group"] for item in entries}, key=lambda label: label.casefold())
    if group_by != "firing":
        paired = unavailable_paired("Paired comparison is offered when measurements are grouped by firing run.")
    elif len(firing_groups) != 2:
        paired = unavailable_paired(
            f"Paired comparison needs exactly two firing runs; this selection contains {len(firing_groups)}."
        )
    else:
        paired = build_paired_comparison(entries, firing_groups)
    return store.save_study({
        "title": title,
        "kind": kind,
        "metric": option,
        "metric_label": unit,
        "group_by": group_by,
        "rows": entries,
        "groups": study_stats(entries),
        "sampling": sampling,
        "paired": paired,
        "comparison_note": capture_note(differences),
        "caveats": caveats + paired["caveats"],
        "pictures": sheets_for_study(store.blobs, title, entries, unit),
    })


def public_metrics():
    return {kind: {key: descriptor[0] for key, descriptor in options.items()} for kind, options in STUDY_CHOICES.items()}
