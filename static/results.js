import {badge, btn, e, img, link, n, percent, plot, stat, table, date, dots} from "./widgets.js";

const colors = ["#58836b","#b26a4a","#818d40","#786f99"];
const layers = {overlay:"Measurement map",pattern:"Local pattern",source:"Source plane",paired:"Paired plane",reference:"Reference image"};

export function metricSet(record) {
  const m = record.metrics;
  const rows = {
    color: () => [["Median L*",n(m.median_lab[0])],["Within-tile p90 ΔE",n(m.delta_e_from_tile_median["90"])],["Pixels measured",n(m.pixels,0)]],
    coverage: () => [["Glazed",percent(m.percent.glazed)],["Bare",percent(m.percent.bare)],["Mixed",percent(m.percent.mixed)],["Unmeasured",percent(m.percent.unmeasured)]],
    defects: () => {
      const reviewed = Object.values(record.decisions);
      return [["Observed",m.total_observations],["Confirmed",reviewed.filter(item=>item.decision==="confirm").length],["Dismissed",reviewed.filter(item=>item.decision==="dismiss").length]];
    },
    glare: () => [["Strong glare",percent(m.strong_glare_percent)],["Counted hotspots",m.hotspot_count],["Paired support",percent(m.support.overlap_percent)]],
    direction: () => [["Left-bright detail",percent(m.left_strong_percent)],["Right-bright detail",percent(m.right_strong_percent)],["Paired support",percent(m.support.overlap_percent)]],
    texture: () => [["Medium local roughness",n(m.grid_roughness["50"],5)],["Ridge axis",m.texture_orientation ? n(m.texture_orientation.ridge_angle_clockwise)+"°" : "—"],["Coherence",n(m.texture_orientation?.coherence,3)]],
    craze: () => [["Trace length",`${n(m.network_length)} ${m.length_unit}`],["Junctions",n(m.junctions,0)],["Closed cells",n(m.closed_cell_count,0)]],
    rim: () => [["Rim median ΔE",n(m.all_edges.median_from_center_delta_e)],["Rim pixels",n(m.all_edges.pixels,0)],["Interior L*",n(m.interior.median_lab[0])]],
    profile: () => [["Distance",`${n(m.line_length)} ${m.distance_unit}`],["Valid stations",`${m.supported_stations}/${m.requested_stations}`],["Median L*",n(m.features.median_lightness)]],
    reference: () => [["Specimen to reference ΔE",n(m.delta_e76)],["Lightness difference",n(m.delta_lab[0])],["Dispersion difference",n(m.dispersion_delta)]],
  }[record.kind]();
  return `<div class="metrics">${rows.map(([name,value])=>stat(name,value)).join("")}</div>`;
}

function legend(record) {
  if (!record.legend) return "";
  if (record.kind==="texture" && record.layer==="pattern") return patternLegend(record);
  const scale=record.legend;
  if (scale.mode==="categorical") {
    return `<div class="legend"><span>Image classification</span><div class="swatches">${scale.labels.map(label=>dots(label.name,label.color)).join("")}</div></div>`;
  }
  return `<div class="legend"><span>${e(scale.label)}</span><span>${n(scale.low,4)} to ${n(scale.high,4)}</span></div><div class="gradient ${e(scale.mode)}"></div>`;
}

function patternLegend(record) {
  const scale=record.legend;
  const m=record.metrics.local_pattern;
  const share=percent(100*m.summary.periodic_cells/Math.max(1,m.cells.length));
  const unit=(scale.pattern_label||"").replace(/^.*\((.*)\)$/,"$1");
  const range=scale.pattern_low==null ? "no confirmed spacing" : `${n(scale.pattern_low,3)}–${n(scale.pattern_high,3)} ${e(unit)}`;
  return `<div class="legend"><span>Confirmed ${e(share)} · ${m.summary.uncertain_cells} uncertain windows</span><span>${e(range)}</span></div>
    <div class="gradient"></div>
    <div class="legend"><div class="swatches">${dots("Confirmed repeat · line shows ridge direction","#ebcc72")}${dots("Uncertain · mixed, occluded or too few repeats","#b0b2a8")}</div></div>`;
}

function colorDetail(m) {
  return `<h3>Color palette</h3><div class="swatches">${m.palette.map(item=>dots(percent(item.coverage_percent), item.swatch)).join(" ")}</div>
    <details><summary>Spatial cells</summary>${table(["Column, row","Pixels","Median Lab","From median ΔE"], m.cells.map(item=>[item.cell.join(", "),item.pixels,item.lab.map(val=>n(val)).join(" / "),n(item.delta_from_center)]))}</details>`;
}

function coverageDetail(m) {
  const series = ["glazed","mixed","bare"].map((topic,index)=>({values:m.depth_bands.map(item=>item.percent[topic]),color:colors[index]}));
  return `<h3>Coverage by tile depth</h3>${plot(series,"Coverage from top to bottom",{min:0,max:100})}
  ${table(["Depth band","Glazed","Mixed","Bare","Unmeasured"],m.depth_bands.map(item=>[`${n(item.relative_top*100,0)}–${n(item.relative_bottom*100,0)}%`,...["glazed","mixed","bare","unmeasured"].map(key=>percent(item.percent[key]))]))}`;
}

function defectsDetail(record) {
  const markers=record.metrics.observations;
  if (!markers.length) return "<p>No spots exceeded this run's current thresholds.</p>";
  return `<h3>Review local observations</h3>
    <div class="divider-list scroll-list">${markers.map(marker=>{
      const decision=record.decisions[marker.key]?.decision || "review";
      return `<div class="row"><div class="grow"><strong>${e(marker.key)} · ${e(marker.type)}</strong><br><small>${n(marker.area_pixels,0)} px · ${e(marker.polarity)} · ${e(decision)}</small></div>
        ${btn("Confirm","defect-choice","tiny secondary",null,`data-marker="${e(marker.key)}" data-value="confirm"`)}
        ${btn("Dismiss","defect-choice","tiny ghost",null,`data-marker="${e(marker.key)}" data-value="dismiss"`)}
        ${btn("Note","defect-note","tiny ghost",null,`data-marker="${e(marker.key)}"`)}</div>`;
    }).join("")}</div>`;
}

function textureDetail(m) {
  const local=m.local_pattern;
  const periodic=local.cells.filter(cell=>cell.periodic);
  const unit=local.summary.spacing_unit || "px";
  const localTable=periodic.length ? table(
    ["Window (col, row)","Ridge axis","Repeat spacing","Cycles seen","First / second correlation"],
    periodic.map(cell=>[
      `${cell.column}, ${cell.row}`,
      n(cell.ridge_angle_clockwise)+"°",
      cell.spacing_mm!=null ? `${n(cell.spacing_mm,3)} mm` : `${n(cell.spacing_px,2)} px`,
      n(cell.visible_repeats,1),
      `${n(cell.first_correlation,3)} / ${n(cell.harmonic_correlation,3)}`,
    ]),
  ) : "<p class='plot-note'>No window showed several clean repeat cycles.</p>";
  const reasons={};
  local.cells.filter(cell=>!cell.periodic).forEach(cell=>{reasons[cell.reason]=(reasons[cell.reason]||0)+1;});
  return `<h3>Texture repeats by direction</h3>
    ${plot(m.lag_curves.map((dir,index)=>({values:dir.points.map(point=>point.correlation),color:colors[index]})),"Spacing along each direction",{min:-1,max:1})}
    <div class="swatches">${m.lag_curves.map((dir,index)=>dots(dir.direction,colors[index])).join(" ")}</div>
    ${table(["Direction","Peak spacing","Correlation"],m.lag_curves.flatMap(dir=>dir.local_highs.map(point=>[dir.direction,point.spacing_mm ? `${n(point.spacing_mm)} mm` : `${n(point.spacing_px)} px`,n(point.correlation,3)])))}
    <h3>Local pattern windows</h3>
    <div class="metrics">${stat(local.summary.periodic_cells,"Confirmed windows")}${stat(local.summary.uncertain_cells,"Uncertain windows")}
      ${local.summary.spacing_median!=null?stat(`${n(local.summary.spacing_median,3)} ${unit}`,"Median local spacing"):""}
      ${local.summary.angle_median!=null?stat(n(local.summary.angle_median)+"°","Median ridge axis"):""}</div>
    <p class="plot-note">${e(m.pattern_note)} Click a window on the Local pattern map to inspect it, then open the Source plane layer to re-check the original texture.</p>
    ${Object.keys(reasons).length?`<div class="swatches">${Object.entries(reasons).map(([reason,count])=>dots(`${count} · ${e(reason)}`,"#b0b2a8")).join(" ")}</div>`:""}
    <details><summary>Confirmed local windows (${periodic.length})</summary>${localTable}</details>
    <p class="plot-note">${e(m.note)}</p>`;
}

function profileDetail(m) {
  return `<h3>Lightness along the marked line</h3>${plot([{values:m.rows.map(item=>item.lightness)}], "Lightness along the station band",{axis:`0–${n(m.line_length)} ${m.distance_unit}`})}
    <details><summary>Sample station values</summary>${table(["Distance","Valid samples","L*","a*","b*"],m.rows.map(item=>[`${n(item.position,3)} ${item.position_unit}`,item.samples,n(item.lightness),n(item.a),n(item.b)]))}</details>`;
}

function glareDetail(m) {
  return `<h3>Glare hotspots</h3><p class="plot-note">${e(m.note)} Balancing gain ${n(m.gain,4)}.</p>
    ${table(["Hotspot","Pixels","Area mm²"],m.hotspots.map((item,index)=>[index+1,item.pixels,n(item.area_mm2,3)]))}`;
}

function directionDetail(m) {
  return `<h3>Raking contrast by cell</h3><p class="plot-note">${e(m.note)}</p>
    ${table(["Column, row","Signed median","Strong contrast"],m.cells.map(item=>[item.cell.join(", "),n(item.median_signed,4),percent(item.strong_percent)]))}`;
}

function crazeDetail(m) {
  return `<h3>Closed lines</h3><p class="plot-note">${e(m.note)}</p>
  ${table(["Cell","Pixels","Area mm²"],m.largest_cells.map((item,index)=>[index+1,item.pixels,n(item.mm2,3)]))}`;
}

function rimDetail(m) {
  return `<h3>Rim color by side</h3><p class="plot-note">${e(m.note)}</p>
  ${table(["Side","Pixels","Median from interior ΔE","Excluded"],m.sides.map(item=>[item.side,item.stats?.pixels,n(item.stats?.median_from_center_delta_e),percent(item.excluded_percent)]))}`;
}

function referenceDetail(m) {
  return `<h3>${e(m.reference_title)}</h3>
    <div class="metrics">${stat("Specimen",m.specimen.swatch)}${stat("Approved reference",m.reference.swatch)}</div>
    <p class="notice">${e(m.capture_note)}</p>`;
}

function details(record) {
  const render = {
    color:colorDetail,coverage:coverageDetail,defects:defectsDetail,glare:glareDetail,
    direction:directionDetail,texture:textureDetail,profile:profileDetail,reference:referenceDetail,craze:crazeDetail,rim:rimDetail,
  }[record.kind];
  return render(record.kind==="defects" ? record : record.metrics);
}

function selectedWindowSvg(record) {
  const cell=record.patternCell;
  if (record.kind!=="texture" || record.layer!=="source" || !cell) return "";
  const [l,t,r,b]=cell.normalized_bounds;
  return `<svg class="pattern-overlay" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-label="Selected texture window">
    <rect x="${(l*1000).toFixed(2)}" y="${(t*1000).toFixed(2)}" width="${((r-l)*1000).toFixed(2)}" height="${((b-t)*1000).toFixed(2)}"
      class="pattern-cell selected" fill="rgba(244,201,109,0.14)" stroke="#f4c96d" stroke-width="3" vector-effect="non-scaling-stroke"/></svg>`;
}

function patternOverlaySvg(record) {
  if (record.kind!=="texture" || record.layer!=="pattern") return "";
  const cells=record.metrics.local_pattern?.cells || [];
  if (!cells.length) return "";
  const selected=record.patternCell;
  const rects=cells.map(cell=>{
    const [l,t,r,b]=cell.normalized_bounds;
    const isSelected=selected && selected.row===cell.row && selected.column===cell.column;
    const label=cell.periodic
      ? `Window ${cell.column},${cell.row} · ridge ${n(cell.ridge_angle_clockwise)}° · ${cell.spacing_mm!=null?n(cell.spacing_mm,3)+" mm":n(cell.spacing_px,2)+" px"}`
      : `Window ${cell.column},${cell.row} · uncertain · ${cell.reason}`;
    return `<rect class="pattern-cell ${cell.periodic?"periodic":"uncertain"}${isSelected?" selected":""}" data-do="pattern-cell" data-row="${cell.row}" data-column="${cell.column}"
      x="${(l*1000).toFixed(2)}" y="${(t*1000).toFixed(2)}" width="${((r-l)*1000).toFixed(2)}" height="${((b-t)*1000).toFixed(2)}">
      <title>${label}</title></rect>`;
  }).join("");
  return `<svg class="pattern-overlay" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-label="Clickable local texture windows">${rects}</svg>`;
}

function patternInspector(record) {
  if (record.kind!=="texture" || !record.patternCell) return "";
  if (record.layer!=="pattern" && record.layer!=="source") return "";
  const cell=record.patternCell;
  const back=btn("Back to local pattern","result-layer","tiny secondary",null,'data-value="pattern"');
  const source=btn("Open source plane here","result-layer","tiny secondary",null,'data-value="source"');
  if (!cell.periodic) {
    return `<div class="notice warn"><strong>Window ${cell.column}, ${cell.row} · uncertain</strong><br>${e(cell.reason)} · ${n(cell.supported_pixels,0)} supported px
      <div class="actions">${record.layer==="pattern"?source:back}</div></div>`;
  }
  return `<div class="notice"><strong>Window ${cell.column}, ${cell.row}</strong> · ridge axis ${n(cell.ridge_angle_clockwise)}° · coherence ${n(cell.coherence,3)}
    · spacing ${cell.spacing_mm!=null?`${n(cell.spacing_mm,3)} mm (${n(cell.spacing_px,2)} px)`:`${n(cell.spacing_px,2)} px`}
    · ${n(cell.visible_repeats,1)} cycles visible · correlations ${n(cell.first_correlation,3)} / ${n(cell.harmonic_correlation,3)}
    <div class="actions">${record.layer==="pattern"?source:back}</div></div>`;
}

export function resultCard(record, layer, currentRevision) {
  const path=`/api/results/${record.id}`;
  const view={...record,layer};
  const available = ["overlay",...(record.pictures.pattern ? ["pattern"] : []),"source",...(record.pictures.paired ? ["paired"] : []),...(record.metrics.reference_image ? ["reference"] : [])];
  const old = record.source_version !== currentRevision;
  return `<section class="card stack">
    <div class="title-row"><div><h3>${e(record.label)}</h3><small>Specimen revision ${record.source_version} · ${e(date(record.created_at))}</small></div>${old ? badge("Earlier specimen revision","caution") : badge("Saved measurement","ready")}</div>
    <div class="actions">${available.map(key=>btn(layers[key],"result-layer",layer===key?"tiny primary":"tiny secondary",null,`data-value="${key}"`)).join("")}</div>
    <figure class="stage pattern-stage">${img(path+"/image/"+layer+"?v="+record.version,`${record.label} · ${layers[layer]}`)}${patternOverlaySvg(view)}${selectedWindowSvg(view)}</figure>
    ${patternInspector(view)}
    ${(layer==="overlay"||layer==="pattern")?legend(view):""}
    ${metricSet(record)}
    <section>${details(record)}</section>
    <details><summary>Measurement source</summary>${table(["Property","Value"],[
      ["Image",record.provenance.photo.display_name],["Image role",record.provenance.photo.role],["Region",record.provenance.region?.name || "Entire plane"],
      ["Photo digest",record.provenance.photo.original_digest],
      ["Supported area",percent(record.support.available_percent)],
      ["Clipped pixels",record.support.clipped_pixels],
      ["Neutral correction",record.provenance.neutral_card?.before_swatch || "Uncorrected"],
    ])}</details>
    <div class="actions">${link("Metrics CSV",path+"/metrics.csv","secondary","download")}${btn("Pin evidence","pin", "primary","archive")}</div>
  </section>`;
}

export function specimenResultRows(items) {
  if (!items.length) return "<p class='soft'>Run a measurement to start this specimen’s history.</p>";
  return `<div class="divider-list scroll-list">${items.map(item=>`<div class="row"><div class="grow"><strong>${e(item.label)}</strong><br><small>Revision ${item.source_version} · ${e(date(item.created_at))}</small></div>${btn("Open","result-open","tiny secondary",null,`data-id="${item.id}"`)}</div>`).join("")}</div>`;
}
