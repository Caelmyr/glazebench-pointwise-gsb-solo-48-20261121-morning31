# GlazeBench

GlazeBench is a local workbench for photographing and inspecting ceramic test tiles. Keep each specimen's differently lit images together, mark a common plane, inspect the surface, and retain the measurements with their source images.

## Start

```bash
docker compose up --build --wait
```

Open [GlazeBench](http://127.0.0.1:19951/). The standard Compose setup uses a local named volume for the SQLite database and image files. Use `docker compose down` to stop. The named volume retains the workspace when containers are recreated.

For an offline machine with the exact dependency set already on the local Docker engine, you can use an image that contains those Python 3.11 packages. On the authoring host `solo50-paintlayer:1.0` supplied the matching wheel set. The GlazeBench image copies only the resulting Python site packages from the dependency stage.

```bash
DOCKER_BUILDKIT=0 COMPOSE_BAKE=false GLAZEBENCH_DEPS_IMAGE=solo50-paintlayer:1.0 docker compose build
docker compose up --no-build --wait
```

For a native install, use a Python 3.11 virtual environment, install `requirements.txt`, then run `./start.sh`. `HOST`, `PORT`, `GLAZEBENCH_DATA`, and `GLAZEBENCH_PYTHON` can override the native defaults. The usual address remains loopback. Treat this as a trusted local studio application.

## Work with a tile

1. **Load training tiles** to open four procedurally rendered specimens. Each has diffuse, parallel-polarized, cross-polarized, and two opposing raking-light photos with known corners. These are synthetic and marked throughout the application. To work with your own photos, create a specimen and add real captures with their lighting labels.
2. **Captures** shows source images, clipping, visible support, and capture metadata. Mark corners clockwise, beginning at the upper-left. Mark the same physical corners in each capture. Use the registered checkerboard, split view, or support map to review the placement.
3. **Sampling regions** lets you draw an inspection polygon, two coverage samples, an exclusion, or a patch on a neutral card. Balance a diffuse base photo by selecting its neutral patch. The registered geometry and the original uploads remain available.
4. **Measure** offers spatial color variation, estimated application coverage, candidate spots, relative polarized glare, signed raking-light contrast, surface texture, crackle tracing, rim color, line profiles, and matching to one approved visual reference. Select a region or use the entire plane. The maps, per-mode tables, and CSV exports preserve which source and method produced each result.
5. **Geometry** measures paths and polygon areas in the registered plane. If you entered both specimen dimensions, lengths and areas use millimeters. Visible area excludes missing source support and marked exclusions. Saved measurements retain the frame on which they were made.
6. **Approve reference** makes an immutable snapshot of a specimen's visual color sample. Compare a subsequent measurement to that sample, or retire references that should no longer be offered for new comparisons.
7. **Firing studies** compares selected, saved measurements from different specimens. Pick one measurement type and one run per tile, then group by firing run, glaze, body, or batch. Save the contact sheet and table. When a study grouped by firing run contains exactly two runs, it also builds a paired comparison: recipes sharing the same body and glaze are matched across both runs, each recipe contributes one equally weighted mean difference (so a run with extra tiles cannot dominate), and recipes photographed in only one run are listed as excluded rather than averaged in. The paired table names every specimen and measurement used; click a specimen chip to inspect its source measurement. The per-run group averages remain available alongside the paired view, and the CSV export contains both tables.
8. **Evidence** pins a dated set of measurements and renders a portable inspection report. The ZIP includes maps, source-frame previews, a manifest, metric rows, and any originals explicitly selected when pinning. Use **Received evidence** to verify image digests and open a copy of an inspection packet from another studio. Packet verification checks internal agreement with its manifest.

## Inputs and image measurements

- PNG, JPEG, WebP, and single-frame TIFF up to 24 MiB and 12 megapixels. The source bytes are retained. Inline photo previews use EXIF orientation and embedded ICC conversion where applicable. Unsigned 16-bit grayscale is converted to an 8-bit display-referred working image. RGB input without an embedded profile is interpreted as sRGB.
- The registered plane is limited to 1,800 pixels along its longer side. Choose front-facing images with clearly identifiable tile corners. Near-singular corner geometry is rejected. A specimen can hold up to ten photographs and 24 live regions.
- Appearance values describe the captured image under its recorded lighting. Neutral-card balance is a simple channel adjustment. The color differences are ΔE76 in display-referred Lab. Glare, apparent roughness, contrast and crackle traces are comparative image measurements tied to the capture and scale.
- Coverage estimates the match to the marked bare body and fully glazed patches. It preserves ambiguous and unsupported areas as separate states. Candidate observations and texture traces can include real surface marks, lighting effects or acquisition artifacts. Use the source image to review them.
- A received ZIP accepts up to 24 MiB compressed, 72 MiB examined, and twelve declared measurements. The inspection reader renders verified image panels with local UI. A GlazeBench evidence ZIP can be read offline with its HTML report and relative measurement images.

## Source

The `glazebench` package holds the image pipeline, transactional data store, measurement and evidence services, demo renderer, and Flask HTTP entry point. The `static` directory contains the browser application. Demo captures are generated locally with deterministic seeds. The scripts directory contains the standalone source line counter.
