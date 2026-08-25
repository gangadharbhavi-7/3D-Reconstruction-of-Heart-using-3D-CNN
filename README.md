# 🫀 3D Cardiac Reconstruction System

A single-script pipeline that takes a cardiac CT or MRI scan (NIfTI format), segments the major heart structures, reconstructs them as photorealistic 3D meshes, runs a rules-based diagnostic summary, and packages everything into a hospital-style PDF report — all served through an in-browser dashboard launched directly from Google Colab and backed by Google Drive.

> ⚠️ **Disclaimer — read before using or sharing:** This is a research / educational prototype, not a certified medical device.
> - The "AI diagnosis" is a hand-written volume-threshold heuristic, **not** a trained or validated machine-learning model — despite labels like "CNN-Based" appearing in the UI/PDF text, no CNN inference actually runs.
> - The generated PDF mimics a real hospital report format (patient info, physician sign-off, letterhead) purely for demonstration. It should not be printed, shared, or treated as an actual clinical document.
> - Output must never be used to make real diagnostic or treatment decisions.

---

## Table of Contents

1. [Features](#-features)
2. [Architecture / Data Flow](#-architecture--data-flow)
3. [Repository & Google Drive Folder Structure](#-repository--google-drive-folder-structure)
4. [Requirements](#-requirements)
5. [Setup & Running in Google Colab](#-setup--running-in-google-colab)
6. [Running Outside Colab](#-running-outside-colab)
7. [Using the Dashboard](#-using-the-dashboard)
8. [Cardiac Structures & Label Mapping](#-cardiac-structures--label-mapping)
9. [Segmentation Logic](#-segmentation-logic)
10. [Mesh Reconstruction Pipeline](#-mesh-reconstruction-pipeline)
11. [AI Diagnosis Logic (Heuristic)](#-ai-diagnosis-logic-heuristic)
12. [PDF Report Contents](#-pdf-report-contents)
13. [API Reference](#-api-reference)
14. [Configuration / Customization](#-configuration--customization)
15. [Known Limitations](#-known-limitations)
16. [Troubleshooting](#-troubleshooting)
17. [License](#-license)

---

## ✨ Features

- **CT & MRI support** — normalizes both standard CT label maps and MRI label maps to a common 8-class scheme (background + 7 cardiac structures) via `CT_LABEL_MAPPING` / `MR_LABEL_MAPPING`.
- **Automatic multiclass fallback** — if the uploaded volume only contains a binary foreground/background mask, `simulate_multiclass()` estimates ventricles, atria, and myocardium using distance-transform and erosion heuristics, so the pipeline still produces a labeled reconstruction.
- **3D mesh generation** — per-structure Gaussian smoothing + Marching Cubes (`skimage.measure.marching_cubes`) to extract surfaces, followed by `trimesh`-based cleanup: vertex merging, hole filling, quadric decimation, and normal recalculation.
- **Procedural texturing** — a simple multi-frequency sine-noise function (`generate_procedural_texture`) assigns per-vertex texture data for a more organic, less "flat" appearance in the viewer.
- **Photorealistic Three.js viewer** — `THREE.MeshPhysicalMaterial` with roughness/metalness/clearcoat/transmission tuned per structure, multiple light sources (key/fill/rim/hemisphere), shadow mapping, ACES tone mapping, mouse-drag orbit + scroll zoom, and auto-rotation.
- **Rules-based cardiac diagnosis** (`ai_cardiac_diagnosis`) — flags patterns such as severe/moderate LV dilatation or hypertrophic cardiomyopathy based on chamber volumes, and generates findings, patient-facing recommendations, and physician treatment suggestions.
- **Hospital-style PDF report** (`generate_hospital_report`, via `reportlab` + `matplotlib`) — patient info, exam details, AI analysis summary, quantitative measurement table, three rendered anatomical views (anterior/lateral/superior), a page per cardiac structure with its own rendered image and clinical blurb, findings, recommendations, technical/export details, disclaimer, and an "authorized by" signature block.
- **Per-structure OBJ/MTL export** — download the full heart or any individual chamber/vessel as a separately colored, textured Wavefront OBJ file.
- **Session history** — in-memory `PATIENT_HISTORY` / `ANALYSIS_SESSIONS` dictionaries power a dashboard history table so you can revisit past reconstructions in the same runtime.
- **Self-contained Flask backend + single-page frontend** — no separate frontend build step; the entire dashboard (HTML/CSS/JS + Three.js) is returned as one big inline string from the `/` route.
- **Colab port proxying** — automatically detects a free local port (5000–5100) and, when running in Colab, uses `google.colab.output.eval_js` to generate a clickable proxied URL; falls back to `http://localhost:<port>` otherwise.

---

## 🏗️ Architecture / Data Flow

```
                    ┌─────────────────────────────┐
                    │   Dashboard (browser, SPA)   │
                    │  Patient form + file upload  │
                    └───────────────┬─────────────┘
                                    │ POST /process (multipart)
                                    ▼
                    ┌─────────────────────────────┐
                    │        Flask backend         │
                    │        process_scan()        │
                    └───────────────┬─────────────┘
                                    ▼
                    nib.load()  →  scan_data (NumPy array) + spacing
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │ len(unique values) > 2 ?                           │
        │   YES → normalize_labels(scan_data, modality)      │
        │   NO  → simulate_multiclass(binary_mask)            │
        └───────────────────────────┬───────────────────────┘
                                    ▼
        For each of the 7 structure labels (1–7):
          mask = (seg == label)
          Gaussian smoothing → marching_cubes → trimesh.Trimesh
          enhance_mesh_realism() (merge, fill holes, decimate, fix normals)
          generate_procedural_texture()
          compute_metrics() → volume in mL / mm³
                                    ▼
                    meshes = { label_id: {...mesh + metrics} }
                                    ▼
        ┌────────────────────┬───────────────────────────────┐
        ▼                    ▼                                ▼
  ai_cardiac_diagnosis()   export OBJ/MTL              generate_hospital_report()
  (volume-threshold        (full heart +                (matplotlib renders +
   heuristic)               per-structure)                reportlab PDF)
        │                    │                                │
        ▼                    ▼                                ▼
   diagnosis dict      /output/*.obj, *.mtl            /reports/Report_*.pdf
        │                    │                                │
        └────────────────────┴────────────┬───────────────────┘
                                           ▼
                          result dict → ANALYSIS_SESSIONS[session_id]
                                        PATIENT_HISTORY.insert(0, result)
                                           ▼
                              JSON response → browser
                                           ▼
              Three.js renders meshes; diagnosis + download buttons populate
```

---

## 📁 Repository & Google Drive Folder Structure

**Google Drive** (mounted at `/content/drive` in Colab) is where all data lives — the git repo should only ever contain the script/notebook and docs.

```
My Drive/
└── DATASET/
    └── Dataset/                       <- BASE_DIR in the script
        ├── ct_train/                  # CT training volumes + label maps (your dataset)
        ├── mr_train/                  # MRI training volumes + label maps (your dataset)
        ├── mr_test/                   # MRI test/held-out volumes (your dataset)
        ├── uploads/                   # scans uploaded via the dashboard  (auto-created)
        ├── output/                    # generated OBJ/MTL meshes         (auto-created)
        ├── reports/                   # generated PDF reports            (auto-created)
        ├── output_ultra_promax_plus/  # optional extra output artifacts
        └── renders/                   # optional extra rendered images
```

`uploads/`, `output/`, and `reports/` are created automatically via `os.makedirs(..., exist_ok=True)` the first time the script runs — you only need to seed `ct_train/`, `mr_train/`, and `mr_test/` yourself.

**Git repository** (what you actually push to GitHub):

```
your-repo/
├── heart_reconstruction.py   (or .ipynb)   # the full script in this README
├── README.md
├── .gitignore
└── LICENSE                                  (optional but recommended)
```

None of the Drive folders above should be tracked in git — see the [.gitignore guidance](#-known-limitations) from the earlier answer, or ask me to regenerate it.

---

## 🧩 Requirements

Installed automatically at runtime by the script itself (`pip install -q ...`), but for reference if you want to pre-build an environment:

```
nibabel
scipy
scikit-image
trimesh
flask
flask-cors
reportlab
matplotlib
Pillow
numpy
```

- Python 3.9+ recommended.
- A modern browser with WebGL support (for the Three.js viewer).
- Google account with Drive access (if running in Colab with Drive mounting).

---

## 🚀 Setup & Running in Google Colab

1. **Prepare your dataset in Drive.** Make sure `My Drive/DATASET/Dataset/ct_train`, `mr_train`, and `mr_test` already contain your NIfTI volumes (`.nii` or `.nii.gz`). If your dataset lives somewhere else, edit the `BASE_DIR` line near the top of the script.
2. **Open the script/notebook in Colab** and run the cell.
3. **Dependency install.** The script pip-installs all required packages quietly on first run — this can take 30–90 seconds.
4. **Drive mount.** You'll be prompted to authorize Google Drive access; approve it so `BASE_DIR` resolves correctly.
5. **Server start.** The script finds a free port (5000–5100), starts a Flask server in a background thread, waits ~3 seconds, and prints:
   ```
   ✨ 3D RECONSTRUCTION SYSTEM READY
   🌐 URL: https://<colab-proxy-url>
   ```
   It also renders a styled "Launch System" button/card inline in the notebook output.
6. **Open the dashboard.** Click the button or the printed URL — it opens the SPA in a new tab, proxied through Colab.
7. **Keep the Colab cell running.** The server runs in a daemon thread with a `while server.is_alive(): time.sleep(1)` loop — do not stop/interrupt the cell while you want the dashboard to stay usable.

---

## 💻 Running Outside Colab

The script auto-detects whether `google.colab` is importable. If not, `IN_COLAB = False` and it uses local temp directories instead of Drive:

```
/tmp/cardiovis_uploads
/tmp/cardiovis_output
/tmp/cardiovis_reports
```

Just run the script directly:

```bash
python heart_reconstruction.py
```

It will print `http://localhost:<port>` — open that in your browser. (Drive mounting, Colab proxying, and the `display(HTML(...))` launch card are skipped automatically.)

---

## 🖱️ Using the Dashboard

1. Go to **Patient Details** in the left nav.
2. Fill in **Patient ID**, **Name**, **Age**, **Gender**, **Modality** (CT or MRI), and **Medical Reason**.
3. Upload a `.nii` or `.nii.gz` scan file.
4. Click **⚡ Run Analysis**. You'll be switched to the **Analysis** tab showing a live log (upload → segmentation → mesh generation).
5. Once complete, you're taken to **3D Reconstruction**:
   - Drag to orbit, scroll to zoom.
   - Use the **Part Selection** panel to isolate a single structure or show all again.
   - View the **AI Diagnosis** card (condition, confidence, risk level, ejection fraction, findings).
   - Download the **Full Heart OBJ**, the **PDF Report**, or any individual structure's OBJ.
6. Check **Dashboard** or the **Patient Details → Patient Analysis History** table to revisit past sessions (only for the current Colab runtime — history is in-memory, not persisted).

---

## 🫀 Cardiac Structures & Label Mapping

| Label ID | Structure | Hex Color | Notes |
|---|---|---|---|
| 0 | Background | `#000000` | Ignored during reconstruction |
| 1 | Left Ventricle | `#8B0000` | Main pumping chamber |
| 2 | Right Ventricle | `#A52A2A` | Pumps blood to lungs |
| 3 | Left Atrium | `#DC143C` | Receives oxygenated blood |
| 4 | Right Atrium | `#CD5C5C` | Receives deoxygenated blood |
| 5 | Myocardium | `#800020` | Muscular wall / septum |
| 6 | Aorta | `#B22222` | Main outflow artery |
| 7 | Pulmonary Artery | `#8B4513` | Carries blood to lungs |

**Raw label normalization:**
- `CT_LABEL_MAPPING`: maps common CT segmentation label values (e.g. `205`, `420`, `500`, `550`, `600`, `820`, `850`) to the standardized 0–7 scheme above.
- `MR_LABEL_MAPPING`: accepts both already-standardized labels (`0`–`7`) and the same CT-style raw values, so MRI segmentations exported either way are handled.

---

## 🔬 Segmentation Logic

- If the uploaded volume already has more than 2 unique intensity values (i.e., it's a pre-segmented multi-label mask), `normalize_labels()` remaps it directly using the appropriate CT/MR mapping table.
- If the volume is effectively binary (foreground vs. background only — e.g. a simple heart mask with no chamber-level labels), `simulate_multiclass()` runs instead:
  1. Fill holes in the binary mask.
  2. Compute a Euclidean distance transform.
  3. Threshold at the 70th percentile of nonzero distances to approximate "ventricle-like" thick, central regions.
  4. Connected-component label each candidate region; discard components smaller than 1000 voxels.
  5. Classify each remaining component as LV/RV/LA/RA based on its centroid position (left vs. right half, upper vs. lower half of the volume).
  6. Estimate the myocardium as the shell between the filled mask and an eroded version of it (3 iterations of binary erosion).

This is a heuristic approximation, not a trained segmentation model — real anatomical accuracy depends entirely on how close your input mask's geometry is to the assumptions above.

---

## 🧱 Mesh Reconstruction Pipeline

For each of the 7 non-background structures with ≥100 voxels:

1. **Smoothing** — `gaussian_filter(mask, sigma=0.5)` softens jagged voxel boundaries before surface extraction.
2. **Surface extraction** — `skimage.measure.marching_cubes(smoothed_mask, level=0.3, spacing=spacing, step_size=2)` produces vertices, faces, and normals in physical (mm) units using the scan's actual voxel spacing.
3. **Mesh cleanup** (`enhance_mesh_realism`):
   - `merge_vertices()` to weld duplicate points.
   - `fill_holes()` if the mesh isn't watertight.
   - **Quadric decimation** to cap face count (targets: 10k faces if >15k, 6k if >8k, otherwise 80% of original, floored at 3k) — keeps the browser viewer performant.
   - `fix_normals()` for consistent shading.
4. **Texturing** — `generate_procedural_texture()` computes a 3-channel per-vertex value from layered sine functions (only for meshes under 20,000 vertices, to control payload size).
5. **Metrics** — `compute_metrics()` converts voxel count × voxel volume (from spacing) into mL and mm³.

---

## 🩺 AI Diagnosis Logic (Heuristic)

`ai_cardiac_diagnosis()` is a deterministic, rule-based function — **not** a trained classifier — that inspects the reconstructed chamber volumes:

| Condition | Trigger | Ejection Fraction | Risk |
|---|---|---|---|
| Severe LV Dilatation | LV volume > 225 mL | 35% | High |
| Moderate LV Dilatation | LV volume 195–225 mL | 45% | Moderate |
| Hypertrophic Cardiomyopathy | Myocardial volume > 252 mL | 68% | Moderate |
| Normal Cardiac Anatomy | none of the above | 60% | Low |

It also derives an approximate wall thickness (`myo_vol / lv_vol * 10`) and LV mass (`myo_vol * 1.05`) as rough illustrative figures, then returns structured findings, patient-facing recommendations, and physician-facing treatment suggestions matched to the detected condition.

If you plan to use this for anything beyond a demo, this is the function to replace with a real, validated clinical model.

---

## 📄 PDF Report Contents

Generated by `generate_hospital_report()` and saved to `/reports/Report_<patient_id>_<session_id>.pdf`:

1. **Header** — "IAE DIAGNOSTIC REPORT" letterhead with placeholder hospital name/contact info.
2. **Patient Information** table (ID, name, age, gender, reason, referring physician, department, scan type).
3. **Examination Details** table (date/time, modality, "AI Analysis System" label, segmentation method, reconstruction type, status).
4. **AI Analysis Summary** table (condition, severity, risk, confidence, ejection fraction) + a one-line clinical summary paragraph.
5. **Quantitative Cardiac Measurements** table (total heart volume, LV/RV volume, myocardial volume, wall thickness, LV mass, abnormality flag) with reference ranges.
6. **3D Heart Reconstruction** page — matplotlib-rendered front/side/top views of the full mesh.
7. **Part-Wise Cardiac Structure Analysis** — one section per structure with its own rendered image, measured volume, a fixed anatomical description, and a fixed "clinical interpretation" string.
8. **AI-Generated Clinical Findings** — bullet list from the diagnosis engine.
9. **Medical Recommendations** and **Physician Treatment Recommendations** — bullet lists from the diagnosis engine.
10. **Technical Details & Export Validation** — algorithm/rendering/export-format notes and a consistency checklist.
11. **Disclaimer** paragraph.
12. **Authorized By** block — currently hard-coded to a placeholder physician name/signature; **update this before sharing the tool with anyone**, since as written it fabricates a signature.

Temporary PNG renders used to build the PDF are deleted after the PDF is written.

---

## 🔌 API Reference

| Endpoint | Method | Body / Params | Returns |
|---|---|---|---|
| `/` | GET | — | The full dashboard HTML/CSS/JS SPA |
| `/process` | POST | multipart form: `scan` (file), `patient_data` (JSON string) | JSON result: structures, diagnosis, filenames, timings |
| `/history` | GET | — | `{ "history": [...] }` — all processed sessions, most recent first |
| `/session/<session_id>` | GET | — | `{ "success": true, "session": {...} }` or 404 |
| `/download/obj/<session_id>` | GET | — | Full-heart `.obj` file download |
| `/download/part/<session_id>/<part_id>` | GET | `part_id` = structure label (1–7) | Single-structure `.obj` file download |
| `/download/pdf/<session_id>` | GET | — | PDF report download |

`patient_data` JSON shape expected by `/process`:
```json
{
  "patient_id": "P001",
  "name": "Jane Doe",
  "age": "54",
  "gender": "Female",
  "reason": "Routine cardiac assessment",
  "modality": "ct"
}
```

---

## ⚙️ Configuration / Customization

- **`BASE_DIR`** — change this to point at a different Drive path if your dataset lives elsewhere.
- **`HEART_STRUCTURES`** — edit colors, roughness/metalness/subsurface values per structure to change the viewer's look.
- **`CT_LABEL_MAPPING` / `MR_LABEL_MAPPING`** — extend if your dataset uses different raw label integer values.
- **Decimation targets** in `enhance_mesh_realism()` — raise/lower for higher-fidelity vs. faster-loading meshes.
- **Diagnosis thresholds** in `ai_cardiac_diagnosis()` — purely illustrative; replace with real clinical criteria or a trained model if you extend this beyond a demo.
- **`PORT` search range** — `find_free_port(start=5000, end=5100)`, adjust if that range is unavailable in your environment.

---

## ⚠️ Known Limitations

- The "AI diagnosis" is a hard-coded volume-threshold heuristic, not a trained/validated ML or CNN model, despite text like "CNN-Based" appearing in the UI and PDF — update this before treating it as anything more than a demo.
- `PATIENT_HISTORY` and `ANALYSIS_SESSIONS` live only in process memory and are lost whenever the Colab runtime restarts or disconnects.
- No authentication/authorization on any Flask endpoint — do not expose the Colab-proxied URL publicly, especially with real patient data.
- `simulate_multiclass()` is a coarse geometric heuristic and will misclassify chambers on atypical anatomy or scan orientations.
- Mesh decimation targets are fixed constants, not adaptive to available compute — very large volumes may still be slow to process in Colab's CPU-bound runtime.
- The PDF's "Authorized By" section is hard-coded to a placeholder physician — this must be fixed (made dynamic or removed) before the tool is shared or demoed as anything resembling a real report.
- Not validated against clinical ground truth of any kind; volumes, ejection fraction, and wall thickness are approximations only.

---

## 🛠️ Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `Drive mount` step fails/hangs | Browser popup blocked or auth not completed | Re-run the cell and complete the Google auth flow in the popup |
| "No structures detected in scan" error | Voxel counts per label all below the 100-voxel minimum, or label values not in the mapping tables | Check your NIfTI label values against `CT_LABEL_MAPPING`/`MR_LABEL_MAPPING`; extend the mapping if needed |
| Blank/black 3D viewer | WebGL not supported/enabled in your browser, or you switched tabs before `initThreeJS()` finished | Try a different browser, or wait a moment after clicking "3D Reconstruction" |
| Colab proxy URL doesn't open | `eval_js` failed (e.g. Colab connectivity issue) | Re-run the cell; as a fallback, try opening `http://localhost:<PORT>` if running locally |
| Very slow processing on large scans | High-resolution volumes hitting CPU-bound marching cubes + decimation | Downsample your input volume before upload, or increase `step_size` in `marching_cubes` |
| Colab session disconnects and dashboard stops working | Colab enforces idle/runtime timeouts | Re-run the cell to restart the server (history will be empty again) |

---
