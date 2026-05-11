# Coral Guard Analyzer — Web App

A Python pipeline for quantifying residual fouling on ceramic tiles after cleaning, using paired **before/after** images. Results are displayed in a browser-based interface with drag-and-drop file input, per-tile overlays, and summary charts.

This project was built for comparing **treated (SLIPS)** vs **untreated control** tiles in an algae-fouling experiment. Each tile is compared against its **own baseline image**, which makes the analysis more robust to natural tile-to-tile variation.

The pipeline supports both RAW images (such as Olympus `.ORF`) and standard image formats. It automatically detects and crops the tile, computes cleanliness metrics per tile, and optionally displays reflectance spectra from separate measurement files.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place score_tiles.py in this directory (same level as app.py)
cp /path/to/score_tiles.py .

# 3. Put your before baseline images in the before/ folder
#    Filenames must match the after images exactly, e.g.:
#      before/S1.orf  ↔  drop: S1.orf
mkdir -p before

# 4. Run the app
python app.py
```

Then open **http://localhost:5001** in your browser.

---

## Using the App

### Tile Scoring
Drag and drop `.ORF` after-images into the drop zone. Each file is matched to its baseline in `before/` by filename. Files with a matching baseline are shown with ✓; unmatched files are flagged before you run.

Click **Run Scoring** to process. Results appear as:
- A collapsible grid of per-tile overlay images (red = flagged dirty pixels, green contour = tile boundary). Click any tile to open it full-size in a lightbox — use ← / → arrow keys or the on-screen buttons to navigate between tiles.
- A **Summary Charts** section with group comparison bar charts and per-tile dot plots.

### Reflectance Analysis
Drop one or more `.xlsx` reflectance files into the same drop zone alongside (or instead of) `.ORF` files.

- **One xlsx** → 4-panel chart: SLIPS spectra (A), Control spectra (B), difference spectrum (C), mean reflectance at λ = 675 ± 1 nm (D)
- **Multiple xlsx files** → multi-timepoint comparison: spectra overlaid by date, and a 675 nm bar chart across timepoints

Dates are extracted automatically from filenames (e.g. `SLIPS_Reflectance_4-28-26.xlsx` → `4/28/26`). If no date is found, the file is labelled "Unknown Date".

**Expected xlsx format:** one sheet per tile (named S1, C6, etc.), with 3 replicate measurements per sheet. Replicates are detected automatically regardless of how many blank columns separate them.

### Save to Disk
Clicking **Save to Disk** writes to `results/`:
- `summary_bar.png` — group comparison bar chart (Dirty %, ΔE₀₀, Δab)
- `per_tile.png` — per-tile dot plot
- `reflectance_single.png` — single-timepoint reflectance chart (if applicable)
- `reflectance_spectra.png` + `reflectance_675nm.png` — multi-timepoint charts (if applicable)
- `tile_summary.xlsx` + `tile_summary.csv` — metrics table

---

## Directory Layout

```
coral_scorer/
├── app.py              ← Flask server
├── score_tiles.py      ← Image scoring pipeline (place here)
├── requirements.txt
├── README.md
├── before/             ← Baseline .orf images (one per tile, named to match)
├── after/              ← Temp storage for uploaded after images (auto-created)
├── outputs/            ← Generated overlays + charts (auto-created)
│   └── debug/          ← Per-tile overlay images, mask previews, crops
├── session/            ← Current session metrics JSON (auto-created)
├── results/            ← Where "Save to Disk" writes (auto-created)
└── templates/
    └── index.html      ← Browser UI
```

---

## What the Pipeline Does

For each tile, the pipeline:

1. Loads the **before** and **after** image (RAW decoding handled automatically)
2. Detects and crops the tile from the background
3. Converts both images to **CIELAB color space**
4. Builds a per-tile baseline from the **before** image
5. Applies a global session-level color shift correction (accounts for lighting/exposure differences between photo sessions)
6. Measures how much the tile's median color changed after the experiment
7. Computes a secondary "dirty area" percentage using a per-pixel threshold on the corrected residual
8. Saves metrics to the session, generates overlays, and returns everything to the browser

---

## Why This Approach

A straight pixel-by-pixel comparison against a global clean reference can easily overestimate fouling when there are differences in:

- Lighting or exposure between sessions
- White balance
- Glare or reflections
- Camera angle or tile position within the frame

To reduce that, this pipeline uses two levels of analysis:

### Primary metric — Median tile color shift

Measures how far the tile's overall median color moved from its own baseline. Two variants are reported:

- **`Median ΔE₀₀`** — perceptual color change across all three Lab channels, weighted to match human vision (ΔE₀₀ < 1 is imperceptible; > 3 is clearly visible)
- **`Median Δab`** — chromatic shift only (a* and b* channels), less sensitive to brightness changes between sessions. This specifically captures the green-brown hue shift associated with algal fouling.

### Secondary metric — Dirty %

The script first estimates the tile's overall color shift between sessions, subtracts that global shift (correcting for lighting drift), and then asks: what fraction of pixels still look unusually different from baseline?

This avoids falsely labeling the whole tile as dirty because one session's photos were slightly darker or warmer. The threshold is applied in a*b* space only — meaning brightness-only changes (dust, glare) are largely ignored.

### Why CIELAB / a*b* space

The a* axis runs green ↔ red and the b* axis runs blue ↔ yellow. Algal and biofilm fouling on a white/grey tile shifts pixels toward **negative a*** (green), which is the opposite direction from common non-biological blemishes like rust (positive a*) or dust (minimal a/b shift). This makes a*b* a more specific signal for biological fouling than raw RGB or even full ΔE.

---

## Reflectance Metrics

The 675 nm wavelength corresponds to the **chlorophyll-a absorption peak**. Higher reflectance at 675 nm means less light is being absorbed by chlorophyll-a, indicating less algal biomass on the tile surface. This provides an independent optical confirmation of the image-based fouling scores.

The area under the reflectance curve (AUC, 400–700 nm) represents the overall **light enhancement factor** — how much visible light is reflected by the tile surface.

---

## Next Steps & Possible Improvements

### Use per-pixel fouling amount instead of binary threshold
The binary fouled/not-fouled threshold provides a good approximation but is limited in accuracy, especially as fouling accumulates in layers. Averaging the raw residual a*b* distance per pixel across the tile would give a continuous score without the need to pick a threshold at all.

### Directional color gating for algae specificity
Currently Dirty % counts any pixel that shifted beyond the threshold in a*b* space, regardless of direction. Adding a directional gate (e.g. `delta_a < 0`, requiring the shift to be toward green) would make the metric specifically sensitive to chlorophyll-bearing algae, reducing false positives from dust or other non-biological surface changes.

### Use a per-image fouling baseline
Currently the before image provides the baseline and color drift is corrected algorithmically. Including a non-fouled reference tile in each photo session could provide a more stable session-level correction.

### Correlate image scores with reflectance
The image-based Dirty % / Δab scores and the reflectance-based 675 nm metric are measuring the same biological signal from different instruments. Plotting them against each other across all tiles and timepoints would validate both approaches and potentially let one predict the other.


## Screenshots
<img width="400" alt="Screenshot 2026-05-11 at 1 08 51 PM" src="https://github.com/user-attachments/assets/69206330-761d-4371-a7cf-16677d089d5a" /> <img width="400" alt="Screenshot 2026-05-11 at 1 09 49 PM" src="https://github.com/user-attachments/assets/3265406f-9ec4-4ddb-9182-e80adcd87fdd" />

<img width="400" alt="Screenshot 2026-05-11 at 1 10 08 PM" src="https://github.com/user-attachments/assets/484c99fb-3295-481e-89f2-48c25e94f7b4" /> <img width="400" alt="Screenshot 2026-05-11 at 1 10 21 PM" src="https://github.com/user-attachments/assets/d2e540cc-9759-44a1-a42c-c76397a482e9" />


