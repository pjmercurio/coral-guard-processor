# Coral Guard Processor

A Python pipeline for quantifying residual fouling on ceramic tiles after cleaning, using paired **before/after** images.

This project was built for comparing **treated** vs **untreated control** tiles in an algae-fouling experiment. Each tile is compared against its **own baseline image**, which makes the analysis more robust to natural tile-to-tile variation.

The script supports both RAW images (such as Olympus `.ORF`) and standard image formats, automatically detects/crops the tile, and outputs per-tile cleanliness metrics plus group-level summaries.

---

## What this does

For each tile, the pipeline:

1. Loads the **before** and **after** image
2. Detects and crops the tile from the background
3. Builds a baseline from the tile’s **before** image
4. Measures how much the tile’s median color changed after the experiment
5. Computes a secondary “dirty area” percentage after correcting for overall image-to-image color shift
6. Saves CSV summaries and debug overlays

This is designed to be more stable than a naive per-pixel comparison against a single global clean reference.

---

## Why this approach

A straight pixel-by-pixel “dirty mask” can easily overestimate fouling when there are differences in:

- lighting
- exposure
- glare
- white balance
- image angle
- tile position within the frame

To reduce that problem, this pipeline uses two levels of analysis:

### Primary metric
**Median tile color shift**

This measures how far the tile’s overall median color moved from its own baseline image.

Two variants are reported:

- `median_deltaE00_full` — overall perceptual color change in Lab space
- `median_delta_ab` — chromatic shift only (`a*` / `b*`), which is less sensitive to brightness changes

### Secondary metric
**Residual dirty percent**

The script first estimates the tile’s overall color shift between sessions, subtracts that global shift, and only then asks:

> What fraction of pixels still look unusually different from baseline?

This helps avoid falsely labeling the whole tile as dirty because one image was slightly darker or warmer.

---

## Supported folder layouts

### Option 1: grouped subfolders (recommended)

```text
before/
  treated/
    S1.ORF
    S2.ORF
    S3.ORF
  control/
    C1.ORF
    C2.ORF
    C3.ORF

after/
  treated/
    S1.ORF
    S2.ORF
    S3.ORF
  control/
    C1.ORF
    C2.ORF
    C3.ORF
```

### Option 2: flat structure
```text
before/
  S1.ORF
  S2.ORF
  S3.ORF
  C1.ORF
  C2.ORF
  C3.ORF

after/
  S1.ORF
  S2.ORF
  S3.ORF
  C1.ORF
  C2.ORF
  C3.ORF
```

## Next Steps & Possible Improvements

### Use per-pixel fouling amount instead of binary threshold
The binary fouled/not fouled approximation provides a good crude approximation, but is limited in accuracy, especially as more fouling accumulates and layers. 

### Use a per-image fouling baseline
Currently we use the original set of images and correct for color variation algorithmically to set a baseline for fouling, however a non-fouled subject alongside each tile may provide a more accurate baseline.

### Experiment with algae spectra
The Lab colorspace works well so far for ignoring non-fouling-related blemishes on the tile, but especially as we explore fouling amount per pixel, we may want to make sure we're only counting specific spectra reflected by algae.
