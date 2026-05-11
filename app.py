"""
Coral Tile Scorer — Flask Web App
Drag-and-drop .orf files → score_tiles.py → visualizations → browser display
Also accepts one or more reflectance .xlsx files for spectral analysis.
"""

import os, io, json, base64, shutil, traceback, re
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# ── Directory layout ───────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
BEFORE_DIR  = BASE_DIR / 'before'
AFTER_DIR   = BASE_DIR / 'after'
OUTPUT_DIR  = BASE_DIR / 'outputs'
DEBUG_DIR   = OUTPUT_DIR / 'debug'
SESSION_DIR = BASE_DIR / 'session'

for d in [BEFORE_DIR, AFTER_DIR, OUTPUT_DIR, DEBUG_DIR, SESSION_DIR]:
    d.mkdir(exist_ok=True, parents=True)

import sys
sys.path.insert(0, str(BASE_DIR))
import score_tiles as st

# ── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    before_tiles = sorted(
        [st.pairing_key(p).upper()
         for p in BEFORE_DIR.iterdir()
         if p.suffix in st.VALID_EXTENSIONS],
        key=st.natural_tile_sort_key
    )
    return render_template('index.html', before_tiles=before_tiles)


@app.route('/process', methods=['POST'])
def process():
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files received'}), 400

    results     = []
    all_metrics = []

    for old in AFTER_DIR.iterdir():
        old.unlink(missing_ok=True)

    saved = []
    for f in files:
        dest = AFTER_DIR / f.filename
        f.save(str(dest))
        saved.append(dest)

    before_lookup = {
        st.pairing_key(p): p
        for p in BEFORE_DIR.iterdir()
        if p.suffix in st.VALID_EXTENSIONS
    }

    for after_path in saved:
        key     = st.pairing_key(after_path)
        tile_id = key.upper()

        before_path = before_lookup.get(key)
        if before_path is None:
            results.append({
                'tile':  tile_id,
                'error': (
                    f'No matching baseline found for "{after_path.name}" in before/.\n'
                    f'Available baselines: {sorted(before_lookup.keys())}'
                )
            })
            continue

        try:
            metrics, overlay_b64 = _score_tile(tile_id, before_path, after_path)
            group = st.infer_group_from_key(key)
            all_metrics.append({
                'Tile':        tile_id,
                'Group':       'Control' if group == 'control' else 'Treated',
                'Dirty %':     metrics['Dirty %'],
                'Median ΔE00': metrics['Median ΔE00'],
                'Median Δab':  metrics['Median Δab'],
            })
            results.append({
                'tile':    tile_id,
                'group':   group,
                'metrics': metrics,
                'overlay': overlay_b64,
            })
        except Exception:
            results.append({
                'tile':  tile_id,
                'error': traceback.format_exc()
            })

    (SESSION_DIR / 'metrics.json').write_text(json.dumps(all_metrics))

    charts = {}
    if all_metrics:
        try:
            charts = _generate_charts(all_metrics)
        except Exception:
            charts = {'error': traceback.format_exc()}

    return jsonify({'results': results, 'charts': charts})


@app.route('/process_reflectance', methods=['POST'])
def process_reflectance():
    """
    Accept one or more reflectance .xlsx files.
    Single file  → 4-panel chart (spectra A/B, difference C, 675nm bar D).
    Multiple files → timepoint comparison: spectra overlay + 675nm over time.
    """
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files received'}), 400

    try:
        if len(files) == 1:
            label     = _extract_date(files[0].filename)
            tile_data, wl_ref = _load_reflectance_xlsx(files[0])
            chart_b64 = _chart_single_timepoint(tile_data, wl_ref, label)
            return jsonify({'mode': 'single', 'chart': chart_b64})
        else:
            timepoints = []
            for f in files:
                label     = _extract_date(f.filename)
                tile_data, wl_ref = _load_reflectance_xlsx(f)
                timepoints.append({'label': label, 'data': tile_data, 'wl': wl_ref})
            # Sort chronologically where possible
            timepoints.sort(key=lambda x: x['label'])
            charts = _chart_multi_timepoint(timepoints)
            return jsonify({'mode': 'multi', **charts})
    except Exception:
        return jsonify({'error': traceback.format_exc()}), 500


@app.route('/save', methods=['POST'])
def save():
    data     = request.json or {}
    save_dir = Path(data.get('path', str(BASE_DIR / 'results')))
    save_dir.mkdir(parents=True, exist_ok=True)

    for chart in ['summary_bar.png', 'per_tile.png',
                  'reflectance_single.png', 'reflectance_spectra.png',
                  'reflectance_675nm.png']:
        src = OUTPUT_DIR / chart
        if src.exists():
            shutil.copy(str(src), str(save_dir / chart))

    session_path = SESSION_DIR / 'metrics.json'
    if session_path.exists():
        metrics = json.loads(session_path.read_text())
        if metrics:
            df = pd.DataFrame(metrics)
            df.to_excel(str(save_dir / 'tile_summary.xlsx'), index=False)
            df.to_csv(str(save_dir / 'tile_summary.csv'),   index=False)

    return jsonify({'success': True, 'path': str(save_dir.resolve())})


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(str(OUTPUT_DIR), filename)


# ── Tile scoring ───────────────────────────────────────────────────────────

def _score_tile(tile_id, before_path, after_path):
    before_rgb = st.read_image_rgb(before_path)
    after_rgb  = st.read_image_rgb(after_path)

    before_crop, before_mask = st.detect_and_crop_tile(before_rgb)
    after_crop,  after_mask  = st.detect_and_crop_tile(after_rgb)

    before_lab = st.extract_lab(before_crop)
    after_lab  = st.extract_lab(after_crop)

    before_pixels = st.tile_pixels_from_mask(before_lab, before_mask)
    after_pixels  = st.tile_pixels_from_mask(after_lab,  after_mask)

    baseline = st.build_before_baseline(before_pixels)

    result = st.score_after_against_before(
        before_pixels = before_pixels,
        after_pixels  = after_pixels,
        after_lab_img = after_lab,
        after_mask    = after_mask,
        threshold_ab  = baseline['before_threshold_ab'],
    )

    metrics = {
        'Dirty %':     result['residual_dirty_percent_ab'],
        'Median ΔE00': result['median_deltaE00_full'],
        'Median Δab':  result['median_delta_ab'],
    }

    if st.SAVE_DEBUG_CROPS:
        import imageio.v3 as iio
        iio.imwrite(str(DEBUG_DIR / f'{tile_id}_before_crop.png'),
                    (before_crop * 255).astype(np.uint8))
        iio.imwrite(str(DEBUG_DIR / f'{tile_id}_after_crop.png'),
                    (after_crop  * 255).astype(np.uint8))

    if st.SAVE_DEBUG_MASKS:
        st.save_mask_preview(before_crop, before_mask,
                             DEBUG_DIR / f'{tile_id}_before_mask.png')
        st.save_mask_preview(after_crop,  after_mask,
                             DEBUG_DIR / f'{tile_id}_after_mask.png')

    overlay_path = DEBUG_DIR / f'{tile_id}_residual_overlay.png'
    st.save_overlay(after_crop, after_mask, result['dirty_vector'], overlay_path)

    with open(str(overlay_path), 'rb') as f:
        overlay_b64 = base64.b64encode(f.read()).decode()

    return metrics, overlay_b64


# ── Cleanliness charts ─────────────────────────────────────────────────────

def _generate_charts(all_metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df          = pd.DataFrame(all_metrics)
    SLIPS_BLUE  = '#1a6bb5'
    CTRL_BLACK  = '#1a1a1a'
    metric_cols = ['Dirty %', 'Median ΔE00', 'Median Δab']
    charts      = {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        'Session Results: SLIPS vs. Control\n'
        'CIELAB Color Space. Mean ± Std. Error (σ / √n)',
        fontsize=13, fontweight='bold', y=1.02
    )

    for ax, col in zip(axes, metric_cols):
        all_vals_max = 0
        bar_info = []
        for gi, (grp, color) in enumerate([('Control', CTRL_BLACK), ('Treated', SLIPS_BLUE)]):
            vals = df[df['Group'] == grp][col].values
            if len(vals) == 0:
                continue
            m   = vals.mean()
            sem = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0
            bar_info.append((gi, m, sem, vals, color))
            all_vals_max = max(all_vals_max, vals.max(), m + sem)

        y_lim = all_vals_max * 1.45 if all_vals_max > 0 else 1
        for gi, m, sem, vals, color in bar_info:
            ax.bar(gi, m, 0.5, yerr=sem, capsize=6, color=color,
                   edgecolor='white', alpha=0.88,
                   error_kw={'elinewidth': 2, 'ecolor': '#444'}, zorder=2)
            np.random.seed(gi)
            jitter = np.random.uniform(-0.12, 0.12, len(vals))
            ax.scatter(gi + jitter, vals, color='white', edgecolor=color,
                       s=45, zorder=5, linewidths=1.5)
            ax.text(gi, 0.6, f'{m:.2f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='white', zorder=10)

        ax.set_ylim(0, y_lim)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Control', 'SLIPS'], fontsize=11)
        ax.set_ylabel(col, fontsize=11)
        ax.set_title(col, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y', linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    bar_path = OUTPUT_DIR / 'summary_bar.png'
    fig.savefig(str(bar_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    with open(str(bar_path), 'rb') as f:
        charts['bar'] = base64.b64encode(f.read()).decode()

    ctrl_tiles  = sorted(df[df['Group']=='Control']['Tile'].tolist(),
                         key=st.natural_tile_sort_key)
    slips_tiles = sorted(df[df['Group']=='Treated']['Tile'].tolist(),
                         key=st.natural_tile_sort_key)
    all_tiles   = ctrl_tiles + slips_tiles
    n_ctrl      = len(ctrl_tiles)

    if all_tiles:
        fig2, axes2 = plt.subplots(
            3, 1,
            figsize=(max(12, len(all_tiles) * 0.75), 12),
            sharex=True
        )
        fig2.suptitle('Per-Tile Results — Current Session',
                      fontsize=13, fontweight='bold', y=1.01)

        for ax2, col in zip(axes2, metric_cols):
            ax2.axvspan(-0.5, n_ctrl - 0.5,                  color='#aaaaaa', alpha=0.10)
            ax2.axvspan(n_ctrl - 0.5, len(all_tiles) - 0.5,  color='#a8c8f0', alpha=0.12)

            for xi, tile in enumerate(all_tiles):
                grp   = 'Treated' if tile.upper().startswith('S') else 'Control'
                color = SLIPS_BLUE if grp == 'Treated' else CTRL_BLACK
                row   = df[df['Tile'] == tile]
                if not row.empty:
                    ax2.scatter(xi, row[col].values[0], color=color, s=80, zorder=5)

            ymax = ax2.get_ylim()[1]
            ax2.text(n_ctrl / 2 - 0.5, ymax * 0.97, 'Control',
                     ha='center', fontsize=10, fontweight='bold', color='#555', va='top')
            ax2.text(n_ctrl + len(slips_tiles) / 2 - 0.5, ymax * 0.97, 'SLIPS',
                     ha='center', fontsize=10, fontweight='bold', color=SLIPS_BLUE, va='top')
            ax2.set_ylabel(col, fontsize=11)
            ax2.set_title(col, fontweight='bold', fontsize=11)
            ax2.grid(True, alpha=0.2, axis='y', linestyle=':')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

        axes2[-1].set_xticks(range(len(all_tiles)))
        axes2[-1].set_xticklabels(all_tiles, rotation=45, ha='right', fontsize=9)
        plt.tight_layout()

        tile_path = OUTPUT_DIR / 'per_tile.png'
        fig2.savefig(str(tile_path), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        with open(str(tile_path), 'rb') as f:
            charts['per_tile'] = base64.b64encode(f.read()).decode()

    return charts


# ── Reflectance helpers ────────────────────────────────────────────────────

def _extract_date(filename: str) -> str:
    """Extract a date label from a filename, or return 'Unknown Date'."""
    stem = Path(filename).stem
    patterns = [
        r'(\d{1,2}[-_/]\d{1,2}[-_/]\d{2,4})',   # M-D-YY or M-D-YYYY
        r'(\d{4}[-_/]\d{1,2}[-_/]\d{1,2})',       # YYYY-M-D
        r'(\d{6,8})',                               # plain digits MMDDYY / YYYYMMDD
    ]
    for pat in patterns:
        m = re.search(pat, stem)
        if m:
            return m.group(1).replace('_', '/').replace('-', '/')
    return 'Unknown Date'


def _load_reflectance_xlsx(file_obj):
    """
    Load a per-tile-per-sheet reflectance Excel file.
    3 replicates per sheet detected dynamically (handles variable blank columns).
    Returns (tile_data dict, wl_ref array).
    """
    xl        = pd.ExcelFile(file_obj)
    tile_data = {}
    wl_ref    = None

    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        try:
            wl = df.iloc[1:, 0].astype(float).values
        except Exception:
            continue
        reps = []
        for ci in range(1, df.shape[1]):          # skip col 0 (wavelength)
            try:
                col = pd.to_numeric(df.iloc[1:, ci], errors='coerce').values
            except Exception:
                continue
            valid = col[~np.isnan(col)]
            if len(valid) < 100:                   # not enough data — blank column
                continue
            col_mean = valid.mean()
            if col_mean < 0 or col_mean > 150:     # wavelength cols (~500) skipped here
                continue
            reps.append(col)
            if len(reps) == 3:                     # stop after 3 replicates
                break
        if not reps:
            continue
        reps = np.array(reps)
        tile_data[sheet] = {'mean': reps.mean(axis=0), 'std': reps.std(axis=0)}
        if wl_ref is None:
            wl_ref = wl

    if wl_ref is None or not tile_data:
        raise ValueError('No readable tile sheets found in the Excel file.')

    return tile_data, wl_ref


def _b64_fig(fig, path: Path) -> str:
    """Save a matplotlib figure and return its base64 encoding."""
    fig.savefig(str(path), dpi=150, bbox_inches='tight', facecolor='white')
    import matplotlib.pyplot as plt
    plt.close(fig)
    with open(str(path), 'rb') as f:
        return base64.b64encode(f.read()).decode()


def _reflectance_arrays(tile_data, wl_ref):
    """Return vis-filtered arrays and group splits."""
    vis    = (wl_ref >= 400) & (wl_ref <= 700)
    wl_vis = wl_ref[vis]

    slips_tiles = sorted([t for t in tile_data if t.upper().startswith('S')],
                         key=st.natural_tile_sort_key)
    ctrl_tiles  = sorted([t for t in tile_data if t.upper().startswith('C')],
                         key=st.natural_tile_sort_key)

    def stack(tiles):
        return np.array([tile_data[t]['mean'][vis] for t in tiles]) \
               if tiles else np.empty((0, int(vis.sum())))

    slips_means = stack(slips_tiles)
    ctrl_means  = stack(ctrl_tiles)

    def grand(arr): return arr.mean(axis=0) if len(arr) else np.zeros(int(vis.sum()))
    def se(arr):    return arr.std(axis=0) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros(int(vis.sum()))

    chl_mask = (wl_vis >= 674) & (wl_vis <= 676)

    return {
        'wl_vis': wl_vis,
        'slips_tiles': slips_tiles,
        'ctrl_tiles':  ctrl_tiles,
        'slips_means': slips_means,
        'ctrl_means':  ctrl_means,
        'slips_grand': grand(slips_means),
        'ctrl_grand':  grand(ctrl_means),
        'slips_se':    se(slips_means),
        'ctrl_se':     se(ctrl_means),
        'chl_mask':    chl_mask,
        'slips_chl':   [tile_data[t]['mean'][vis][chl_mask].mean() for t in slips_tiles],
        'ctrl_chl':    [tile_data[t]['mean'][vis][chl_mask].mean() for t in ctrl_tiles],
    }


# ── Single-timepoint 4-panel chart ────────────────────────────────────────

def _chart_single_timepoint(tile_data, wl_ref, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    a = _reflectance_arrays(tile_data, wl_ref)
    wl_vis = a['wl_vis']; chl_mask = a['chl_mask']

    SLIPS_BLUE = '#1a6bb5'; CTRL_BLACK = '#1a1a1a'
    SLIPS_FILL = '#a8c8f0'; CTRL_FILL  = '#aaaaaa'
    slips_tile_c = ['#5b9bd5','#2e75b6','#1f4e79','#9dc3e6','#0070c0',
                    '#3a7abf','#6baed6','#2171b5','#084594','#4292c6',
                    '#6baed6','#2171b5','#084594']
    ctrl_tile_c  = ['#595959','#404040','#808080','#262626','#bfbfbf',
                    '#737373','#999999','#4d4d4d','#1a1a1a','#d9d9d9',
                    '#737373','#999999','#4d4d4d']

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                            height_ratios=[1.1, 1])
    ax_s    = fig.add_subplot(gs[0, 0])
    ax_c    = fig.add_subplot(gs[0, 1])
    ax_diff = fig.add_subplot(gs[1, 0])
    ax_chl  = fig.add_subplot(gs[1, 1])

    YMIN, YMAX = 20, 100
    diff = a['slips_grand'] - a['ctrl_grand']

    def plot_spectra(ax, means, tc, gnd, gnd_se, gc, gf, title):
        for i in range(len(means)):
            ax.plot(wl_vis, means[i], color=tc[i % len(tc)], lw=1.0, alpha=0.55, zorder=2)
        ax.fill_between(wl_vis, gnd - gnd_se, gnd + gnd_se, color=gf, alpha=0.50, zorder=3)
        ax.plot(wl_vis, gnd, color=gc, lw=2.5, label='Mean ± Std. Error', zorder=4)
        ax.axvspan(674, 676, color='red', alpha=0.15, zorder=1)
        ax.axvline(675, color='red', lw=1.2, ls='--', alpha=0.75, zorder=5)
        ax.set_xlim(400, 700); ax.set_ylim(YMIN, YMAX)
        ax.set_xlabel('Wavelength (nm)', fontsize=11)
        ax.set_ylabel('Reflectance (%)', fontsize=11)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.2, linestyle=':')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(fontsize=8.5, loc='lower right')

    n_s = len(a['slips_tiles']); n_c = len(a['ctrl_tiles'])
    plot_spectra(ax_s, a['slips_means'], slips_tile_c, a['slips_grand'], a['slips_se'],
                 SLIPS_BLUE, SLIPS_FILL,
                 f'A  |  SLIPS Tiles (n={n_s})\nMean ± Std. Error, individual tiles shown')
    plot_spectra(ax_c, a['ctrl_means'],  ctrl_tile_c,  a['ctrl_grand'],  a['ctrl_se'],
                 CTRL_BLACK, CTRL_FILL,
                 f'B  |  Control Tiles (n={n_c})\nMean ± Std. Error, individual tiles shown')
    ax_c.set_ylabel('')

    ax_diff.axhline(0, color='k', lw=1.0, alpha=0.4)
    ax_diff.fill_between(wl_vis, 0, diff, where=(diff >= 0),
                         color=SLIPS_BLUE, alpha=0.35, label='SLIPS > Control')
    ax_diff.fill_between(wl_vis, 0, diff, where=(diff < 0),
                         color='tomato', alpha=0.35, label='Control > SLIPS')
    ax_diff.plot(wl_vis, diff, color=SLIPS_BLUE, lw=2.0)
    ax_diff.axvspan(674, 676, color='red', alpha=0.15)
    ax_diff.axvline(675, color='red', lw=1.2, ls='--', alpha=0.75, label='675 nm (Chl-a)')
    ax_diff.set_xlim(400, 700)
    ax_diff.set_xlabel('Wavelength (nm)', fontsize=11)
    ax_diff.set_ylabel('ΔReflectance (%)\n[SLIPS − Control]', fontsize=11)
    ax_diff.set_title('C  |  Difference Spectrum (SLIPS − Control)\n'
                      'Positive = SLIPS reflects more; Negative = Control reflects more',
                      fontweight='bold', fontsize=11)
    ax_diff.grid(True, alpha=0.2, linestyle=':')
    ax_diff.spines['top'].set_visible(False); ax_diff.spines['right'].set_visible(False)
    ax_diff.legend(fontsize=8.5, loc='upper left')
    if chl_mask.any():
        chl_diff = diff[chl_mask].mean()
        ax_diff.annotate(f'Δ675nm = {chl_diff:+.1f}%',
                         xy=(675, chl_diff), xytext=(600, chl_diff + 3),
                         fontsize=9, color='red',
                         arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

    def safe_mean(lst): return float(np.nanmean(lst)) if lst else 0.0
    def safe_std(lst):  return float(np.nanstd(lst))  if lst else 0.0

    sc = safe_mean(a['slips_chl']); ss = safe_std(a['slips_chl'])
    cc = safe_mean(a['ctrl_chl']);  cs = safe_std(a['ctrl_chl'])
    colors = [CTRL_BLACK, SLIPS_BLUE]
    ax_chl.bar(['Control','SLIPS'], [cc, sc], yerr=[cs, ss],
               capsize=7, width=0.45, color=colors, edgecolor='white',
               linewidth=1.2, alpha=0.85,
               error_kw={'elinewidth': 2, 'ecolor': '#444'}, zorder=2)
    np.random.seed(7)
    jitter = np.linspace(-0.10, 0.10, max(len(a['ctrl_chl']), len(a['slips_chl']), 1))
    for j, (vals, xpos) in enumerate(zip([a['ctrl_chl'], a['slips_chl']], [0, 1])):
        if vals:
            sidx = np.argsort(vals)
            ax_chl.scatter(jitter[:len(vals)] + xpos, np.array(vals)[sidx],
                           color='white', edgecolor=colors[j], s=58, zorder=5, linewidths=1.5)
    y_top = max(cc + cs, sc + ss, 1.0)
    y_lim = y_top * 1.42
    if not np.isfinite(y_lim) or y_lim <= 0:
        y_lim = 100.0
    ax_chl.text(0, cc + cs + y_lim * 0.04, f'{cc:.1f}\n±{cs:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=CTRL_BLACK)
    ax_chl.text(1, sc + ss + y_lim * 0.04, f'{sc:.1f}\n±{ss:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=SLIPS_BLUE)
    ax_chl.text(0.5, y_lim * 0.96, f'Δ = {sc - cc:+.1f}%',
                ha='center', va='top', fontsize=10, color='dimgray', style='italic')
    ax_chl.set_ylim(0, y_lim)
    ax_chl.set_ylabel('Reflectance (%)', fontsize=10)
    ax_chl.set_title('D  |  Mean reflectance for λ = 675 ± 1\n(Mean ± SD)',
                     fontweight='bold', fontsize=11)
    ax_chl.grid(True, alpha=0.2, axis='y', linestyle=':')
    ax_chl.spines['top'].set_visible(False); ax_chl.spines['right'].set_visible(False)
    ax_chl.text(0.5, -0.13, 'Higher reflectance → less Chl-a absorption',
                transform=ax_chl.transAxes, ha='center', fontsize=8, color='gray', style='italic')

    title_label = f' — {label}' if label != 'Unknown Date' else ''
    fig.suptitle(f'SLIPS vs. Control: Visible Reflectance Analysis (λ = 400–700 nm){title_label}',
                 fontsize=14, fontweight='bold', y=1.02)

    return _b64_fig(fig, OUTPUT_DIR / 'reflectance_single.png')


# ── Multi-timepoint reflectance charts ────────────────────────────────────

def _chart_multi_timepoint(timepoints):
    """
    Generate two charts for multiple reflectance files:
    1. Spectra overlay — SLIPS + Control grand means per timepoint
    2. 675nm metric over time — grouped bar chart
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n   = len(timepoints)
    # Colour ramps: light → dark per group
    ctrl_palette  = ['#aaaaaa', '#666666', '#333333', '#1a1a1a',
                     '#888888', '#444444'][:n]
    slips_palette = ['#89c4e1', '#4a9fd4', '#1a6bb5', '#0f3d66',
                     '#6baed6', '#2171b5'][:n]

    SLIPS_BLUE = '#1a6bb5'
    CTRL_BLACK = '#1a1a1a'
    labels     = [tp['label'] for tp in timepoints]

    # Pre-compute per-timepoint arrays
    arrays = [_reflectance_arrays(tp['data'], tp['wl']) for tp in timepoints]
    wl_vis = arrays[0]['wl_vis']

    # ── Chart 1: Spectra overlay ──────────────────────────────────────────
    fig1, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig1.suptitle('SLIPS vs. Control Through Time — Visible Reflectance (λ = 400–700 nm)\n'
                  'CIELAB Color Space. Mean ± Std. Error (σ / √n)',
                  fontsize=13, fontweight='bold', y=1.02)

    for i, (a, lbl) in enumerate(zip(arrays, labels)):
        ls = ['-', '--', ':', '-.'][i % 4]
        ax_s.plot(wl_vis, a['slips_grand'], color=slips_palette[i], lw=2.0,
                  ls=ls, label=lbl)
        ax_s.fill_between(wl_vis, a['slips_grand'] - a['slips_se'],
                          a['slips_grand'] + a['slips_se'],
                          color=slips_palette[i], alpha=0.15)
        ax_c.plot(wl_vis, a['ctrl_grand'], color=ctrl_palette[i], lw=2.0,
                  ls=ls, label=lbl)
        ax_c.fill_between(wl_vis, a['ctrl_grand'] - a['ctrl_se'],
                          a['ctrl_grand'] + a['ctrl_se'],
                          color=ctrl_palette[i], alpha=0.15)

    for ax, title, color in [(ax_s, 'SLIPS', SLIPS_BLUE), (ax_c, 'Control', CTRL_BLACK)]:
        ax.axvspan(674, 676, color='red', alpha=0.12)
        ax.axvline(675, color='red', lw=1.0, ls='--', alpha=0.6)
        ax.set_xlim(400, 700); ax.set_ylim(20, 100)
        ax.set_xlabel('Wavelength (nm)', fontsize=11)
        ax.set_ylabel('Reflectance (%)', fontsize=11)
        ax.set_title(f'{title} — Mean Spectra by Timepoint\n(shading = ± Std. Error)',
                     fontweight='bold', fontsize=11, color=color)
        ax.legend(fontsize=9, title='Timepoint', title_fontsize=8)
        ax.grid(True, alpha=0.2, linestyle=':')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    spectra_b64 = _b64_fig(fig1, OUTPUT_DIR / 'reflectance_spectra.png')

    # ── Chart 2: 675nm over time ──────────────────────────────────────────
    fig2, axes = plt.subplots(1, 1, figsize=(max(8, n * 2.5), 6))
    fig2.suptitle('Mean Reflectance at λ = 675 ± 1 nm Over Time\n'
                  'CIELAB Color Space. Mean ± Std. Error (σ / √n)',
                  fontsize=13, fontweight='bold', y=1.02)

    bar_w      = 0.35
    x          = np.arange(n)
    ctrl_means = [np.mean(a['ctrl_chl'])  if a['ctrl_chl']  else 0 for a in arrays]
    ctrl_sems  = [np.std(a['ctrl_chl']) / np.sqrt(len(a['ctrl_chl']))
                  if len(a['ctrl_chl']) > 1 else 0 for a in arrays]
    slips_means= [np.mean(a['slips_chl']) if a['slips_chl'] else 0 for a in arrays]
    slips_sems = [np.std(a['slips_chl']) / np.sqrt(len(a['slips_chl']))
                  if len(a['slips_chl']) > 1 else 0 for a in arrays]

    axes.bar(x - bar_w/2, ctrl_means, bar_w, yerr=ctrl_sems, capsize=5,
             color=ctrl_palette, edgecolor='white', alpha=0.88,
             error_kw={'elinewidth': 1.8, 'ecolor': '#333'}, zorder=2,
             label='Control')
    axes.bar(x + bar_w/2, slips_means, bar_w, yerr=slips_sems, capsize=5,
             color=slips_palette, edgecolor='white', alpha=0.88,
             error_kw={'elinewidth': 1.8, 'ecolor': '#333'}, zorder=2,
             label='SLIPS')

    # Per-tile dots
    np.random.seed(7)
    for i, a in enumerate(arrays):
        jc = np.random.uniform(-0.08, 0.08, len(a['ctrl_chl']))
        js = np.random.uniform(-0.08, 0.08, len(a['slips_chl']))
        axes.scatter(i - bar_w/2 + jc, a['ctrl_chl'],
                     color='white', edgecolor=ctrl_palette[i], s=30, zorder=5, linewidths=1.2)
        axes.scatter(i + bar_w/2 + js, a['slips_chl'],
                     color='white', edgecolor=slips_palette[i], s=30, zorder=5, linewidths=1.2)

    # Mean labels inside bars
    for i in range(n):
        axes.text(i - bar_w/2, 0.8, f'{ctrl_means[i]:.1f}',
                  ha='center', va='bottom', fontsize=8, fontweight='bold',
                  color='white', zorder=10)
        axes.text(i + bar_w/2, 0.8, f'{slips_means[i]:.1f}',
                  ha='center', va='bottom', fontsize=8, fontweight='bold',
                  color='white', zorder=10)

    axes.set_xticks(x)
    axes.set_xticklabels(labels, fontsize=11)
    axes.set_ylabel('Reflectance (%) at 675 ± 1 nm', fontsize=11)
    axes.set_title('Higher reflectance → less Chl-a absorption → less algae',
                   fontsize=10, color='gray', style='italic')
    all_vals = [v for v in ctrl_means + slips_means if np.isfinite(v) and v > 0]
    top = max(all_vals) if all_vals else 10.0
    axes.set_ylim(0, top * 1.45)
    axes.grid(True, alpha=0.2, axis='y', linestyle=':')
    axes.spines['top'].set_visible(False); axes.spines['right'].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CTRL_BLACK,  label='Control'),
        Patch(facecolor=SLIPS_BLUE,  label='SLIPS'),
    ]
    axes.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()
    chl_b64 = _b64_fig(fig2, OUTPUT_DIR / 'reflectance_675nm.png')

    return {'spectra': spectra_b64, 'chl675': chl_b64}


if __name__ == '__main__':
    print("\n  Coral Guard Tile Scorer running at http://localhost:5001\n")
    app.run(debug=True, port=5001)
