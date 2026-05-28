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
import cv2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024  # 2 GB

# ── Directory layout ───────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
BEFORE_DIR  = BASE_DIR / 'before'
AFTER_DIR   = BASE_DIR / 'after'
OUTPUT_DIR  = BASE_DIR / 'outputs'
DEBUG_DIR   = OUTPUT_DIR / 'debug'
SESSION_DIR = BASE_DIR / 'session'
CORAL_LABEL_DIR = BASE_DIR / 'coral_labels'
_coral_labels = {}   # stem → [[x,y], ...] polygon points

for d in [BEFORE_DIR, AFTER_DIR, OUTPUT_DIR, DEBUG_DIR, SESSION_DIR]:
    d.mkdir(exist_ok=True, parents=True)

import sys
sys.path.insert(0, str(BASE_DIR))
import score_tiles as st

def _load_coral_labels():
    global _coral_labels
    if not CORAL_LABEL_DIR.exists():
        return
    for json_path in CORAL_LABEL_DIR.glob('*.json'):
        try:
            data = json.load(open(json_path))
            for shape in data['shapes']:
                if shape['label'] == 'coral' and shape['points']:
                    _coral_labels[json_path.stem] = shape['points']
                    break
        except Exception:
            pass
    print(f'  Loaded {len(_coral_labels)} coral labels from {CORAL_LABEL_DIR}')

_load_coral_labels()

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


@app.route('/progress')
def get_progress():
    try:
        data = json.loads((SESSION_DIR / 'progress.json').read_text())
    except Exception:
        data = {'pct': 0, 'msg': ''}
    return jsonify(data)

def _set_progress(pct, msg=''):
    (SESSION_DIR / 'progress.json').write_text(
        json.dumps({'pct': pct, 'msg': msg})
    )


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
                  'reflectance_675nm.png', 'treatment_chart.png']:
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
            ax.text(gi, m * 0.5, f'{m:.2f}', ha='center', va='bottom',
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




# ═══════════════════════════════════════════════════════════════════════════
# TREATMENT EXPERIMENT MODE
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/process_treatments', methods=['POST'])
def process_treatments():
    """
    Accept ORF files grouped by treatment (form field = 'treatment:{name}').
    Score each tile excluding coral fragment, using pooled before/ as baseline.
    Returns 3-panel chart with one bar per treatment.
    """
    # Group uploaded files by treatment name
    _set_progress(0, 'Starting treatment analysis…')
    treatments = {}
    for field_name in request.files:
        if field_name.startswith('treatment:'):
            treatment = field_name[len('treatment:'):]
            treatments[treatment] = request.files.getlist(field_name)

    if not treatments:
        return jsonify({'error': 'No treatment files received'}), 400

    # Build global baseline from before/ images (pooled)
    try:
        _set_progress(5, 'Building baseline…')
        global_baseline = _compute_global_baseline()
        _set_progress(15, 'Baseline ready. Scoring tiles…')
    except Exception as e:
        return jsonify({'error': f'Failed to compute baseline from before/ folder: {traceback.format_exc()}'}), 500

    # Score each treatment
    total = sum(len(f) for f in treatments.values())
    scored = 0
    treatment_results = {}
    treatment_tiles   = {}
    for treatment, files in treatments.items():
        tile_metrics = []
        tiles_out    = []
        for f in files:
            tmp_path = AFTER_DIR / f.filename
            f.save(str(tmp_path))
            tile_id  = Path(f.filename).stem.upper()
            try:
                m, overlay_b64 = _score_tile_treatment(tile_id, tmp_path, global_baseline, treatment)
                tile_metrics.append(m)
                tiles_out.append({'id': tile_id, 'metrics': m, 'overlay': overlay_b64})
                scored += 1
                pct = 15 + int(80 * scored / total)
                _set_progress(pct, f'Scored {scored}/{total} tiles…')
            except Exception:
                pass
            finally:
                tmp_path.unlink(missing_ok=True)

        if tile_metrics:
            treatment_results[treatment] = {
                'Dirty %':        float(np.mean([m['Dirty %']     for m in tile_metrics])),
                'Median ΔE00':    float(np.mean([m['Median ΔE00'] for m in tile_metrics])),
                'Median Δab':     float(np.mean([m['Median Δab']  for m in tile_metrics])),
                'Dirty % sd':     float(np.std( [m['Dirty %']     for m in tile_metrics])),
                'Median ΔE00 sd': float(np.std( [m['Median ΔE00'] for m in tile_metrics])),
                'Median Δab sd':  float(np.std( [m['Median Δab']  for m in tile_metrics])),
                'n': len(tile_metrics),
            }
            treatment_tiles[treatment] = tiles_out

    if not treatment_results:
        return jsonify({'error': 'All tiles failed to process.'}), 500

    # Save treatment metrics to session
    (SESSION_DIR / 'treatment_metrics.json').write_text(json.dumps(treatment_results))

    try:
        _set_progress(95, 'Generating charts…')
        chart_b64 = _generate_treatment_charts(treatment_results)
        _set_progress(100, 'Done!')
    except Exception:
        return jsonify({'error': traceback.format_exc()}), 500

    return jsonify({
        'treatment_metrics': treatment_results,
        'treatment_tiles':   treatment_tiles,
        'chart':             chart_b64,
    })


def _compute_global_baseline():
    """
    Pool reference images into a single baseline for treatment scoring.
    Uses tile_reference/ folder if it exists (clean tile patches),
    otherwise falls back to the first image in before/.
    """
    ref_dir = BASE_DIR / 'birch_treatment_reference'

    if ref_dir.exists() and any(ref_dir.iterdir()):
        ref_imgs = [p for p in ref_dir.iterdir()
                    if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}]
        print(f'  Using {len(ref_imgs)} clean tile reference patches from tile_reference/')
    else:
        ref_imgs = [p for p in BEFORE_DIR.iterdir()
                    if p.suffix in st.VALID_EXTENSIONS][:1]
        print(f'  No tile_reference/ found — using before/ images as baseline')

    if not ref_imgs:
        raise ValueError('No reference images found in tile_reference/ or before/.')

    all_pixels = []
    for img_path in ref_imgs:
        try:
            rgb  = st.read_image_rgb(img_path)
            crop, mask = st.detect_and_crop_tile(rgb)
            lab  = st.extract_lab(crop)
            pixels = st.tile_pixels_from_mask(lab, mask)
            del rgb, crop, lab, mask
            all_pixels.append(pixels)
        except Exception as e:
            print(f'  Skipping {img_path.name}: {e}')
            continue

    if not all_pixels:
        raise ValueError('Could not process any reference images.')

    pooled   = np.vstack(all_pixels)
    baseline = st.build_before_baseline(pooled)
    baseline['pixels'] = pooled
    return baseline


def _segment_coral_color(lab_img, tile_mask):
    """LAB color classifier + largest connected component fallback."""
    L = lab_img[:, :, 0]
    A = lab_img[:, :, 1]
    B = lab_img[:, :, 2]

    if tile_mask.sum() < 10:
        return np.zeros(lab_img.shape[:2], dtype=bool)

    brown_coral  = (L < 55) & (A >  5) & (B > -5)
    green_polyps = (L < 65) & (A < -8)
    raw_mask = ((brown_coral | green_polyps) & tile_mask).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if n_labels <= 1:
        return raw_mask.astype(bool) & tile_mask

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    blob    = (labels == largest).astype(np.uint8)

    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(blob)
    if contours:
        cv2.drawContours(
            filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED
        )

    coral_mask = filled.astype(bool) & tile_mask
    if coral_mask.sum() / max(tile_mask.sum(), 1) > 0.85:
        thresh = float(np.percentile(L[tile_mask], 40))
        coral_mask = (L < thresh) & tile_mask

    return coral_mask

# USE THE UPDATED LAB SPACE
# def _segment_coral(lab_img, tile_mask):
#     L = lab_img[:, :, 0]
#     A = lab_img[:, :, 1]
#     B = lab_img[:, :, 2]

#     if tile_mask.sum() < 10:
#         return np.zeros(lab_img.shape[:2], dtype=bool)

#     # Brown coral tissue
#     brown_coral = (L < 55) & (A >  5) & (B > -5)
#     # Green zoanthid polyps
#     green_polyps = (L < 65) & (A < -8)

#     raw_mask = ((brown_coral | green_polyps) & tile_mask).astype(np.uint8)

#     # Large closing kernel to bridge gaps between polyps and tissue
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
#     closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

#     # Find the largest connected component (the coral body)
#     n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
#     if n_labels <= 1:
#         return raw_mask.astype(bool) & tile_mask

#     largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#     largest_blob = (labels == largest).astype(np.uint8)

#     # Fill the entire contour solid — this catches all holes including polyps
#     contours, _ = cv2.findContours(largest_blob, cv2.RETR_EXTERNAL,
#                                    cv2.CHAIN_APPROX_SIMPLE)
#     filled = np.zeros_like(largest_blob)
#     if contours:
#         biggest = max(contours, key=cv2.contourArea)
#         cv2.drawContours(filled, [biggest], -1, 255, cv2.FILLED)

#     # Sanity check
#     coral_mask = filled.astype(bool) & tile_mask
#     if coral_mask.sum() / tile_mask.sum() > 0.85:
#         thresh = float(np.percentile(L[tile_mask], 40))
#         coral_mask = (L < thresh) & tile_mask

#     return coral_mask


def _segment_coral(lab_img, tile_mask):
    L = lab_img[:, :, 0]
    A = lab_img[:, :, 1]
    tile_L = L[tile_mask]

    if len(tile_L) < 10:
        return np.zeros(lab_img.shape[:2], dtype=bool)

    thresh = float(np.percentile(tile_L, 28))

    # Require both dark AND reddish — excludes green algae and neutral tile
    raw_mask = (L < thresh) & (L < 55) & (A > 2) & tile_mask

    # Sanity check
    if raw_mask.sum() / tile_mask.sum() > 0.70:
        thresh   = float(np.percentile(tile_L, 15))
        raw_mask = (L < thresh) & (L < 55) & (A > 2) & tile_mask

    # Close small gaps between detected coral pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    closed = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # Keep only the largest connected component (the coral body)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if n_labels <= 1:
        return raw_mask & tile_mask

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    blob    = (labels == largest).astype(np.uint8)

    # Fill the entire contour solid — polyp holes get included automatically
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(blob)
    if contours:
        cv2.drawContours(
            filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED
        )

    return filled.astype(bool) & tile_mask


def _save_treatment_overlay(crop, tile_mask, coral_mask, dirty_vector, path):
    """
    3-layer overlay for treatment tiles:
      - Brown shading  = coral fragment (excluded from scoring)
      - Red pixels     = fouling detected on tile surface
      - Green contour  = tile boundary
    """
    import cv2
    overlay = (crop * 255).astype(np.uint8).copy()
    # overlay[coral_mask] = [101, 67, 33]                # brown = coral
    yellow = np.array([255, 255, 0], dtype=np.float32) # yellow = coral
    orig  = overlay[coral_mask].astype(np.float32)
    overlay[coral_mask] = (0.45 * yellow + 0.55 * orig).clip(0, 255).astype(np.uint8)
    tile_only = tile_mask & ~coral_mask
    dirty_2d  = np.zeros(tile_mask.shape, dtype=bool)
    dirty_2d[tile_only] = dirty_vector
    orig = overlay[dirty_2d].astype(np.float32)
    red  = np.array([220, 50, 50], dtype=np.float32) # red = fouling
    overlay[dirty_2d] = (0.6 * red + 0.4 * orig).clip(0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        tile_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 200, 0), 2)  # green contour
    import imageio.v3 as iio
    iio.imwrite(str(path), overlay)


def _score_tile_treatment(tile_id, image_path, global_baseline, treatment_name=''):
    print(f"  Loading {tile_id}...")
    """
    Score one treatment tile against the global pooled baseline,
    excluding pixels identified as coral fragment.
    """
    rgb  = st.read_image_rgb(image_path)

    # Normalize exposure so tile detection works regardless of how dark the photo is.
    # Stretches the brightest pixels back to 1.0 without clipping shadows.
    rgb_max = rgb.max()
    if rgb_max > 0.01:
        rgb = (rgb / rgb_max).astype(np.float32)

    print(f"  Cropping {tile_id}...")
    crop, mask = st.detect_and_crop_tile(rgb)

    # Downsample to max 800px on longest side — cuts memory ~10-20x for high-res ORF
    import cv2
    h, w = crop.shape[:2]
    scale = min(1.0, 800 / max(h, w))
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    print(f"  [{tile_id}] crop shape after resize: {crop.shape}")

    del rgb
    print(f"  Extracting LAB {tile_id}...")
    lab  = st.extract_lab(crop)
    print(f"  Segmenting coral {tile_id}...")
    # Exclude coral — mask it out before scoring
    coral_mask      = _segment_coral(lab, mask)
    tile_only_mask  = mask & ~coral_mask
    print(f"  Scoring {tile_id}...")

    if tile_only_mask.sum() < 100:
        raise ValueError(f'Too few tile pixels after coral segmentation ({tile_only_mask.sum()})')

    after_pixels = st.tile_pixels_from_mask(lab, tile_only_mask)

    before_median = np.median(global_baseline['pixels'], axis=0)
    after_median  = np.median(after_pixels, axis=0)

    residual_ab = np.linalg.norm(
        after_pixels[:, 1:3] - before_median[1:3], axis=1
    )
    dirty_vector = residual_ab > global_baseline['before_threshold_ab']

    from skimage.color import deltaE_ciede2000
    b_arr = before_median.reshape(1, 1, 3).astype(np.float64)
    a_arr = after_median.reshape(1, 1, 3).astype(np.float64)

    result = {
        'residual_dirty_percent_ab': round(100.0 * float(np.mean(dirty_vector)), 2),
        'median_deltaE00_full':      round(float(deltaE_ciede2000(b_arr, a_arr)[0, 0]), 2),
        'median_delta_ab':           round(float(np.linalg.norm(after_median[1:3] - before_median[1:3])), 2),
        'dirty_vector':              dirty_vector,
    }

    # result = st.score_after_against_before(
    #     before_pixels = global_baseline['pixels'] if 'pixels' in global_baseline
    #                     else st.tile_pixels_from_mask(lab, tile_only_mask),  # fallback
    #     after_pixels  = after_pixels,
    #     after_lab_img = lab,
    #     after_mask    = tile_only_mask,
    #     threshold_ab  = global_baseline['before_threshold_ab'],
    # )

    metrics = {
        'Dirty %':     result['residual_dirty_percent_ab'],
        'Median ΔE00': result['median_deltaE00_full'],
        'Median Δab':  result['median_delta_ab'],
    }

    treatment_dir = DEBUG_DIR / treatment_name
    treatment_dir.mkdir(exist_ok=True)
    overlay_path = treatment_dir / f'{tile_id}_treatment_overlay.png'
    _save_treatment_overlay(crop, mask, coral_mask, result['dirty_vector'], overlay_path)
    with open(str(overlay_path), 'rb') as f:
        overlay_b64 = base64.b64encode(f.read()).decode()

    return metrics, overlay_b64


def _generate_treatment_charts(treatment_results):
    """
    3-panel bar chart (Dirty %, ΔE₀₀, Δab) with one bar per treatment.
    Bars are coloured with a qualitative palette; SD shown as error bars;
    per-tile n shown above each bar.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Sort treatments by embedded number
    def _tsort(name):
        m = re.search(r'\d+', name)
        return int(m.group()) if m else 0

    # ONLY FOR THE SPECIAL 4-TREATMENTS CASE
    # CUSTOM_ORDER = [
    #     'Treatment 2 (C)',
    #     'Treatment 9 (C)',
    #     'Treatment 6 (T)',
    #     'Treatment 7 (T)',
    # ]
    # def _tsort(name):
    #     if name in CUSTOM_ORDER:
    #         return CUSTOM_ORDER.index(name)
    #     m = re.search(r'\d+', name)
    #     return 1000 + (int(m.group()) if m else 0)
    # palette = ['#1a1a1a', '#c7c7c7', '#8c564b', '#f49e3f']


    treatments  = sorted(treatment_results.keys(), key=_tsort)
    n_bars      = len(treatments)

    # Qualitative palette — up to 10 distinct colours
    # palette = [
    #     '#1a6bb5','#e07b39','#2ca02c','#d62728','#9467bd',
    #     '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
    # ]
    palette = [
        '#d62728',  # red
        '#1a1a1a',  # black
        '#b39ddb',  # light purple
        '#98df8a',  # light green
        '#ffdd57',  # yellow
        '#8c564b',  # brown
        '#f49e3f',  # orange
        '#1a6bb5',  # blue
        '#c7c7c7',  # light gray
        '#f9b8c9',  # pink
    ]
    colors = [palette[i % len(palette)] for i in range(n_bars)]

    metric_cols = [
        ('Dirty %',     'Dirty %',     'Dirty % sd'),
        ('Median ΔE00', 'Median ΔE₀₀', 'Median ΔE00 sd'),
        ('Median Δab',  'Median Δab',  'Median Δab sd'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(max(15, n_bars * 1.8), 7))
    fig.suptitle(
        'Treatment Comparison — Tile Cleanliness (Coral Excluded)\n'
        'CIELAB Color Space. Mean ± SD across tiles per treatment',
        fontsize=13, fontweight='bold', y=1.02
    )

    for ax, (col_key, col_label, col_sd) in zip(axes, metric_cols):
        means = [treatment_results[t][col_key] for t in treatments]
        sds   = [treatment_results[t][col_sd]  for t in treatments]
        ns    = [treatment_results[t]['n']      for t in treatments]

        finite_vals = [v for v in means if np.isfinite(v)]
        y_max  = max(finite_vals) if finite_vals else 1.0
        y_lim  = max(y_max * 1.45, 1.0)

        ax.bar(range(n_bars), means, yerr=sds, capsize=5,
               color=colors, edgecolor='white', alpha=0.88, linewidth=1.2,
               error_kw={'elinewidth': 1.8, 'ecolor': '#444'}, zorder=2)

        # Mean label inside bar
        for i, (m, sd) in enumerate(zip(means, sds)):
            if np.isfinite(m):
                ax.text(i, 0.5, f'{m:.2f}',
                        ha='center', va='bottom', fontsize=8.5,
                        fontweight='bold', color='white', zorder=10)

        # n= label above error bar
        for i, (m, sd, n) in enumerate(zip(means, sds, ns)):
            if np.isfinite(m):
                ax.text(i, m + sd + y_lim * 0.03,
                        f'n={n}', ha='center', va='bottom',
                        fontsize=7.5, color='#555')

        ax.set_ylim(0, y_lim)
        ax.set_xticks(range(n_bars))
        ax.set_xticklabels(treatments, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(col_label, fontsize=11)
        ax.set_title(col_label, fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.2, axis='y', linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / 'treatment_chart.png'
    fig.savefig(str(chart_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    with open(str(chart_path), 'rb') as f:
        return base64.b64encode(f.read()).decode()


if __name__ == '__main__':
    print("\n  Coral Guard Tile Scorer running at http://localhost:5001\n")
    app.run(debug=False, port=5001)
