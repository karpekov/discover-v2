#!/usr/bin/env python3
"""
Publication-quality t-SNE visualisation for SCAN clustering results.

Reads pre-computed t-SNE coordinates from tsne_data.csv (produced by
scan_tsne_visualization.py) and renders a two-panel figure:

  Left  – points coloured by ground-truth activity label
  Right – points coloured AND shaped by resident (multi-resident datasets)
          or by SCAN cluster id (single-resident datasets)

If the CSV does not exist yet, the script can optionally invoke
scan_tsne_visualization.py first to generate it.

Usage
-----
# Most common – point at the tsne/ output folder
python src/evals/scan/scan_publication.py \
    --tsne_dir results/scan/kyoto/scan_fl20_20cl_discover_v1/tsne

# Auto-run the visualizer if CSV is missing
python src/evals/scan/scan_publication.py \
    --tsne_dir results/scan/kyoto/scan_fl20_20cl_discover_v1/tsne \
    --model_dir trained_models/kyoto/scan_fl20_20cl_discover_v1

# Override output file name
python src/evals/scan/scan_publication.py \
    --tsne_dir results/scan/kyoto/scan_fl20_20cl_discover_v1/tsne \
    --output  results/scan/kyoto/scan_fl20_20cl_discover_v1/tsne/tsne_publication.pdf
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root & metadata
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Global rcParams – matches the rest of the publication suite
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':            'sans-serif',
    'font.sans-serif':        ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':              11,
    'axes.titlesize':         13,
    'axes.titleweight':       'bold',
    'axes.labelsize':         11,
    'xtick.labelsize':        9,
    'ytick.labelsize':        9,
    'legend.fontsize':        9,
    'legend.framealpha':      0.92,
    'legend.edgecolor':       '#cccccc',
    'figure.titlesize':       14,
    'figure.titleweight':     'bold',
    'axes.spines.top':        False,
    'axes.spines.right':      False,
    'axes.linewidth':         0.8,
    'axes.edgecolor':         '#444444',
    'axes.grid':              True,
    'grid.color':             '#dddddd',
    'grid.linestyle':         '--',
    'grid.linewidth':         0.6,
    'grid.alpha':             0.8,
    'xtick.major.width':      0.8,
    'ytick.major.width':      0.8,
    'xtick.major.size':       4,
    'ytick.major.size':       4,
    'xtick.direction':        'out',
    'ytick.direction':        'out',
    'text.color':             '#333333',
    'axes.labelcolor':        '#333333',
    'xtick.color':            '#333333',
    'ytick.color':            '#333333',
    'savefig.dpi':            300,
    'savefig.bbox':           'tight',
    'savefig.facecolor':      'white',
})

# ---------------------------------------------------------------------------
# Resident styling — chosen to be distinct from kyoto activity palette
# (blues, reds, oranges, greens, pinks are all taken by activity labels)
# ---------------------------------------------------------------------------
RESIDENT_STYLE = {
    'R1':      {'color': '#1a1a1a', 'marker': 'o',  'label': 'Resident 1'},   # near-black / charcoal
    'R2':      {'color': '#e6ab02', 'marker': '^',  'label': 'Resident 2'},   # amber / gold
    'both':    {'color': '#2ca5a5', 'marker': 'P',  'label': 'Both residents'},  # deep teal
    'unknown': {'color': '#bbbbbb', 'marker': 's',  'label': 'Unknown'},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metadata_colors(dataset_name: str) -> dict:
    """Load hex activity colours from casas_metadata.json."""
    meta_path = PROJECT_ROOT / 'metadata' / 'casas_metadata.json'
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    ds = meta.get(dataset_name, {})
    return (
        ds.get('label_color') or
        ds.get('label') or
        ds.get('lable') or
        {}
    )


def infer_dataset_name(tsne_dir: Path) -> str:
    """Best-effort: walk up the path looking for a known dataset name."""
    known = ['milan', 'aruba', 'cairo', 'kyoto', 'tulum']
    for part in tsne_dir.parts:
        if part.lower() in known:
            return part.lower()
    return 'unknown'


def ensure_csv(tsne_dir: Path, model_dir: str | None) -> Path:
    """Return path to tsne_data.csv, running the visualizer if needed."""
    csv = next(tsne_dir.glob('tsne_data*.csv'), None)
    if csv is not None:
        return csv

    if model_dir is None:
        raise FileNotFoundError(
            f"No tsne_data*.csv found in {tsne_dir}.\n"
            "Pass --model_dir to auto-generate it, or run "
            "scan_tsne_visualization.py first."
        )

    print(f"tsne_data.csv not found – running scan_tsne_visualization.py ...")
    script = PROJECT_ROOT / 'src' / 'evals' / 'scan' / 'scan_tsne_visualization.py'
    subprocess.run(
        [sys.executable, str(script), '--model_dir', model_dir],
        check=True
    )
    csv = next(tsne_dir.glob('tsne_data*.csv'), None)
    if csv is None:
        raise FileNotFoundError("scan_tsne_visualization.py ran but no CSV was produced.")
    return csv


def build_gt_palette(labels: np.ndarray, metadata_colors: dict) -> dict:
    """Map each unique label → hex color, using metadata where available."""
    unique = sorted(set(labels))
    cmap = plt.cm.get_cmap('tab20', len(unique))
    palette = {}
    for i, lbl in enumerate(unique):
        palette[lbl] = metadata_colors.get(lbl) or \
                       f'#{int(cmap(i)[0]*255):02x}{int(cmap(i)[1]*255):02x}{int(cmap(i)[2]*255):02x}'
    return palette


def build_cluster_palette(cluster_ids: np.ndarray) -> dict:
    """Assign a distinct color to each cluster id."""
    unique = sorted(set(cluster_ids))
    cmap = plt.cm.get_cmap('tab20', max(len(unique), 1))
    return {c: cmap(i / max(len(unique) - 1, 1)) for i, c in enumerate(unique)}


# ---------------------------------------------------------------------------
# Panel drawers — return (handles, labels) instead of creating legends,
# so build_figure can place both legends centrally between the subplots.
# ---------------------------------------------------------------------------

def draw_gt_panel(ax, df: pd.DataFrame, palette: dict, title: str, is_mr: bool):
    """Left panel – coloured by ground-truth activity."""
    resident_marker = {'R1': 'o', 'R2': '^', 'both': 'P', 'unknown': 's'}

    for label, grp in df.groupby('ground_truth'):
        color = palette.get(label, '#888888')
        nice  = label.replace('_', ' ')

        if is_mr and 'resident' in df.columns:
            first = True
            for res, rgrp in grp.groupby('resident'):
                marker = resident_marker.get(res, 'o')
                ax.scatter(
                    rgrp['tsne_x'], rgrp['tsne_y'],
                    c=color, marker=marker,
                    s=30, alpha=0.50, linewidths=0.0,
                    label=nice if first else '_nolegend_',
                    zorder=3,
                )
                first = False
        else:
            ax.scatter(
                grp['tsne_x'], grp['tsne_y'],
                c=color, marker='o',
                s=30, alpha=0.50, linewidths=0.0,
                label=nice, zorder=3,
            )

    ax.set_title(title)
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    return ax.get_legend_handles_labels()


def draw_resident_panel(ax, df: pd.DataFrame, title: str):
    """Right panel – coloured and shaped by resident."""
    for res in sorted(df['resident'].unique()):
        style = RESIDENT_STYLE.get(res, {'color': '#aaaaaa', 'marker': 'o', 'label': res})
        grp = df[df['resident'] == res]
        ax.scatter(
            grp['tsne_x'], grp['tsne_y'],
            c=style['color'], marker=style['marker'],
            s=30, alpha=0.55, linewidths=0.0,
            label=style['label'], zorder=3,
        )

    ax.set_title(title)
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    return ax.get_legend_handles_labels()


def draw_cluster_panel(ax, df: pd.DataFrame, palette: dict, title: str):
    """Right panel fallback – coloured by cluster id (single-resident)."""
    for cluster_id, grp in df.groupby('cluster'):
        color = palette.get(cluster_id, '#888888')
        ax.scatter(
            grp['tsne_x'], grp['tsne_y'],
            c=[color], marker='o',
            s=30, alpha=0.50, linewidths=0.0,
            label=f'Cluster {cluster_id}', zorder=3,
        )

    ax.set_title(title)
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    return ax.get_legend_handles_labels()


def style_axis(ax, xlim=None, ylim=None):
    """Apply shared cosmetic polish to an axis."""
    ax.set_facecolor('#fafafa')
    ax.tick_params(which='both', length=4, width=0.8)
    ax.xaxis.set_minor_locator(plt.AutoLocator())
    ax.yaxis.set_minor_locator(plt.AutoLocator())
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def build_figure(
    df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    output_path: Path,
    xlim_min: float | None = None,
    ylim_min: float | None = None,
):
    is_mr = 'resident' in df.columns and df['resident'].nunique() > 1

    metadata_colors = load_metadata_colors(dataset_name)
    gt_palette      = build_gt_palette(df['ground_truth'].values, metadata_colors)
    cluster_palette = build_cluster_palette(df['cluster'].values)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))

    # Symmetric outer margins + explicit wspace so the gap centre lands at
    # exactly x = (left + right) / 2 = 0.5 in figure coordinates.
    # With left=0.06, right=0.94, wspace=0.50:
    #   subplot_w = (0.88) / (2 + 0.50) = 0.352
    #   gap centre = 0.06 + 0.352 + 0.352*0.50/2 = 0.500  ✓
    fig.subplots_adjust(left=0.06, right=0.94, top=0.91, bottom=0.09, wspace=0.50)

    fig.suptitle(
        f't-SNE Embeddings  ·  {dataset_name.capitalize()}  ·  {model_name}',
        y=0.98,
    )

    # Shared axis limits — set to None for auto, or override per call-site
    xlim = (xlim_min, None) if xlim_min is not None else None
    ylim = (ylim_min, None) if ylim_min is not None else None

    # --- Panel A ---
    gt_handles, gt_labels = draw_gt_panel(
        axes[0], df, gt_palette,
        title='(a)  CASAS annotations',
        is_mr=is_mr,
    )
    style_axis(axes[0], xlim=xlim, ylim=ylim)

    # --- Panel B ---
    if is_mr:
        b_handles, b_labels = draw_resident_panel(
            axes[1], df,
            title='(b)  Resident ID',
        )
        b_title = 'Resident ID'
    else:
        b_handles, b_labels = draw_cluster_panel(
            axes[1], df, cluster_palette,
            title='(b)  SCAN cluster',
        )
        b_title = 'SCAN cluster'
    style_axis(axes[1], xlim=xlim, ylim=ylim)

    # -----------------------------------------------------------------------
    # Two figure-level legends, both centred in the gap (x = 0.5).
    # Upper legend (activity colors): top anchor grows downward.
    # Lower legend (resident / cluster): bottom anchor grows upward.
    # They naturally sit one above the other without overlap.
    # -----------------------------------------------------------------------
    _leg_kw = dict(
        bbox_transform=fig.transFigure,
        borderaxespad=0,
        handletextpad=0.6,
        markerscale=1.6,
        fontsize=11,
        framealpha=0.93,
        edgecolor='#cccccc',
    )

    fig.legend(
        gt_handles, gt_labels,
        bbox_to_anchor=(0.5, 0.93), loc='upper center',
        **_leg_kw,
    )
    fig.legend(
        b_handles, b_labels,
        bbox_to_anchor=(0.5, 0.07), loc='lower center',
        title=b_title, title_fontsize=11,
        **_leg_kw,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Publication-quality t-SNE figure from pre-computed CSV data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--tsne_dir', type=str, required=True,
        help='Path to the tsne/ output folder containing tsne_data.csv',
    )
    parser.add_argument(
        '--model_dir', type=str, default=None,
        help='SCAN model dir – used to auto-generate tsne_data.csv if missing',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file path (default: <tsne_dir>/tsne_publication.pdf)',
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Dataset name override for metadata colour lookup (e.g. kyoto)',
    )
    parser.add_argument(
        '--suffix', type=str, default='',
        help='Label suffix used when generating tsne_data (e.g. _all_labels)',
    )
    parser.add_argument(
        '--xlim_min', type=float, default=None,
        help='Minimum value for t-SNE dimension 1 axis (e.g. -70)',
    )
    parser.add_argument(
        '--ylim_min', type=float, default=None,
        help='Minimum value for t-SNE dimension 2 axis (e.g. -70)',
    )
    args = parser.parse_args()

    tsne_dir = Path(args.tsne_dir)

    # Locate / generate the CSV
    if args.suffix:
        csv_path = tsne_dir / f'tsne_data{args.suffix}.csv'
        if not csv_path.exists():
            csv_path = ensure_csv(tsne_dir, args.model_dir)
    else:
        csv_path = ensure_csv(tsne_dir, args.model_dir)

    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    # Infer dataset and model names from paths
    dataset_name = args.dataset or infer_dataset_name(tsne_dir)
    model_name   = tsne_dir.parent.name  # folder above tsne/

    # Default output: same folder, PDF for vector quality
    output_path = Path(args.output) if args.output else \
                  tsne_dir / f'tsne_publication{args.suffix}.pdf'

    print(f"Dataset : {dataset_name}")
    print(f"Model   : {model_name}")
    print(f"Rows    : {len(df)}")
    print(f"Columns : {list(df.columns)}")
    print(f"Output  : {output_path}")

    build_figure(df, dataset_name, model_name, output_path,
                 xlim_min=args.xlim_min, ylim_min=args.ylim_min)

    # Also save a PNG for quick preview
    png_path = output_path.with_suffix('.png')
    df_tmp = pd.read_csv(csv_path)  # re-read to avoid state issues
    build_figure(df_tmp, dataset_name, model_name, png_path,
                 xlim_min=args.xlim_min, ylim_min=args.ylim_min)
    print(f"Saved:  {png_path}")


if __name__ == '__main__':
    main()
