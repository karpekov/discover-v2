"""
Two-pane t-SNE of the DISCOVER clustering features (TF s1 + ToD, training windows): left colored by TDOST (L2)
activity class, right by SCAN cluster (K=20, balanced ToD, seed 1; opacity = cluster confidence). Same layout as the
v1 appendix panels `discover-v1/figures/tsne_label_and_cluster_{milan,aruba,cairo}.pdf`.

    python -m tist_revision.pipeline.tsne_panels --city orange

Output: TEMP_DIR/tsne_panels/tsne_label_and_cluster_<city>.{pdf,png}
"""
import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .mapping_eval import ARTIFACTS_DIR, TEMP_DIR, load_protocol
from .report_final import HOUSE_NAMES
from .section5_figures import tsne_coords

OUT = TEMP_DIR / 'tsne_panels'
L2_COLORS = {
    'Sleep': '#4c72b0', 'Bed_to_Toilet': '#8da0cb', 'Bathing': '#3182bd', 'Personal_Hygiene': '#6baed6',
    'Cook': '#e6782b', 'Eat': '#f2b26b', 'Work': '#e377c2', 'Relax': '#c44e52', 'Take_Medicine': '#e8d44d',
    'Leave_Home': '#55a868', 'Enter_Home': '#8fd18f', 'Other': '#4d4d4d', 'No_Activity': '#c3c8d0',
}


def panel(city):
    idx, Z = tsne_coords(city)
    _, pool, _ = load_protocol(city, level='l2')
    labels = pool['label'].to_numpy()[idx]
    a = np.load(ARTIFACTS_DIR / 'assignments' / f'{city}__tf_s1+tod__scan_k20_todwbal.npz')
    cluster, conf = a['pool_cluster'][idx], a['pool_conf'][idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={'wspace': 0.12})
    fig.suptitle(f'{HOUSE_NAMES[city].upper()}: tSNE Projection of SCAN Embeddings', fontsize=15, fontweight='bold')
    classes = [c for c in L2_COLORS if c in set(labels)]
    for c in classes:
        m = labels == c
        axes[0].scatter(Z[m, 0], Z[m, 1], s=5, c=L2_COLORS[c], alpha=0.55, linewidths=0, label=c, rasterized=True)
    axes[0].legend(title='Activity (TDOST classes)', loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4,
                   fontsize=8, title_fontsize=9,
                   markerscale=2.5, frameon=True, facecolor='#f0f0f0', edgecolor='none')
    cmap = plt.get_cmap('tab20')
    rgba = cmap(cluster % 20)
    rgba[:, 3] = np.clip(0.15 + 0.85 * (conf - conf.min()) / max(conf.max() - conf.min(), 1e-9), 0.15, 1.0)
    axes[1].scatter(Z[:, 0], Z[:, 1], s=5, c=rgba, linewidths=0, rasterized=True)
    handles = [Line2D([], [], marker='o', ls='', color=cmap(k), markersize=5, label=str(k)) for k in range(20)]
    axes[1].legend(handles=handles, title='Cluster', loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=7,
                   title_fontsize=8, frameon=True, facecolor='#f0f0f0', edgecolor='none')
    for ax, t in zip(axes, ('Activity label', 'Cluster ID')):
        ax.set_title('Colored by: ' + r'$\bf{' + t.replace(' ', r'\ ') + '}$', fontsize=12)
        ax.set_xlabel('dim1', fontsize=8)
        ax.set_ylabel('dim2', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(color='#e5e5e5', lw=0.6)
        ax.set_axisbelow(True)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'tsne_label_and_cluster_{city}.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved {OUT}/tsne_label_and_cluster_{city}.pdf ({len(idx)} windows, {len(classes)} classes)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--city', default='orange')
    panel(ap.parse_args().city)


if __name__ == '__main__':
    main()
