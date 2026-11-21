"""
2x2 grid of the re-rendered floor plans (metadata/floor_plans_augmented/{milan,aruba,cairo,orange}.png) for the
appendix "Household Layouts" figure.

    python -m tist_revision.pipeline.house_layouts

Output: TEMP_DIR/house_layouts/all_house_layouts.{pdf,png}
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .mapping_eval import PROJECT_DIR, TEMP_DIR
from .report_final import CITIES, HOUSE_NAMES

PLANS = PROJECT_DIR.parent / 'metadata' / 'floor_plans_augmented'
OUT = TEMP_DIR / 'house_layouts'


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 12.6), gridspec_kw={'wspace': 0.04, 'hspace': 0.08})
    fig.suptitle('Re-rendered and Augmented Floor Plans', fontsize=17, fontweight='bold', y=0.925)
    for i, (city, ax) in enumerate(zip(CITIES, axes.flat)):
        ax.imshow(plt.imread(PLANS / f'{city}.png'), interpolation='lanczos')
        ax.set_title(f'({"abcd"[i]}) {HOUSE_NAMES[city]}', fontsize=14)
        ax.axis('off')
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'all_house_layouts.{ext}', bbox_inches='tight', dpi=250)
    print(f'Saved {OUT}/all_house_layouts.pdf')


if __name__ == '__main__':
    main()
