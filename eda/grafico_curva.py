"""Figura de la curva de aprendizaje del modelo final (curva_frac* de la suite).

    .venv/bin/python eda/grafico_curva.py  ->  graficos/curva_aprendizaje.png
"""

import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

TAGS = [('curva_frac25', 25), ('curva_frac50', 50), ('curva_frac75', 75), ('feat_ordinal', 100)]


def main():
    vals = collections.defaultdict(list)
    for path in (REPO / 'resultados').glob('*.json'):
        d = json.loads(path.read_text())
        vals[d['config']['tag']].append(d['test']['pr_auc'])
    xs = [p for _, p in TAGS]
    ms = [np.mean(vals[t]) for t, _ in TAGS]
    ss = [np.std(vals[t]) for t, _ in TAGS]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(xs, ms, yerr=ss, color='#0E9B7E', lw=2, marker='o', ms=6, capsize=3)
    for x, m in zip(xs, ms):
        ax.annotate(f'{m:.3f}', (x, m), textcoords='offset points', xytext=(0, 9),
                    ha='center', fontsize=8.5, color='#1B2530')
    ax.axhline(0.762, color='#D42A63', ls='--', lw=1)
    ax.text(99, 0.7585, 'GBM con el 100% de los datos: 0.762', color='#D42A63',
            fontsize=8, ha='right')
    ax.set_xlabel('% de las queries de train (val y test fijos)')
    ax.set_ylabel('PR-AUC test (media ± desvío, 6 seeds)')
    ax.set_title('Curva de aprendizaje — feat_ordinal (modelo final)', fontsize=10.5)
    ax.set_xticks(xs)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = REPO / 'graficos' / 'curva_aprendizaje.png'
    fig.savefig(out, dpi=140)
    print(f'-> {out.relative_to(REPO)}')


if __name__ == '__main__':
    main()
