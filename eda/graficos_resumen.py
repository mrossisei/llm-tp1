"""Grafico resumen de la suite: PR-AUC de test por configuracion (media +- desvio).

    .venv/bin/python eda/graficos_resumen.py   ->  graficos/resumen_prauc.png

Barras horizontales, protocolo base (paciencia 8) y pac20 apareados por color.
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


def main():
    grupos = collections.defaultdict(list)
    for path in sorted((REPO / 'resultados').glob('*.json')):
        d = json.loads(path.read_text())
        grupos[d['config']['tag'] or d['nombre']].append(d['test']['pr_auc'])

    filas = sorted(((np.mean(v), np.std(v), k) for k, v in grupos.items()), reverse=True)
    fig, ax = plt.subplots(figsize=(8.5, 0.28 * len(filas) + 1.2))
    ys = np.arange(len(filas))[::-1]
    for y, (m, s, k) in zip(ys, filas):
        pac = k.startswith('pac20_')
        color = '#1C8A76' if pac else '#5B6B85'
        if 'intrinseco' in k or 'causal' in k.replace('causal_last', ''):
            color = '#C22B5E' if 'causal' in k else '#B58117'
        ax.barh(y, m, xerr=s, color=color, height=0.72, error_kw=dict(lw=0.8))
        ax.text(0.005, y, k, va='center', ha='left', fontsize=6.5, color='white',
                fontweight='bold')
    ax.axvline(0.762, color='#C22B5E', ls='--', lw=1)
    ax.text(0.762, len(filas) + 0.2, 'GBM 0.762', color='#C22B5E', fontsize=7, ha='center')
    ax.axvline(0.131, color='#999', ls=':', lw=1)
    ax.text(0.131, len(filas) + 0.2, 'azar 0.131', color='#777', fontsize=7, ha='center')
    ax.set_yticks([])
    ax.set_xlabel('PR-AUC test (media ± desvío, 6 seeds)')
    ax.set_title('Suite completa — verde: paciencia 20 · gris: paciencia 8 · '
                 'ámbar: intrínseco · rojo: causal degenerado', fontsize=8.5)
    ax.set_xlim(0, 0.92)
    fig.tight_layout()
    out = REPO / 'graficos' / 'resumen_prauc.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f'-> {out.relative_to(REPO)}')


if __name__ == '__main__':
    main()
