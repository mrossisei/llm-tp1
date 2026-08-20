"""Curvas train/val del modelo final: el diagnostico de over/underfitting.

    .venv/bin/python eda/grafico_curvas_entrenamiento.py
        -> graficos/curvas_entrenamiento.png  (seed con mejor val, PR-AUC y loss por epoca)
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    import glob
    mejor = max(glob.glob(str(REPO / 'resultados' / 'feat_ordinal_features_*.json')),
                key=lambda f: json.load(open(f))['val']['pr_auc'])
    d = json.load(open(mejor))
    ep = [h['epoch'] for h in d['historial']]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, met, titulo in ((axes[0], 'pr_auc', 'PR-AUC'), (axes[1], 'loss', 'loss (BCE)')):
        tr = [h['train'][met] for h in d['historial']]
        va = [h['val'][met] for h in d['historial']]
        ax.plot(ep, tr, color='#0E9B7E', lw=1.8, label='train')
        ax.plot(ep, va, color='#7052C9', lw=1.8, label='validación')
        i_best = int(np.argmax([h['val']['pr_auc'] for h in d['historial']]))
        ax.axvline(ep[i_best], color='#D42A63', ls='--', lw=1,
                   label='mejor val (early stopping)' if met == 'pr_auc' else None)
        ax.set_xlabel('época'); ax.set_title(titulo, fontsize=10)
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle(f'Curvas de entrenamiento — modelo final (seed {d["seed"]}): '
                 'el gap train/val es el diagnóstico de overfitting', fontsize=10.5)
    fig.tight_layout()
    out = REPO / 'graficos' / 'curvas_entrenamiento.png'
    fig.savefig(out, dpi=140)
    print(f'-> {out.relative_to(REPO)}')


if __name__ == '__main__':
    main()
