"""Importancia por permutacion del modelo final (cero reentrenamiento).

Complemento OUTCOME-based de los mapas de atencion: la atencion muestra a quien
MIRA el modelo; esto mide cuanto EMPEORA (ΔPR-AUC de test) si se destruye cada
feature permutando su columna. Responde por adelantado la objecion clasica
"attention is not explanation": aca las dos evidencias se comparan.

    .venv/bin/python eda/importancia.py    ->  tabla + graficos/importancia.png
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from btr.data import CAT_FEATURES, NUM_FEATURES, prepare  # noqa: E402
from btr.model import load_checkpoint  # noqa: E402

TAG = 'feat_ordinal'
SEEDS = range(42, 48)


def main():
    feats = CAT_FEATURES + NUM_FEATURES
    deltas = {f: [] for f in feats}
    for seed in SEEDS:
        ckpt = next((REPO / 'pesos').glob(f'{TAG}_*seed{seed}.pt'))
        model, _ = load_checkpoint(ckpt)
        _, _, splits = prepare(REPO / 'supermarket_products.csv', seed=seed)
        x_cat, x_num, x_text, y = splits['test']
        y = y.numpy()
        base = average_precision_score(y, model.predict_proba(x_cat, x_num, x_text).numpy())
        gen = torch.Generator().manual_seed(0)
        perm = torch.randperm(x_cat.shape[0], generator=gen)
        for i, f in enumerate(feats):
            xc, xn = x_cat.clone(), x_num.clone()
            if f in CAT_FEATURES:
                xc[:, i] = xc[perm, i]
            else:
                j = NUM_FEATURES.index(f)
                xn[:, j] = xn[perm, j]
            pr = average_precision_score(y, model.predict_proba(xc, xn, x_text).numpy())
            deltas[f].append(base - pr)
        print(f'  seed {seed}: base {base:.4f}', flush=True)

    filas = sorted(((np.mean(v), np.std(v), f) for f, v in deltas.items()), reverse=True)
    print(f'\nimportancia por permutacion de {TAG} (caida de PR-AUC test, media 6 seeds):')
    for m, s, f in filas:
        print(f'  {f:<20} {m:+.4f} ± {s:.4f}')

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ys = np.arange(len(filas))[::-1]
    ax.barh(ys, [m for m, _, _ in filas], xerr=[s for _, s, _ in filas],
            color='#0E9B7E', height=0.7, error_kw=dict(lw=0.8))
    ax.set_yticks(ys, [f for _, _, f in filas], fontsize=8)
    ax.set_xlabel('caída de PR-AUC test al permutar la columna (media ± desvío, 6 seeds)')
    ax.set_title(f'Importancia por permutación — {TAG} (modelo final)', fontsize=10)
    ax.axvline(0, color='#999', lw=0.8)
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    out = REPO / 'graficos' / 'importancia.png'
    fig.savefig(out, dpi=140)
    print(f'\ngrafico -> {out.relative_to(REPO)}')


if __name__ == '__main__':
    main()
