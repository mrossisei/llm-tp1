"""Figura del SOM (6ta tanda, sia_som16): que organizo Kohonen y por que no ayudo.

    .venv/bin/python eda/grafico_som.py   ->  graficos/som_btr.png

Carga el SOM 4x4 guardado en el checkpoint de sia_som16 (seed 42), asigna la
celda BMU a cada fila del dataset (numericas z-scoreadas con los estadisticos
de train de ese split) y muestra dos paneles: BTR medio por celda y tamano de
cada celda. La lectura esperada: el SOM organiza precio/peso/dimensiones en
regiones suaves, pero el BTR por celda queda cerca de la tasa base en casi
todas — la senal de compra vive en listing_status (categorica), que el SOM no
ve; por eso la celda BMU como feature extra no aporto (-0.017).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(REPO))
from btr.data import load_dataset  # noqa: E402
from btr.train import asignar_som  # noqa: E402

CKPT = REPO / 'pesos' / 'sia_som16_features_d32_h4_l2_linear_catordinal_som4_seed42.pt'
GRID = 4

def main():
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    som = ckpt['som']
    prep = ckpt['preprocessor']
    assert som is not None and som.shape[0] == GRID * GRID

    df = load_dataset(str(REPO / 'supermarket_products.csv'))
    x_cat, x_num, x_text, y = prep.transform(df)
    celdas = asignar_som(x_num.numpy(), som)

    btr = np.full(GRID * GRID, np.nan)
    n = np.zeros(GRID * GRID, dtype=int)
    for c in range(GRID * GRID):
        m = celdas == c
        n[c] = int(m.sum())
        if n[c]:
            btr[c] = float(y.numpy()[m].mean())
    base = float(y.numpy().mean())

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, mat, titulo, fmt, cmap in [
        (axes[0], btr.reshape(GRID, GRID), f'BTR medio por celda (base {base:.3f})', '{:.3f}', 'viridis'),
        (axes[1], n.reshape(GRID, GRID).astype(float), 'filas por celda', '{:.0f}', 'magma'),
    ]:
        im = ax.imshow(mat, cmap=cmap)
        for i in range(GRID):
            for j in range(GRID):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=9,
                            color='white', path_effects=None)
        ax.set_title(titulo, fontsize=11)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('Kohonen 4×4 sobre las numéricas de train (sia_som16, seed 42) — '
                 'la señal de compra no vive en las numéricas', fontsize=11)
    fig.tight_layout()
    out = REPO / 'graficos' / 'som_btr.png'
    fig.savefig(out, dpi=140)
    print(f'escrito {out.relative_to(REPO)} | BTR por celda: '
          f'min {np.nanmin(btr):.3f} max {np.nanmax(btr):.3f} (base {base:.3f})')

if __name__ == '__main__':
    main()
