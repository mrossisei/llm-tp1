"""Figura EDA para abrir la presentacion: matriz de asociacion + PCA 2D.

    .venv/bin/python eda/grafico_correlacion.py  ->  graficos/matriz_asociacion.png

El dataset es mixto (categoricas + numericas), asi que "matriz de correlacion"
a secas no existe: usamos la matriz de ASOCIACION estandar para tipos mixtos,
todo en [0, 1]:
  - numerica vs numerica:     |Spearman|            (monotona, robusta a colas)
  - categorica vs categorica: V de Cramer           (chi2 normalizado)
  - categorica vs numerica:   razon de correlacion  (eta: var entre-grupos / total)
`bought` (el target) entra como categorica binaria; con una numerica, eta
coincide con la |correlacion punto-biserial|.

Panel derecho: PCA 2D de la matriz cruda (one-hot | numericas z-scoreadas),
coloreado por bought — la contracara visual de sia_pca_mlp (6a tanda): las dos
clases quedan mezcladas porque la senal (el tier del status) casi no aporta
varianza. Reconstruir varianza != predecir.

Incluye tambien las columnas DESCARTADAS en la v1 (volume, pkg_qty, hour...):
la matriz muestra la redundancia que motivo el descarte (propuesta 2.6),
verificado despues con feat_extras/feat_tiempo.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from btr.data import load_dataset  # noqa: E402

TEAL, VIOLETA = '#0E9B7E', '#7052C9'   # par categorico validado (CVD ok)
CMAP = LinearSegmentedColormap.from_list('teal_seq', ['#F4FAF8', '#0E9B7E', '#073D31'])

CATS = ['bought', 'listing_status', 'category', 'brand', 'storage_type',
        'unit_of_measure', 'country_of_origin', 'allergens', 'pkg_unit', 'dow']
NUMS = ['price', 'price_rel', 'filter_price_min', 'filter_price_max',
        'net_weight_oz', 'nutrition_score', 'volume_in3', 'pkg_qty',
        'n_ingredients', 'hour']
DESCARTADAS = {'pkg_unit', 'dow', 'volume_in3', 'pkg_qty', 'n_ingredients', 'hour'}


def cramers_v(a, b):
    import pandas as pd
    ct = pd.crosstab(a, b).to_numpy().astype(float)
    n = ct.sum()
    esperado = np.outer(ct.sum(1), ct.sum(0)) / n
    chi2 = ((ct - esperado) ** 2 / np.where(esperado == 0, 1, esperado)).sum()
    k = min(ct.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def eta(cat, num):
    num = np.asarray(num, dtype=float)
    total = ((num - num.mean()) ** 2).sum()
    if total == 0:
        return 0.0
    entre = sum(len(g) * (g.mean() - num.mean()) ** 2
                for _, g in _agrupar(cat, num))
    return float(np.sqrt(entre / total))


def _agrupar(cat, num):
    import pandas as pd
    return pd.Series(num).groupby(pd.Series(np.asarray(cat)), observed=True)


def matriz_asociacion(df):
    cols = CATS + NUMS
    n = len(cols)
    m = np.eye(n)
    spearman = df[NUMS].corr('spearman').abs().to_numpy()
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            if a in CATS and b in CATS:
                v = cramers_v(df[a], df[b])
            elif a in NUMS and b in NUMS:
                v = spearman[NUMS.index(a), NUMS.index(b)]
            else:
                c, x = (a, b) if a in CATS else (b, a)
                v = eta(df[c], df[x])
            m[i, j] = m[j, i] = v
    return cols, m


def pca_2d(df):
    import pandas as pd
    x = pd.get_dummies(df[[c for c in CATS if c != 'bought']].astype(str)).to_numpy(float)
    z = df[NUMS].to_numpy(float)
    z = np.nan_to_num((z - z.mean(0)) / np.where(z.std(0) == 0, 1, z.std(0)))
    todo = np.hstack([x - x.mean(0), z])
    _, _, vt = np.linalg.svd(todo, full_matrices=False)
    proy = todo @ vt[:2].T
    var = np.var(todo @ vt.T, axis=0)
    return proy, var[:2] / var.sum()


def main():
    df = load_dataset(REPO / 'supermarket_products.csv')
    df['bought'] = df['bought'].astype(str)
    cols, m = matriz_asociacion(df)

    fig, (ax, axp) = plt.subplots(1, 2, figsize=(14.5, 6.6),
                                  gridspec_kw={'width_ratios': [1.25, 1]})
    im = ax.imshow(m, cmap=CMAP, vmin=0, vmax=1)
    etiquetas = [(c + ' †') if c in DESCARTADAS else c for c in cols]
    etiquetas[0] = 'bought (target)'
    ax.set_xticks(range(len(cols)), etiquetas, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(cols)), etiquetas, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i == j:
                continue
            if m[i, j] >= 0.30 or 0 in (i, j):
                ax.text(j, i, f'{m[i, j]:.2f}', ha='center', va='center',
                        fontsize=6.2,
                        color='white' if m[i, j] > 0.55 else '#333333')
    ax.set_title('Matriz de asociación (Spearman / Cramér V / η, todo en [0,1])\n'
                 '† = columnas descartadas en la v1 por redundancia', fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label='fuerza de asociación')

    proy, var2 = pca_2d(df)
    compra = df['bought'] == 'True'
    axp.scatter(*proy[~compra].T, s=5, c=VIOLETA, alpha=0.25, lw=0,
                label=f'no comprado ({(~compra).sum():,})')
    axp.scatter(*proy[compra].T, s=8, c=TEAL, alpha=0.6, lw=0,
                label=f'comprado ({compra.sum():,})')
    axp.set_xlabel(f'PC1 ({var2[0]:.0%} de la varianza)')
    axp.set_ylabel(f'PC2 ({var2[1]:.0%})')
    axp.set_title('PCA 2D de la matriz cruda (one-hot | numéricas):\n'
                  'las clases NO se separan — la señal no vive en la varianza',
                  fontsize=10)
    axp.legend(loc='upper right', fontsize=9, frameon=False, markerscale=2)
    axp.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    out = REPO / 'graficos' / 'matriz_asociacion.png'
    fig.savefig(out, dpi=140)
    print(f'escrito {out.relative_to(REPO)}')

    orden = np.argsort(-m[0])
    print('\nasociación con bought (target):')
    for k in orden:
        if cols[k] != 'bought':
            print(f'  {cols[k]:20s} {m[0, k]:.3f}')


if __name__ == '__main__':
    main()
