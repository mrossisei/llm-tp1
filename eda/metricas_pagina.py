"""Metricas POR PAGINA del modelo final (cero reentrenamiento).

El uso de negocio es rankear dentro de la pagina de resultados: ¿el modelo pone
arriba el producto que efectivamente se compro? Sobre las queries de test con
>= 2 productos y >= 1 compra:

  top-1:  ¿el producto con mayor p(bought) es uno comprado?
  MRR:    1 / posicion del primer comprado al ordenar por p
  NDCG:   ganancias binarias, normalizado por el orden ideal
  azar:   esperanza del top-1 aleatorio = compras/productos por pagina

    .venv/bin/python eda/metricas_pagina.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from btr.data import load_dataset, split_by_query  # noqa: E402
from btr.model import load_checkpoint  # noqa: E402

TAG = 'feat_ordinal'
SEEDS = range(42, 48)


def ndcg(y_orden):
    dcg = sum(y / np.log2(i + 2) for i, y in enumerate(y_orden))
    ideal = sorted(y_orden, reverse=True)
    idcg = sum(y / np.log2(i + 2) for i, y in enumerate(ideal))
    return dcg / idcg if idcg > 0 else np.nan


def main():
    df = load_dataset(REPO / 'supermarket_products.csv')
    res = {k: [] for k in ('top1', 'mrr', 'ndcg', 'azar', 'paginas')}
    for seed in SEEDS:
        _, _, test_df = split_by_query(df, seed=seed)
        ckpt = next((REPO / 'pesos').glob(f'{TAG}_*seed{seed}.pt'))
        model, prep = load_checkpoint(ckpt)
        x_cat, x_num, x_text, y = prep.transform(test_df)
        with torch.no_grad():
            probs = model.predict_proba(x_cat, x_num, x_text).numpy()
        t = test_df.copy()
        t['p'] = probs
        top1, mrr, ndcgs, azar = [], [], [], []
        for _, g in t.groupby('query_id', sort=False):
            yq = g['bought'].to_numpy().astype(float)
            if len(g) < 2 or yq.sum() == 0:
                continue
            orden = np.argsort(-g['p'].to_numpy())
            y_orden = yq[orden]
            top1.append(y_orden[0])
            mrr.append(1.0 / (np.argmax(y_orden > 0) + 1))
            ndcgs.append(ndcg(y_orden))
            azar.append(yq.mean())
        res['top1'].append(np.mean(top1))
        res['mrr'].append(np.mean(mrr))
        res['ndcg'].append(np.mean(ndcgs))
        res['azar'].append(np.mean(azar))
        res['paginas'].append(len(top1))
        print(f'  seed {seed}: {len(top1)} paginas evaluables', flush=True)

    print(f'\nmetricas por pagina de {TAG} (test, media ± desvio sobre 6 seeds;'
          f' ~{np.mean(res["paginas"]):.0f} paginas con >=2 productos y >=1 compra):')
    for k, label in (('top1', 'top-1 (el mas rankeado fue comprado)'),
                     ('mrr', 'MRR (posicion del primer comprado)'),
                     ('ndcg', 'NDCG de la pagina'),
                     ('azar', 'azar esperado para top-1')):
        v = np.array(res[k])
        print(f'  {label:<40} {v.mean():.3f} ± {v.std():.3f}')


if __name__ == '__main__':
    main()
