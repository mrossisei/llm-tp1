"""Ensemble de configuraciones sobre checkpoints YA entrenados (cero GPU).

Idea clasica de "ultimo empujon": promediar las probabilidades de varios modelos
diversos. Es valido apareado porque los checkpoints de un MISMO seed comparten
el split (el seed controla split e inicializacion a la vez): para cada seed se
promedian las p(bought) de las K configs sobre su test y se compara contra el
mejor modelo solo, seed a seed.

Disciplina: la COMPOSICION del ensemble (que configs entran) se elige por
VALIDACION; test solo se reporta para la composicion elegida.

    .venv/bin/python eda/ensemble.py
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from btr.data import prepare  # noqa: E402
from btr.model import load_checkpoint  # noqa: E402

# las 4 mejores configuraciones por val (todas formulation=features, sin drops)
TAGS = ['feat_ordinal', 'pac20_feat_h1', 'feat_target', 'camp_d64h1l4']
SEEDS = range(42, 48)
MEJOR_SOLO = 'feat_ordinal'


def probs_de(tag, seed, splits):
    ckpt = next((REPO / 'pesos').glob(f'{tag}_features_*seed{seed}.pt'))
    model, _ = load_checkpoint(ckpt)
    out = {}
    with torch.no_grad():
        for split in ('val', 'test'):
            x_cat, x_num, x_text, y = splits[split]
            out[split] = (model.predict_proba(x_cat, x_num, x_text).numpy(), y.numpy())
    return out


def main():
    por_seed = {}
    for seed in SEEDS:
        _, _, splits = prepare(REPO / 'supermarket_products.csv', seed=seed)
        por_seed[seed] = {t: probs_de(t, seed, splits) for t in TAGS}
        print(f'  seed {seed}: probs de {len(TAGS)} configs listas', flush=True)

    def pr(comb, split):
        vals = []
        for seed in SEEDS:
            ps = np.mean([por_seed[seed][t][split][0] for t in comb], axis=0)
            y = por_seed[seed][comb[0]][split][1]
            vals.append(average_precision_score(y, ps))
        return np.array(vals)

    # eleccion de composicion por VAL
    print('\ncomposicion (elegida por val):')
    candidatas = [c for k in range(1, len(TAGS) + 1) for c in combinations(TAGS, k)]
    val_scores = {c: pr(c, 'val').mean() for c in candidatas}
    for c, v in sorted(val_scores.items(), key=lambda x: -x[1])[:6]:
        print(f'  val {v:.4f}  {" + ".join(c)}')
    ganadora = max(val_scores, key=val_scores.get)

    # reporte de test SOLO para la ganadora, apareado contra el mejor modelo solo
    te_ens = pr(ganadora, 'test')
    te_solo = pr((MEJOR_SOLO,), 'test')
    delta = te_ens - te_solo
    print(f'\nensemble elegido: {" + ".join(ganadora)}')
    print(f'  test PR-AUC ensemble:    {te_ens.mean():.4f} ± {te_ens.std():.3f}')
    print(f'  test PR-AUC {MEJOR_SOLO}: {te_solo.mean():.4f} ± {te_solo.std():.3f}')
    print(f'  Δ apareado: {delta.mean():+.4f} ± {delta.std():.4f}  '
          f'(gana {(delta > 0).sum()}/{len(delta)} seeds)')


if __name__ == '__main__':
    main()
