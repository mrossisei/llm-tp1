"""Deep-ensemble PURO: promediar N inicializaciones del MISMO modelo y split.

A diferencia de eda/ensemble.py (configs distintas), aca los modelos son
identicos (feat_ordinal) y solo cambia la seed de inicializacion (robu_init*):
el ensemble clasico de Lakshminarayanan et al. Para cada split se promedian las
p(bought) de las inits disponibles (la original + las 5 de --init-seed,
deduplicando cuando coinciden) y se compara apareado contra el modelo solo.

    .venv/bin/python eda/deep_ensemble.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from btr.data import prepare  # noqa: E402
from btr.model import load_checkpoint  # noqa: E402

SEEDS = range(42, 48)
INITS = range(43, 48)


def main():
    solos, ens = [], []
    for seed in SEEDS:
        _, _, splits = prepare(REPO / 'supermarket_products.csv', seed=seed)
        x_cat, x_num, x_text, y = splits['test']
        y = y.numpy()
        ckpts = {seed: next((REPO / 'pesos').glob(f'feat_ordinal_features_*seed{seed}.pt'))}
        for i in INITS:
            if i != seed:  # robu_init{seed} seria duplicado exacto de la original
                ckpts[i] = next((REPO / 'pesos').glob(f'robu_init{i}_*seed{seed}.pt'))
        probs = []
        with torch.no_grad():
            for ck in ckpts.values():
                model, _ = load_checkpoint(ck)
                probs.append(model.predict_proba(x_cat, x_num, x_text).numpy())
        solos.append(average_precision_score(y, probs[0]))
        ens.append(average_precision_score(y, np.mean(probs, axis=0)))
        print(f'  split {seed}: solo {solos[-1]:.4f} | ensemble de {len(probs)} inits {ens[-1]:.4f}',
              flush=True)
    solos, ens = np.array(solos), np.array(ens)
    d = ens - solos
    print(f'\nfeat_ordinal solo:            {solos.mean():.4f} ± {solos.std():.3f}')
    print(f'deep-ensemble (mismas inits): {ens.mean():.4f} ± {ens.std():.3f}')
    print(f'Δ apareado: {d.mean():+.4f} ± {d.std():.4f}  (gana {(d > 0).sum()}/{len(d)})')


if __name__ == '__main__':
    main()
