"""Uso del modelo entrenado: p(bought) por impresion y BTR por producto.

    python predecir.py                          # usa el mejor checkpoint entregado
    python predecir.py pesos/modelo_final_seed46.pt

Muestra las metricas sobre el test de ese seed y el uso de negocio: el ranking
de productos por BTR estimado (promedio de p(bought) sobre sus impresiones),
que es exactamente "que productos conviene promocionar".
"""

import sys
from pathlib import Path

import torch

from data import load_dataset, split_by_query
from model import cargar
from train import compute_metrics

AQUI = Path(__file__).resolve().parent
CSV = AQUI.parent.parent / 'supermarket_products.csv'


def main():
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI / 'pesos' / 'modelo_final_seed46.pt'
    model, prep = cargar(ckpt)
    seed = torch.load(ckpt, map_location='cpu', weights_only=False)['seed']

    df = load_dataset(CSV)
    _, _, test_df = split_by_query(df, seed=seed)
    x_cat, x_num, y = prep.transform(test_df)
    probs = model.predict_proba(x_cat, x_num)

    m = compute_metrics(y.numpy(), probs.numpy(), 0.0)
    print(f"checkpoint: {ckpt.name} (test del seed {seed}, {len(test_df)} impresiones)")
    print(f"  PR-AUC {m['pr_auc']:.4f} | ROC-AUC {m['roc_auc']:.4f} | "
          f"F1 max {m['f1_best']:.3f} @ umbral {m['thr_f1_best']:.2f} | Brier {m['brier']:.4f}")

    t = test_df.copy()
    t['p'] = probs.numpy()
    btr = (t.groupby('title', sort=False)
            .agg(btr_estimado=('p', 'mean'), btr_real=('bought', 'mean'),
                 impresiones=('p', 'size'))
            .sort_values('btr_estimado', ascending=False))
    print("\ntop 10 productos por BTR estimado (la respuesta de negocio):")
    for titulo, fila in btr.head(10).iterrows():
        print(f"  {fila.btr_estimado:.3f} (real {fila.btr_real:.2f}, "
              f"n={int(fila.impresiones)})  {titulo[:70]}")


if __name__ == '__main__':
    main()
