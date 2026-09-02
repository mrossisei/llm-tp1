"""Entrenamiento del modelo final — reproduce los pesos entregados.

    python train.py                # 6 seeds (42..47), device auto
    python train.py --seeds 1     # una corrida rapida
    python train.py --device cpu  # ~1-2 min por seed en CPU (el modelo es chico)

Protocolo (decidido por experimentacion, ver ../analisis.md):
  - Split por query 70/15/15; PROMEDIO DE 6 SEEDS (lo priorizado por la
    catedra por sobre cross-validation; igualmente validamos con GroupKFold
    5x6: 0.8207 ± 0.0119, consistente).
  - AdamW lr 1e-3, batch 256, early stopping por PR-AUC de VALIDACION con
    paciencia 20 (tope 300 epocas; ninguna corrida lo alcanza).
  - Disciplina: los hiperparametros se eligen mirando validacion; test se
    reporta al final.

Metricas: se calculan SIEMPRE las 16 (dos familias):
  - Sin umbral (las de decision, sugeridas por el enunciado): PR-AUC (principal,
    por el desbalance 13% — el azar da 0.131), ROC-AUC (complementaria, azar
    0.5), log-loss y Brier (calidad de las probabilidades, no solo del orden).
  - Con umbral (informativas; el enunciado aclara que NO hace falta definir
    umbral y nuestro uso de negocio es un ranking): accuracy, balanced
    accuracy, precision/recall/F1/especificidad/MCC @ 0.5, y F1 maximo con su
    umbral optimo (~0.40 — evidencia de que el 0.5 seria arbitrario).
  Overfitting/underfitting: el historial guarda todas las metricas de train y
  val POR EPOCA — las curvas train/val son el diagnostico.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, log_loss, matthews_corrcoef, precision_recall_curve, roc_auc_score,
)

from data import prepare
from model import ModeloBTR, guardar

AQUI = Path(__file__).resolve().parent
CSV = AQUI.parent / 'supermarket_products.csv'
EVAL_BATCH = 1024
TRAIN_EVAL_ROWS = 4000  # submuestra fija de train para las metricas por epoca


def compute_metrics(y_true, probs, loss):
    """Las 16 metricas de clasificacion binaria desbalanceada (ver docstring)."""
    pred = probs >= 0.5
    prec_c, rec_c, thr_c = precision_recall_curve(y_true, probs)
    f1_c = 2 * prec_c * rec_c / np.clip(prec_c + rec_c, 1e-12, None)
    i_best = int(np.nanargmax(f1_c))
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return {
            'loss': float(loss),
            'roc_auc': float(roc_auc_score(y_true, probs)),
            'pr_auc': float(average_precision_score(y_true, probs)),
            'log_loss': float(log_loss(y_true, probs, labels=[0, 1])),
            'brier': float(brier_score_loss(y_true, probs)),
            'accuracy': float(accuracy_score(y_true, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
            'precision': float(tp / (tp + fp)) if tp + fp else 0.0,
            'recall': float(tp / (tp + fn)) if tp + fn else 0.0,
            'f1': float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            'specificity': float(tn / (tn + fp)) if tn + fp else 0.0,
            'mcc': float(matthews_corrcoef(y_true, pred)),
            'f1_best': float(f1_c[i_best]),
            'thr_f1_best': float(thr_c[i_best]) if i_best < len(thr_c) else 1.0,
            'tasa_pred_pos': float(pred.mean()),
            'tasa_real_pos': float(y_true.mean()),
        }


def evaluate(model, split, max_rows=None):
    x_cat, x_num, y = split
    if max_rows is not None and x_cat.shape[0] > max_rows:
        idx = torch.randperm(x_cat.shape[0], generator=torch.Generator().manual_seed(0))[:max_rows]
        idx = idx.to(x_cat.device)
        x_cat, x_num, y = x_cat[idx], x_num[idx], y[idx]
    model.eval()
    logits = []
    with torch.no_grad():
        for s in range(0, x_cat.shape[0], EVAL_BATCH):
            lg, _ = model(x_cat[s:s + EVAL_BATCH], x_num[s:s + EVAL_BATCH])
            logits.append(lg)
        logits = torch.cat(logits)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y).item()
    return compute_metrics(y.cpu().numpy(), torch.sigmoid(logits).cpu().numpy(), loss)


def entrenar_seed(seed, device, epochs=300, batch_size=256, lr=1e-3, patience=20, verbose=True):
    torch.manual_seed(seed)
    prep, splits = prepare(CSV, seed=seed)
    splits = {k: tuple(t.to(device) for t in v) for k, v in splits.items()}
    model = ModeloBTR([t.to(device) for t in prep.cat_tables]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    x_cat, x_num, y = splits['train']
    history, best_pr, best_state, since_best = [], -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(x_cat.shape[0], device=device)
        for s in range(0, len(perm), batch_size):
            idx = perm[s:s + batch_size]
            _, loss = model(x_cat[idx], x_num[idx], y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        tr = evaluate(model, splits['train'], max_rows=TRAIN_EVAL_ROWS)
        va = evaluate(model, splits['val'])
        history.append({'epoch': epoch, 'train': tr, 'val': va})
        if verbose:
            print(f"  epoch {epoch:3d} | loss train {tr['loss']:.4f} val {va['loss']:.4f} | "
                  f"PR-AUC train {tr['pr_auc']:.4f} val {va['pr_auc']:.4f}")
        if va['pr_auc'] > best_pr:
            best_pr, since_best = va['pr_auc'], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break
    model.load_state_dict(best_state)
    return model, prep, history


def main():
    parser = argparse.ArgumentParser(description='Entrena el modelo final (feat_ordinal)')
    parser.add_argument('--seeds', type=int, default=6)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' \
        else args.device
    (AQUI / 'pesos').mkdir(exist_ok=True)

    resultados = []
    for seed in range(42, 42 + args.seeds):
        print(f"=== seed {seed} (split e inicializacion) | device {device} ===")
        model, prep, history = entrenar_seed(seed, device, verbose=not args.quiet)
        prep_cpu_splits = prepare(CSV, seed=seed)[1]
        val_m = evaluate(model.cpu(), prep_cpu_splits['val'])
        test_m = evaluate(model, prep_cpu_splits['test'])
        print(f"  -> {len(history)} epocas | VAL PR-AUC {val_m['pr_auc']:.4f} | "
              f"TEST PR-AUC {test_m['pr_auc']:.4f} ROC-AUC {test_m['roc_auc']:.4f}")
        guardar(AQUI / 'pesos' / f'modelo_final_seed{seed}.pt', model, prep, seed,
                {'val': val_m, 'test': test_m})
        (AQUI / 'pesos' / f'historial_seed{seed}.json').write_text(json.dumps(history))
        resultados.append((test_m['roc_auc'], test_m['pr_auc']))

    rocs, prs = zip(*resultados)
    print(f"\n===== {args.seeds} seed(s) | TEST PR-AUC {np.mean(prs):.4f} ± {np.std(prs):.4f} "
          f"| ROC-AUC {np.mean(rocs):.4f} ± {np.std(rocs):.4f} =====")


if __name__ == '__main__':
    main()
