"""Calibracion de los modelos entrenados (propuesta #3 de junior_proposals.md).

AUC mide si el modelo ORDENA bien; nunca si los numeros son creibles. Y el BTR de
negocio se define como el promedio de las p(bought) por producto (propuesta 1.2):
si las probabilidades estan corridas, ese promedio hereda el sesgo aunque el
ranking sea perfecto. Este script lo mide sobre checkpoints YA entrenados (cero
reentrenamiento):

  1. Reliability diagram: deciles de p predicha vs tasa real de bought por decil.
  2. ECE (expected calibration error): promedio ponderado de |predicho - observado|.
  3. Brier score (ya se guarda por corrida desde compute_metrics).
  4. Temperature scaling (Guo et al. 2017): un escalar T ajustado en VALIDACION
     (el modelo queda congelado; solo se dividen los logits antes del sigmoide),
     reportado en test. Es la correccion minima si hay descalibracion.

Uso:
    .venv/bin/python eda/calibracion.py pesos/<checkpoint>.pt [...]
    .venv/bin/python eda/calibracion.py            # default: el mejor por val de
                                                   # pac20_feat_h1 + feat_base seed42
Salida: salidas/graficos/calibracion_<nombre>.png + tabla por consola.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from btr.data import prepare  # noqa: E402
from btr.model import load_checkpoint  # noqa: E402
from btr.train import drop_feature_columns  # noqa: E402

N_BINS = 10


def logits_de(ckpt_path):
    """(logits_val, y_val, logits_test, y_test) del checkpoint, con su split exacto."""
    data = json.loads((REPO / 'salidas' / 'resultados' / f'{ckpt_path.stem}.json').read_text())
    cfg = data['config']
    model, _ = load_checkpoint(ckpt_path)
    _, _, splits = prepare(REPO / 'supermarket_products.csv', seed=data['seed'])
    drop = {f.strip() for f in cfg.get('drop_features', '').split(',') if f.strip()}
    splits, _, _ = drop_feature_columns(splits, drop)
    out = []
    with torch.no_grad():
        for split in ('val', 'test'):
            a, b, c, y = splits[split]
            logits = torch.cat([model(a[s:s + 1024], b[s:s + 1024], c[s:s + 1024])[0]
                                for s in range(0, a.shape[0], 1024)])
            if a.dim() == 3:
                mask = c if c.dtype == torch.bool else (c != 0).any(-1)
                logits, y = logits[mask], y[mask]
            out += [logits, y]
    return out


def ece_y_bins(probs, y, n_bins=N_BINS):
    """ECE con bins por cuantiles de p (equal-mass, robusto con probas concentradas)."""
    qs = np.quantile(probs, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -1e-9, 1 + 1e-9
    idx = np.clip(np.searchsorted(qs, probs, side='right') - 1, 0, n_bins - 1)
    pred, obs, peso = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        pred.append(probs[m].mean())
        obs.append(y[m].mean())
        peso.append(m.mean())
    pred, obs, peso = map(np.array, (pred, obs, peso))
    return float((peso * np.abs(pred - obs)).sum()), pred, obs, peso


def ajustar_temperatura(logits_val, y_val):
    """Minimiza la BCE de validacion sobre T (busqueda escalar, modelo congelado)."""
    ts = np.exp(np.linspace(np.log(0.25), np.log(4.0), 61))
    y = torch.as_tensor(y_val, dtype=torch.float32)
    lo = torch.as_tensor(logits_val, dtype=torch.float32)
    nll = [torch.nn.functional.binary_cross_entropy_with_logits(lo / t, y).item() for t in ts]
    return float(ts[int(np.argmin(nll))])


def calibrar(ckpt_path, ax=None):
    lo_va, y_va, lo_te, y_te = (t.numpy() for t in logits_de(ckpt_path))
    sig = lambda z: 1 / (1 + np.exp(-z))
    T = ajustar_temperatura(lo_va, y_va)
    p_antes, p_desp = sig(lo_te), sig(lo_te / T)
    ece_a, pred_a, obs_a, _ = ece_y_bins(p_antes, y_te)
    ece_d, pred_d, obs_d, _ = ece_y_bins(p_desp, y_te)
    brier_a = float(np.mean((p_antes - y_te) ** 2))
    brier_d = float(np.mean((p_desp - y_te) ** 2))
    fila = dict(nombre=ckpt_path.stem, T=T, ece_antes=ece_a, ece_despues=ece_d,
                brier_antes=brier_a, brier_despues=brier_d,
                btr_real=float(y_te.mean()), btr_pred=float(p_antes.mean()),
                btr_pred_T=float(p_desp.mean()))

    if ax is not None:
        ax.plot([0, 1], [0, 1], '--', color='#999', lw=1, label='calibracion perfecta')
        ax.plot(pred_a, obs_a, 'o-', color='#C22B5E', lw=1.5, ms=4,
                label=f'sin corregir (ECE {ece_a:.3f})')
        ax.plot(pred_d, obs_d, 'o-', color='#1C8A76', lw=1.5, ms=4,
                label=f'T = {T:.2f} (ECE {ece_d:.3f})')
        ax.set_xlabel('p(bought) predicha (promedio del decil)')
        ax.set_ylabel('tasa real de bought en el decil')
        ax.set_title(ckpt_path.stem, fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    return fila


def main():
    if len(sys.argv) > 1:
        ckpts = [Path(p) for p in sys.argv[1:]]
    else:
        # mejor seed por val PR de pac20_feat_h1 (el campeon) + feat_base seed 42
        candidatos = sorted((REPO / 'salidas' / 'pesos').glob('pac20_feat_h1_*.pt'))
        mejor = max(candidatos, key=lambda p: json.loads(
            (REPO / 'salidas' / 'resultados' / f'{p.stem}.json').read_text())['val']['pr_auc'])
        ckpts = [mejor, REPO / 'salidas' / 'pesos' / 'feat_base_features_d32_h4_l2_linear_seed42.pt']

    (REPO / 'salidas' / 'graficos').mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, len(ckpts), figsize=(5.4 * len(ckpts), 4.6), squeeze=False)
    filas = [calibrar(c, axes[0][i]) for i, c in enumerate(ckpts)]
    fig.suptitle('Reliability diagram (test) — deciles de probabilidad predicha', fontsize=11)
    fig.tight_layout()
    out = REPO / 'salidas' / 'graficos' / 'calibracion.png'
    fig.savefig(out, dpi=130)
    print(f'grafico -> {out.relative_to(REPO)}\n')

    print(f"{'checkpoint':<52} {'T':>5} {'ECE antes':>10} {'ECE desp':>9} "
          f"{'Brier a':>8} {'Brier d':>8} {'BTR real':>9} {'BTR pred':>9} {'pred/T':>7}")
    for f in filas:
        print(f"{f['nombre']:<52} {f['T']:>5.2f} {f['ece_antes']:>10.4f} {f['ece_despues']:>9.4f} "
              f"{f['brier_antes']:>8.4f} {f['brier_despues']:>8.4f} "
              f"{f['btr_real']:>9.4f} {f['btr_pred']:>9.4f} {f['btr_pred_T']:>7.4f}")


if __name__ == '__main__':
    main()
