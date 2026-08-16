"""Completa las metricas de corridas viejas SIN reentrenar, desde sus checkpoints.

Las 285 corridas de la suite en GPU (16/08) se ejecutaron con el codigo previo a
compute_metrics: sus JSON guardan solo loss/roc_auc/pr_auc. Este script recarga
cada checkpoint de pesos/, reconstruye los splits exactamente como run() (misma
seed, mismos flags) y recalcula el dict COMPLETO de metricas de val y test,
actualizando el JSON in place (agrega 'metricas_recalculadas': true).

El historial por epoca queda como estaba (loss/roc/pr): para curvas alcanza, y
recalcularlo si requeriria reentrenar.

    .venv/bin/python eda/recalcula_metricas.py            # repara lo que falte
    .venv/bin/python eda/recalcula_metricas.py --dry-run  # solo lista
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from btr.data import CAT_FEATURES, NUM_FEATURES, prepare, prepare_listwise  # noqa: E402
from btr.model import load_checkpoint  # noqa: E402
from btr.train import drop_feature_columns, evaluate  # noqa: E402

_CACHE = {}


def splits_de(config):
    """Reconstruye los splits de la corrida (cacheado: prepare domina el costo)."""
    listwise = config['arch'] == 'listwise'
    key = (listwise, config['seed_'], config.get('max_text_len', 256),
           bool(config.get('strip_status')))
    if key not in _CACHE:
        if listwise:
            _, _, splits = prepare_listwise(REPO / 'supermarket_products.csv', seed=config['seed_'])
        else:
            _, _, splits = prepare(REPO / 'supermarket_products.csv', seed=config['seed_'],
                                   max_text_len=config.get('max_text_len', 256),
                                   strip_status=bool(config.get('strip_status')))
        _CACHE[key] = splits
        if len(_CACHE) > 8:  # los splits de texto pesan; no acumular de mas
            _CACHE.pop(next(iter(_CACHE)))
    splits = _CACHE[key]
    drop = {f.strip() for f in config.get('drop_features', '').split(',') if f.strip()}
    splits, _, _ = drop_feature_columns(splits, drop, listwise)
    return splits


def reparar(path, dry):
    data = json.loads(path.read_text())
    if 'f1_best' in data['test']:
        return 'ok'
    ckpt = REPO / 'pesos' / f"{data['nombre']}.pt"
    if not ckpt.exists():
        return 'sin-checkpoint'
    if dry:
        return 'pendiente'
    config = dict(data['config'], seed_=data['seed'])
    model, _ = load_checkpoint(ckpt)
    splits = splits_de(config)
    data['val'] = evaluate(model, splits['val'])
    data['test'] = evaluate(model, splits['test'])
    data['metricas_recalculadas'] = True
    path.write_text(json.dumps(data, indent=2))
    return 'reparado'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    conteo = {}
    paths = sorted((REPO / 'resultados').glob('*.json'))
    for i, path in enumerate(paths, 1):
        estado = reparar(path, args.dry_run)
        conteo[estado] = conteo.get(estado, 0) + 1
        if estado == 'reparado' and conteo[estado] % 20 == 0:
            print(f"[{i}/{len(paths)}] {conteo}", flush=True)
    print('final:', conteo)
    if conteo.get('sin-checkpoint'):
        print('AVISO: corridas sin checkpoint quedaron con las metricas viejas')


if __name__ == '__main__':
    main()
