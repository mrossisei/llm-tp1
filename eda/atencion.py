"""Mapas de atencion de un checkpoint tabular: ¿a que atiende cada feature?

Con ~14 tokens la matriz de atencion se puede MIRAR — es el analisis de
interpretabilidad natural de la formulacion features-como-tokens: ¿el CLS
concentra su atencion en listing_status (la senal dominante)? ¿price_rel
atiende a los limites del filtro (la senal relacional)?

    .venv/bin/python eda/atencion.py [pesos/<ckpt>.pt ...]
    # default: el mejor por val de pac20_feat_h1 (1 cabeza -> un mapa por capa)

Salida: graficos/atencion_<nombre>.png (promedio sobre todo el test) + top-5
de la fila CLS por consola. Requiere checkpoints de formulation='features'.
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from btr.data import prepare  # noqa: E402
from btr.model import load_checkpoint  # noqa: E402
from btr.train import drop_feature_columns  # noqa: E402

# secuencial de UN hue (dataviz): claro -> oscuro sobre el teal del proyecto
CMAP = LinearSegmentedColormap.from_list('teal', ['#F5FAF8', '#9AD6C6', '#2E9B82', '#0A4A3E'])

ETIQUETAS_CORTAS = {'listing_status': 'status', 'category': 'categ', 'brand': 'marca',
                    'storage_type': 'almac', 'unit_of_measure': 'unidad',
                    'country_of_origin': 'origen', 'allergens': 'alérg',
                    'price_rel': 'p_rel', 'filter_price_min': 'f_min',
                    'filter_price_max': 'f_max', 'net_weight_oz': 'peso',
                    'nutrition_score': 'nutri'}


def mapas_de(ckpt_path):
    """Matrices de atencion (capa, cabeza, T, T) promediadas sobre el test."""
    data = json.loads((REPO / 'resultados' / f'{ckpt_path.stem}.json').read_text())
    cfg = data['config']
    if cfg['arch'] != 'transformer' or cfg.get('formulation') != 'features':
        raise SystemExit(f'{ckpt_path.stem}: los mapas legibles son de formulation=features')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model, _ = load_checkpoint(ckpt_path)
    _, _, splits = prepare(REPO / 'supermarket_products.csv', seed=data['seed'])
    drop = {f.strip() for f in cfg.get('drop_features', '').split(',') if f.strip()}
    splits, _, _ = drop_feature_columns(splits, drop, False)
    x_cat, x_num, x_text, _ = splits['test']

    heads = [h for blk in model.blocks for h in blk.sa.heads]
    for h in heads:
        h.guardar_atencion = True
    n_capas = len(model.blocks)
    n_heads = len(model.blocks[0].sa.heads)
    suma, n = None, 0
    with torch.no_grad():
        for s in range(0, x_cat.shape[0], 1024):
            model(x_cat[s:s + 1024], x_num[s:s + 1024], x_text[s:s + 1024])
            lote = torch.stack([h.ultima_atencion.mean(0) for h in heads])  # (capas*heads, T, T)
            b = x_cat[s:s + 1024].shape[0]
            suma = lote * b if suma is None else suma + lote * b
            n += b
    att = (suma / n).reshape(n_capas, n_heads, suma.shape[-1], suma.shape[-1]).numpy()
    etiquetas = ['CLS'] + [ETIQUETAS_CORTAS.get(f, f) for f in ckpt['cat_features']] \
                        + [ETIQUETAS_CORTAS.get(f, f) for f in ckpt['num_features']]
    return att, etiquetas


def graficar(ckpt_path):
    att, etiquetas = mapas_de(ckpt_path)
    n_capas, n_heads, T, _ = att.shape
    fig, axes = plt.subplots(n_heads, n_capas, figsize=(4.4 * n_capas, 4.0 * n_heads),
                             squeeze=False)
    vmax = att.max()
    for c in range(n_capas):
        for h in range(n_heads):
            ax = axes[h][c]
            im = ax.imshow(att[c, h], cmap=CMAP, vmin=0, vmax=vmax)
            ax.set_xticks(range(T)), ax.set_yticks(range(T))
            ax.set_xticklabels(etiquetas, rotation=90, fontsize=6.5, color='#3D4B5A')
            ax.set_yticklabels(etiquetas, fontsize=6.5, color='#3D4B5A')
            ax.set_title(f'capa {c + 1}' + (f' · cabeza {h + 1}' if n_heads > 1 else ''),
                         fontsize=9, color='#1B2530')
            ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, shrink=0.75, label='peso de atención promedio (test)')
    # sin titulo: la presentacion ya lo pone en el encabezado de la slide
    fig.suptitle('fila = token que consulta · columna = token atendido', fontsize=10)
    out = REPO / 'graficos' / f'atencion_{ckpt_path.stem}.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'grafico -> {out.relative_to(REPO)}')

    for c in range(n_capas):
        fila_cls = att[c, :, 0, :].mean(0)  # promedio de cabezas: a quien mira el CLS
        top = np.argsort(fila_cls)[::-1][:5]
        detalle = ' · '.join(f'{etiquetas[i]} {fila_cls[i]:.2f}' for i in top)
        print(f'  capa {c + 1}, fila CLS (top-5): {detalle}')


def main():
    if len(sys.argv) > 1:
        ckpts = [Path(p) for p in sys.argv[1:]]
    else:
        candidatos = sorted((REPO / 'pesos').glob('pac20_feat_h1_*.pt'))
        mejor = max(candidatos, key=lambda p: json.loads(
            (REPO / 'resultados' / f'{p.stem}.json').read_text())['val']['pr_auc'])
        ckpts = [mejor]
    for c in ckpts:
        graficar(c)


if __name__ == '__main__':
    main()
