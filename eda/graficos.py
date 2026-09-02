"""Los graficos de la presentacion, con tipografia grande (legible proyectada).

    .venv/bin/python eda/graficos.py              ->  salidas/graficos/*.png (todos)
    .venv/bin/python eda/graficos.py grilla mlp   ->  solo esos

Cada figura se dibuja al tamano fisico con el que entra en la diapositiva, asi los
puntos tipograficos son reales: un texto de 18 pt aca es 18 pt en la pantalla.
Datos: leidos de salidas/resultados/*.json (media ± desvio poblacional sobre 6 seeds). Los
graficos de decision usan PR-AUC de VALIDACION (la metrica con la que se decidio);
SPLIT='test' los reproduce con test. importancia() y atencion() recalculan sobre los
checkpoints de salidas/pesos/ del modelo final.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RESULTADOS = REPO / 'salidas' / 'resultados'
PESOS = REPO / 'salidas' / 'pesos'
OUT = REPO / 'salidas' / 'graficos'
OUT.mkdir(parents=True, exist_ok=True)

TEAL, VIOLETA, GRIS, ROJO, ORO, INK, INK2, MUTED = (
    '#0E9B7E', '#7052C9', '#98A5B3', '#C22B5E', '#B58117', '#17242C', '#3D4B5A', '#5F7183')
FUENTE = 'Carlito' if any(f.name == 'Carlito' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({
    'font.family': FUENTE, 'font.size': 18, 'axes.titlesize': 20, 'axes.labelsize': 18,
    'xtick.labelsize': 17, 'ytick.labelsize': 18, 'legend.fontsize': 17,
    'axes.edgecolor': '#B8C2CC', 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 220,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})
COLOR_ROL = {'e': TEAL, 'a': VIOLETA, 'c': GRIS}
SPLIT = 'val'   # 'val' = metrica de decision (defensa) · 'test' = lo que se reporta del final
SPLIT_ES = {'val': 'validación', 'test': 'test'}[SPLIT]


def _stats(grupo, split=None):
    """(media, desvio poblacional) del PR-AUC de un grupo de resultados, seeds 42-47."""
    vals = [json.loads(f.read_text())[split or SPLIT]['pr_auc']
            for f in sorted(RESULTADOS.glob(f'{grupo}_seed4[2-7].json'))]
    assert len(vals) == 6, f'{grupo}: {len(vals)} corridas'
    v = np.array(vals)
    return float(v.mean()), float(v.std())


def guardar(fig, nombre):
    fig.savefig(OUT / nombre, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print('  ->', nombre)


# ---------- 1. un eje por experimento: punto +- desvio, la elegida en teal ----------
def decision(nombre, filas, xlabel=None, figsize=(6.6, 4.2), xlim=None, valores='arriba'):
    """filas: (etiqueta, grupo de resultados, rol) con rol e=elegido, a=alternativa, c=contexto.
    valores='derecha': el numero en una columna a la derecha de las barras (para muchas filas)."""
    xlabel = xlabel or f'PR-AUC {SPLIT_ES} (media ± desvío, 6 seeds)'
    filas = [(et, *_stats(grupo), rol) for et, grupo, rol in filas]
    n = len(filas)
    fig, ax = plt.subplots(figsize=figsize)
    x_num = max(m + s for _, m, s, _ in filas) + 0.006
    for i, (et, m, s, rol) in enumerate(reversed(filas)):
        c = COLOR_ROL[rol]
        ax.errorbar(m, i, xerr=s, fmt='o', color=c, ecolor=c, elinewidth=3 if rol == 'e' else 2.2,
                    capsize=5, capthick=2, markersize=13 if rol == 'e' else 10, zorder=3)
        if valores == 'derecha':
            ax.text(x_num, i, f'{m:.3f}', ha='left', va='center', fontsize=17,
                    fontweight='bold' if rol == 'e' else 'normal', color=INK if rol == 'e' else INK2)
        else:
            ax.annotate(f'{m:.3f}', (m, i), xytext=(0, 12), textcoords='offset points', ha='center',
                        fontsize=18, fontweight='bold' if rol == 'e' else 'normal',
                        color=INK if rol == 'e' else INK2)
    ax.set_yticks(range(n), [f[0] for f in reversed(filas)])
    for lab, (et, m, s, rol) in zip(ax.get_yticklabels(), reversed(filas)):
        lab.set_fontweight('bold' if rol == 'e' else 'normal')
        lab.set_color(INK if rol != 'c' else MUTED)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', color='#E3E8EE', linewidth=1)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, n - 0.4 + (0.35 if valores == 'arriba' else 0))
    if xlim:
        ax.set_xlim(*xlim)
    elif valores == 'derecha':
        ax.set_xlim(right=x_num + 0.03)
    ax.tick_params(axis='y', length=0)
    guardar(fig, nombre)


def decisiones():
    print('decisiones por eje')
    decision('cabezas.png', [
        ('4 cabezas · ordinal', 'feat_ordinal_features_d32_h4_l2_linear_catordinal', 'e'),
        ('1 cabeza · ordinal', 'camp_ordinal_h1_features_d32_h1_l2_linear_catordinal', 'a'),
        ('1 cabeza · embeddings', 'pac20_feat_h1_features_d32_h1_l2_linear', 'c'),
        ('2 cabezas · embeddings', 'pac20_feat_h2_features_d32_h2_l2_linear', 'c'),
        ('4 cabezas · embeddings', 'pac20_feat_base_features_d32_h4_l2_linear', 'c'),
    ])
    decision('bloques.png', [
        ('2 bloques · 26k params', 'feat_ordinal_features_d32_h4_l2_linear_catordinal', 'e'),
        ('4 bloques · 51k params', 'camp_ordinal_l4_features_d32_h4_l4_linear_catordinal', 'a'),
        ('1 bloque · 14k params', 'min_l1_features_d32_h4_l1_linear_catordinal', 'a'),
        ('4 bloques · embeddings', 'pac20_feat_l4_features_d32_h4_l4_linear', 'c'),
        ('1 bloque · embeddings', 'pac20_feat_l1_features_d32_h4_l1_linear', 'c'),
    ])
    decision('dmodel.png', [
        ('d = 32 · 26.177 params', 'feat_ordinal_features_d32_h4_l2_linear_catordinal', 'e'),
        ('d = 16 · 6.945', 'min_d16_features_d16_h4_l2_linear_catordinal', 'a'),
        ('d = 8 · 1.937', 'min_d8_features_d8_h4_l2_linear_catordinal', 'a'),
        ('d = 16, 1 bloque · 3.713', 'min_d16l1_features_d16_h4_l1_linear_catordinal', 'c'),
        ('d = 64 · embeddings · 105k', 'pac20_feat_d64_features_d64_h4_l2_linear', 'c'),
    ])
    decision('encoding.png', [
        ('ordinal (rango por BTR)', 'feat_ordinal_features_d32_h4_l2_linear_catordinal', 'e'),
        ('target (BTR suavizado)', 'feat_target_features_d32_h4_l2_linear_cattarget', 'a'),
        ('embedding aprendido', 'pac20_feat_base_features_d32_h4_l2_linear', 'a'),
        ('one-hot (MLP)', 'mlp_onehot_mlp_d32_h4_l2_linear_catonehot', 'c'),
        ('hashing (8 buckets)', 'feat_hash8_features_d32_h4_l2_linear_cathashing8', 'a'),
        ('frecuencia', 'feat_freq_features_d32_h4_l2_linear_catfreq', 'a'),
    ], figsize=(6.6, 4.6), xlim=(0.1, 0.9))
    decision('init.png', [
        ('aleatoria · ordinal', 'feat_ordinal_features_d32_h4_l2_linear_catordinal', 'e'),
        ('MLM 20 épocas · ordinal', 'feat_ordinal_mlm20_features_d32_h4_l2_linear_catordinal_mlm20', 'a'),
        ('MLM 20 épocas · embeddings', 'feat_mlm20_features_d32_h4_l2_linear_mlm20', 'c'),
        ('aleatoria · embeddings', 'feat_base_features_d32_h4_l2_linear', 'c'),
        ('w2v-init · words (fusión)', 'fusion_words_w2v_fusion_d32_h4_l2_linear_wordsw2v', 'c'),
    ])


# ---------- 2. formulacion (texto) ----------
def formulacion():
    print('formulacion (texto)')
    filas = [('features (referencia, embeddings)', 'pac20_feat_base_features_d32_h4_l2_linear', TEAL),
             ('fusión: texto → 1 token', 'fusion_base_fusion_d32_h4_l2_linear', VIOLETA),
             ('torre: texto → vector + MLP', 'pac20_tower_base_tower_d32_h4_l2_linear', VIOLETA),
             ('híbrido sin regex', 'hybrid_sin_regex_hybrid_d32_h4_l2_linear_sin-listingstatus', VIOLETA),
             ('híbrido: features + 256 chars', 'hybrid_full_hybrid_d32_h4_l2_linear', VIOLETA),
             ('texto puro (chars)', 'text_base_text_d32_h4_l2_linear', VIOLETA),
             ('techo sin señal de estado', 'pac20_feat_intrinseco_features_d32_h4_l2_linear_sin-listingstatus', ORO)]
    filas = [(et, _stats(grupo)[0], c) for et, grupo, c in filas]
    filas = [filas[0], *sorted(filas[1:-1], key=lambda f: -f[1]), filas[-1]]  # texto: de mayor a menor
    fig, ax = plt.subplots(figsize=(12.0, 4.3))
    y = np.arange(len(filas))[::-1]
    ax.barh(y, [f[1] for f in filas], color=[f[2] for f in filas], height=0.62)
    for yi, (et, v, c) in zip(y, filas):
        ax.text(v + 0.008, yi, f'{v:.3f}', va='center', fontsize=18,
                fontweight='bold' if c == TEAL else 'normal', color=INK)
    ax.set_yticks(y, [f[0] for f in filas])
    ax.get_yticklabels()[0].set_fontweight('bold')
    ax.set_xlim(0, 0.9)
    ax.set_xlabel(f'PR-AUC {SPLIT_ES} (media, 6 seeds)')
    ax.grid(axis='x', color='#E3E8EE')
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    guardar(fig, 'formulacion.png')


# ---------- 2b. grilla d_model x cabezas (dos heatmaps) y d_model x bloques (10ª y 11ª tandas) ----------
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from experimentos import D_GRILLA, H_GRILLA, L_GRILLA, GrillaIncompleta, celda_grilla, mejor_h


def _stats_o_nada(grupo, split):
    """Como _stats, pero None si la celda todavia no tiene sus 6 corridas (grilla a medio correr)."""
    try:
        return _stats(grupo, split)
    except AssertionError:
        return None


def _params(grupo):
    f = next(RESULTADOS.glob(f'{grupo}_seed42.json'), None)
    return json.loads(f.read_text())['n_parametros'] if f else None


def _heatmap(ax, datos, filas, cols, elegida, titulo, cmap, norm, xlabel, ylabels, negrita=False):
    """datos[i][j] = (media, desvio) o None; la celda elegida va recuadrada."""
    M_ = np.array([[np.nan if c is None else c[0] for c in fila] for fila in datos])
    cmap = cmap.copy()
    cmap.set_bad('#FFFFFF')
    ax.imshow(M_, cmap=cmap, norm=norm, aspect='auto')
    for i in range(len(filas)):
        for j in range(len(cols)):
            c = datos[i][j]
            if c is None:
                ax.text(j, i, '—', ha='center', va='center', fontsize=16, color=MUTED)
                continue
            m, s = c
            claro = norm(m) > 0.55
            ax.text(j, i - 0.13, f'{m:.3f}', ha='center', va='center', fontsize=17, fontweight='bold',
                    color='white' if claro else INK)
            ax.text(j, i + 0.27, f'± {s:.3f}', ha='center', va='center', fontsize=11.5,
                    color='#E8F3EF' if claro else INK2)
    if elegida:
        i, j = elegida
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=3.5, zorder=4))
    ax.set_xticks(range(len(cols)), [str(c) for c in cols])
    ax.set_yticks(range(len(filas)), ylabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('d_model · parámetros')
    ax.set_title(titulo, fontsize=19, color=INK, pad=10, fontweight='bold' if negrita else 'normal')
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


CMAP_TEAL = LinearSegmentedColormap.from_list('teal', ['#F3F9F7', '#9FD8C8', '#0E9B7E', '#0B5B4C'])


def _colorbar(fig, axes, cmap, norm, split_es):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(f'PR-AUC {split_es}\n(media, 6 seeds)', fontsize=13)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_visible(False)


def _ylabel(grupo, d):
    p = _params(grupo)
    return f'{d} · {p / 1000:.0f}k' if p else str(d)


def grilla(split=None, nombre='grilla.png'):
    """Dos heatmaps (ordinal | embedding) de PR-AUC medio por celda d_model x cabezas, escala compartida.
    Las celdas que todavia no corrieron quedan en blanco (la grilla crece con la 11ª tanda)."""
    split = split or SPLIT
    split_es = {'val': 'validación', 'test': 'test'}[split]
    print(f'grilla d_model x cabezas ({split})')
    paneles = [('o', 'Encoding ordinal (el elegido)'), ('e', 'Embedding aprendido')]
    datos = {enc: [[_stats_o_nada(celda_grilla(enc, d, h), split) for h in H_GRILLA] for d in D_GRILLA]
             for enc, _ in paneles}
    # filas y columnas que todavia no tienen NINGUNA celda corrida no se dibujan (la grilla crece por tandas)
    ds = [i for i, d in enumerate(D_GRILLA) if any(datos[e][i][j] for e in datos for j in range(len(H_GRILLA)))]
    hs = [j for j, h in enumerate(H_GRILLA) if any(datos[e][i][j] for e in datos for i in range(len(D_GRILLA)))]
    D_, H_ = [D_GRILLA[i] for i in ds], [H_GRILLA[j] for j in hs]
    datos = {enc: [[datos[enc][i][j] for j in hs] for i in ds] for enc in datos}
    todos = [c[0] for enc in datos for fila in datos[enc] for c in fila if c]
    faltan = sum(c is None for enc in datos for fila in datos[enc] for c in fila)
    if faltan:
        print(f'  (grilla incompleta: faltan {faltan} celdas, quedan en blanco)')
    norm = Normalize(np.floor(min(todos) * 100) / 100, max(todos))
    fig, axes = plt.subplots(1, 2, figsize=(11.9, 3.55 if len(D_) == 3 else 4.3), gridspec_kw={'wspace': 0.42})
    for ax, (enc, titulo) in zip(axes, paneles):
        _heatmap(ax, datos[enc], D_, H_, (D_.index(32), H_.index(4)) if enc == 'o' else None, titulo,
                 CMAP_TEAL, norm, 'cabezas de atención',
                 [_ylabel(celda_grilla(enc, d, 4), d) for d in D_], negrita=(enc == 'o'))
    _colorbar(fig, axes, CMAP_TEAL, norm, split_es)
    guardar(fig, nombre)


def grilla_test():
    grilla('test', 'grilla_test.png')


def celda_bloques(d, l):
    """Prefijo de archivo de la celda (d_model, bloques) de la grilla de bloques (ordinal): cada d usa la
    cantidad de cabezas que mejor le dio en la grilla de cabezas; la fila l=2 es esa misma celda."""
    h = mejor_h('o', d)
    return celda_grilla('o', d, h) if l == 2 else f'gl_o_d{d}l{l}_features_d{d}_h{h}_l{l}_linear_catordinal'


def grilla_bloques(split=None, nombre='grilla_bloques.png'):
    """Heatmap d_model x bloques (ordinal, cabezas = las mejores de cada d_model), 11ª tanda."""
    split = split or SPLIT
    split_es = {'val': 'validación', 'test': 'test'}[split]
    print(f'grilla d_model x bloques ({split})')
    try:
        hs = {d: mejor_h('o', d) for d in D_GRILLA}
    except GrillaIncompleta as e:
        print(f'  todavia no: {e}')
        return
    datos = [[_stats_o_nada(celda_bloques(d, l), split) for l in L_GRILLA] for d in D_GRILLA]
    if not any(c for i, fila in enumerate(datos) for j, c in enumerate(fila) if L_GRILLA[j] != 2):
        print('  todavia no corrio ninguna celda gl_*')
        return
    todos = [c[0] for fila in datos for c in fila if c]
    faltan = sum(c is None for fila in datos for c in fila)
    if faltan:
        print(f'  (grilla incompleta: faltan {faltan} celdas, quedan en blanco)')
    norm = Normalize(np.floor(min(todos) * 100) / 100, max(todos))
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    _heatmap(ax, datos, D_GRILLA, L_GRILLA, (0, L_GRILLA.index(2)), 'Encoding ordinal · cabezas: las mejores de cada d',
             CMAP_TEAL, norm, 'bloques (atención + FFN)',
             [f'{d} · {hs[d]} cab.' for d in D_GRILLA], negrita=True)
    _colorbar(fig, ax, CMAP_TEAL, norm, split_es)
    guardar(fig, nombre)


def grilla_bloques_test():
    grilla_bloques('test', 'grilla_bloques_test.png')


# ---------- 2c. 10ª tanda: el mejor MLP que pudimos, sobre el mismo encoding ----------
def mlp():
    print('mlp (10 variantes sobre ordinal)')
    filas = [('Transformer d32 · 4 cabezas', 'feat_ordinal_features_d32_h4_l2_linear_catordinal', 'e'),
             ('MLP 256·64', 'mlp_ordinal_mlp_d32_h4_l2_linear_catordinal', 'a'),
             ('MLP 256·64 · dropout 0.2', 'mlp_ord_do2_mlp_d32_h4_l2_linear_catordinal_do0.2', 'a'),
             ('MLP 256·64 · dropout 0.3', 'mlp_ord_do3_mlp_d32_h4_l2_linear_catordinal_do0.3', 'a'),
             ('MLP 512·256', 'mlp_ord_ancho_mlp_d32_h4_l2_linear_catordinal_mh512x256', 'a'),
             ('MLP 256·128·64', 'mlp_ord_prof3_mlp_d32_h4_l2_linear_catordinal_mh256x128x64', 'a'),
             ('MLP 256·128·64·32', 'mlp_ord_prof4_mlp_d32_h4_l2_linear_catordinal_mh256x128x64x32', 'a'),
             ('MLP 1024·256 · dropout 0.2', 'mlp_ord_grande_mlp_d32_h4_l2_linear_catordinal_do0.2_mh1024x256', 'a'),
             ('MLP 64·32', 'mlp_ord_mini_mlp_d32_h4_l2_linear_catordinal_mh64x32', 'a'),
             ('MLP 256', 'mlp_ord_h256_mlp_d32_h4_l2_linear_catordinal_mh256', 'a'),
             ('MLP 128', 'mlp_ord_h128_mlp_d32_h4_l2_linear_catordinal_mh128', 'a'),
             ('MLP 256·64 · one-hot', 'mlp_onehot_mlp_d32_h4_l2_linear_catonehot', 'c'),
             ('MLP 256·64 · embeddings', 'mlp_base_mlp_d32_h4_l2_linear', 'c')]
    filas = [(f'{et} · {_params(grupo) / 1000:.0f}k', grupo, rol) for et, grupo, rol in filas]
    # el transformer arriba; los MLP sobre ordinal de mayor a menor; los otros encodings al final
    ord_ = sorted(filas[1:-2], key=lambda f: -_stats(f[1])[0])
    decision('mlp.png', [filas[0], *ord_, *filas[-2:]], figsize=(7.3, 5.05), valores='derecha')


# ---------- 3. curva de aprendizaje ----------
def _prauc_test(tag_prefix):
    vals = []
    for f in RESULTADOS.glob(f'{tag_prefix}_*seed4[2-7].json'):
        vals.append(json.loads(f.read_text())['test']['pr_auc'])
    return np.array(vals)


def curva_aprendizaje():
    print('curva de aprendizaje')
    tags = [('curva_frac25_features', 25), ('curva_frac50_features', 50),
            ('curva_frac75_features', 75), ('feat_ordinal_features', 100)]
    xs, ms, ss = [], [], []
    for t, p in tags:
        v = _prauc_test(t)
        xs.append(p); ms.append(v.mean()); ss.append(v.std())
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.axhline(0.762, color=ROJO, linestyle='--', linewidth=2)
    ax.text(101, 0.7655, 'GBM con el 100%: 0.762', color=ROJO, fontsize=16, ha='right')
    ax.errorbar(xs, ms, yerr=ss, color=TEAL, linewidth=3, marker='o', markersize=10, capsize=5,
                capthick=2, elinewidth=2)
    for x, m in zip(xs, ms):
        ax.annotate(f'{m:.3f}', (x, m), xytext=(0, 14), textcoords='offset points', ha='center',
                    fontsize=17, color=INK)
    ax.set_xticks(xs, [f'{x}%' for x in xs])
    ax.set_xlabel('fracción de las búsquedas de train (val y test fijos)')
    ax.set_ylabel('PR-AUC test')
    ax.set_ylim(0.70, 0.87)
    ax.grid(color='#E3E8EE')
    ax.set_axisbelow(True)
    guardar(fig, 'curva_aprendizaje.png')


# ---------- 4. curvas de entrenamiento del modelo final (seed 46) ----------
def curvas_entrenamiento():
    print('curvas de entrenamiento')
    d = json.loads((RESULTADOS /
                    'feat_ordinal_features_d32_h4_l2_linear_catordinal_seed46.json').read_text())
    h = d['historial']
    ep = [e['epoch'] for e in h]
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.3))
    for ax, met, tit in zip(axs, ['pr_auc', 'loss'], ['PR-AUC por época', 'loss (BCE) por época']):
        tr = [e['train'][met] for e in h]; va = [e['val'][met] for e in h]
        ax.plot(ep, tr, color=TEAL, linewidth=3, label='train')
        ax.plot(ep, va, color=VIOLETA, linewidth=3, label='validación')
        best = int(np.argmax([e['val']['pr_auc'] for e in h]))
        ax.axvline(ep[best], color=ROJO, linestyle='--', linewidth=2,
                   label='mejor val → early stopping')
        ax.set_title(tit, loc='left', color=INK)
        ax.set_xlabel('época')
        ax.grid(color='#E3E8EE'); ax.set_axisbelow(True)
    axs[0].legend(frameon=False, loc='lower right')
    fig.tight_layout(w_pad=2.5)
    guardar(fig, 'curvas_entrenamiento.png')


# ---------- 5. importancia por permutacion (recalculada, 6 seeds) ----------
def importancia():
    print('importancia por permutacion (6 seeds, CPU)')
    import torch
    from sklearn.metrics import average_precision_score
    from btr.data import prepare, CAT_FEATURES, NUM_FEATURES
    from btr.model import load_checkpoint
    caidas = []
    for seed in range(42, 48):
        ck = PESOS / f'feat_ordinal_features_d32_h4_l2_linear_catordinal_seed{seed}.pt'
        model, _ = load_checkpoint(ck)
        _, _, splits = prepare(REPO / 'supermarket_products.csv', seed=seed)
        x_cat, x_num, x_text, y = splits['test']
        y = y.numpy()
        base = average_precision_score(y, model.predict_proba(x_cat, x_num, x_text).numpy())
        rng = np.random.default_rng(0)
        fila = []
        for j in range(x_cat.shape[1] + x_num.shape[1]):
            xc, xn = x_cat.clone(), x_num.clone()
            perm = torch.tensor(rng.permutation(len(y)))
            if j < x_cat.shape[1]:
                xc[:, j] = xc[perm, j]
            else:
                xn[:, j - x_cat.shape[1]] = xn[perm, j - x_cat.shape[1]]
            fila.append(base - average_precision_score(y, model.predict_proba(xc, xn, x_text).numpy()))
        caidas.append(fila)
        print(f'  seed {seed}: base {base:.4f}')
    caidas = np.array(caidas)
    m, s = caidas.mean(0), caidas.std(0)
    nombres = CAT_FEATURES + NUM_FEATURES
    orden = np.argsort(m)[::-1]
    for i in orden:
        print(f'  {nombres[i]:<18} {m[i]:+.4f} ± {s[i]:.4f}')
    (OUT / 'importancia.json').write_text(json.dumps(
        {nombres[i]: [float(m[i]), float(s[i])] for i in orden}, indent=1))
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    top = orden[:7]
    y = np.arange(len(top))[::-1]
    ax.barh(y, m[top], xerr=s[top], color=[TEAL if k == 0 else (VIOLETA if k == 1 else GRIS)
                                            for k in range(len(top))],
            height=0.62, capsize=4, error_kw={'elinewidth': 2, 'ecolor': INK2})
    ax.set_yticks(y, [nombres[i] for i in top])
    for yi, i in zip(y, top):
        ax.text(max(m[i] + s[i], 0) + 0.012, yi, f'{m[i]:+.2f}', va='center', fontsize=17, color=INK)
    ax.set_xlim(-0.03, max(m) + 0.16)
    ax.set_xlabel('caída de PR-AUC test al permutar la columna')
    ax.grid(axis='x', color='#E3E8EE'); ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    ax.text(0.98, 0.04, 'las otras 6 features: ≈ 0', transform=ax.transAxes, ha='right',
            fontsize=15, color=MUTED)
    guardar(fig, 'importancia.png')


# ---------- 6. atencion del modelo final: capa 1, promedio de las 4 cabezas ----------
def atencion():
    print('mapa de atencion (capa 1, promedio de cabezas)')
    from eda.atencion import mapas_de
    ck = PESOS / 'feat_ordinal_features_d32_h4_l2_linear_catordinal_seed46.pt'
    att, et = mapas_de(ck)
    a = att[0].mean(0)  # capa 1, promedio de cabezas, (T, T)
    j = et.index('status')
    print(f'  CLS -> status (capa 1, promedio de cabezas): {a[0, j]:.3f}; por cabeza:',
          np.round(att[0][:, 0, j], 3).tolist())
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    im = ax.imshow(a, cmap='BuGn', vmin=0, vmax=max(0.6, a.max()))
    ax.set_xticks(range(len(et)), et, rotation=60, ha='right', fontsize=15)
    ax.set_yticks(range(len(et)), et, fontsize=15)
    ax.set_xlabel('token atendido', fontsize=16)
    ax.set_ylabel('token que consulta', fontsize=16)
    for s in ax.spines.values():
        s.set_visible(True)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), len(et), 1, fill=False, edgecolor=ROJO, linewidth=2.5))
    ax.set_title(f'la fila del CLS pone {a[0, j]:.2f} de su atención en status', loc='left',
                 color=ROJO, fontsize=17, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=14)
    cb.set_label('peso de atención promedio (test)', fontsize=14)
    guardar(fig, 'atencion.png')


if __name__ == '__main__':
    que = sys.argv[1:] or ['decisiones', 'formulacion', 'grilla', 'grilla_test', 'grilla_bloques',
                           'grilla_bloques_test', 'mlp', 'curva_aprendizaje',
                           'curvas_entrenamiento', 'importancia', 'atencion']
    for q in que:
        globals()[q]()
