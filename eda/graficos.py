"""Los graficos de la presentacion, con tipografia grande (legible proyectada).

    .venv/bin/python eda/graficos.py              ->  salidas/graficos/*.png (todos)
    .venv/bin/python eda/graficos.py grilla mlp   ->  solo esos

Cada figura se dibuja al tamano fisico con el que entra en la diapositiva, asi los
puntos tipograficos son reales: un texto de 18 pt aca es 18 pt en la pantalla.
Datos: leidos de salidas/resultados/*.json (media ± desvio poblacional sobre 6 seeds). Los
barridos usan PR-AUC de VALIDACION (la metrica con la que se decidio); SPLIT='test' los
reproduce con test. Las celdas que todavia no corrieron quedan en blanco. importancia() y
atencion() recalculan sobre los checkpoints de salidas/pesos/ del modelo final.
"""
import json
import re
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


MIN_SEEDS = 1   # la 12a tanda llega de a seeds: una celda se dibuja con las que haya (y marca n)


def _seeds(grupo):
    return {int(f.name.rsplit('_seed', 1)[1][:-5]) for f in RESULTADOS.glob(f'{grupo}_seed4[2-7].json')}


def _stats(grupo, split=None, seeds=None):
    """(media, desvio poblacional) del PR-AUC de un grupo (seeds 42-47 las que haya, o solo `seeds`)."""
    vals = [json.loads(f.read_text())[split or SPLIT]['pr_auc']
            for f in sorted(RESULTADOS.glob(f'{grupo}_seed4[2-7].json'))
            if seeds is None or int(f.name.rsplit('_seed', 1)[1][:-5]) in seeds]
    assert len(vals) >= MIN_SEEDS, f'{grupo}: {len(vals)} corridas'
    v = np.array(vals)
    return float(v.mean()), float(v.std())


def _n(grupo):
    return len(_seeds(grupo))


def seeds_comunes(grupos):
    """Las seeds que tienen TODOS los grupos ya corridos: mientras una tanda llega de a seeds, todas las
    celdas se comparan sobre las mismas seeds (una celda con 1 seed contra medias de 6 engana).
    Devuelve None si todos tienen las 6."""
    conj = [_seeds(g) for g in grupos if _seeds(g)]
    if not conj:
        return None
    if all(len(c) == 6 for c in conj):
        return None
    # un grupo con UNA sola seed no restringe a los demas: se dibuja con su seed y marca n=1
    base = [c for c in conj if len(c) > 1] or conj
    return set.intersection(*base) or None


def _nota_seeds(seeds):
    return '' if seeds is None else f' — provisorio: seed{"s" if len(seeds) > 1 else ""} {", ".join(map(str, sorted(seeds)))}'


def guardar(fig, nombre):
    fig.savefig(OUT / nombre, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print('  ->', nombre)


# ---------- 2b. grilla d_model x cabezas (dos heatmaps) y d_model x bloques (10ª y 11ª tandas) ----------
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from experimentos import (D_ENC, D_GRILLA, H_GRILLA, L_GRILLA, LRS, BATCHES, MLM_EPOCAS, MEJOR_ARQ,
                          GrillaIncompleta, canon_mejor, celda_grilla, grupo_canonico, mejor_h)


def _stats_o_nada(grupo, split, seeds=None):
    """Como _stats, pero None si la celda todavia no corrio (grilla a medio correr). Si la celda
    no tiene todas las `seeds` pedidas, usa las suyas (y el n de la celda lo cuenta)."""
    try:
        if seeds is not None and not seeds <= _seeds(grupo):
            seeds = None
        return _stats(grupo, split, seeds)
    except AssertionError:
        return None


def _params(grupo):
    f = next(RESULTADOS.glob(f'{grupo}_seed42.json'), None)
    return json.loads(f.read_text())['n_parametros'] if f else None


def _heatmap(ax, datos, filas, cols, elegida, titulo, cmap, norm, xlabel, ylabels, negrita=False, ns=None):
    """datos[i][j] = (media, desvio) o None; la(s) celda(s) elegida(s) van recuadradas (una tupla o una
    lista de tuplas); ns[i][j] = seeds de la celda."""
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
            n = ns[i][j] if ns else 6
            ax.text(j, i + 0.27, f'± {s:.3f}' if n >= 6 else (f'± {s:.3f} · n={n}' if n > 1 else 'n=1'),
                    ha='center', va='center', fontsize=11.5 if n >= 6 else 10,
                    color='#E8F3EF' if claro else INK2)
    if elegida:
        for i, j in (elegida if isinstance(elegida, list) else [elegida]):
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


def _norma(valores, rango=0.06):
    """Escala de color: del maximo hacia abajo como mucho `rango`; lo que colapsa (hashing,
    frecuencia, 1 cabeza a d128) queda en el color mas claro y su numero cuenta la verdad."""
    hi = max(valores)
    lo = max(np.floor(min(valores) * 100) / 100, hi - rango)
    return Normalize(lo, hi)


def _colorbar(fig, axes, cmap, norm, split_es):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(f'PR-AUC {split_es}\n(media entre seeds)', fontsize=13)
    cb.ax.tick_params(labelsize=12)
    cb.ax.yaxis.set_major_formatter('{x:.3f}')
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
    norm = _norma(todos)
    fig, axes = plt.subplots(1, 2, figsize=(11.9, 3.55 if len(D_) == 3 else 4.3), gridspec_kw={'wspace': 0.42})
    for ax, (enc, titulo) in zip(axes, paneles):
        # ordinal: se recuadra la mejor celda de cada fila (las cabezas con las que sigue el Exp. 2)
        cajas = None
        if enc == 'o':
            cajas = [(i, max(range(len(H_)), key=lambda j: (datos[enc][i][j] or (-1,))[0])) for i in range(len(D_))
                     if any(datos[enc][i])]
        _heatmap(ax, datos[enc], D_, H_, cajas, titulo, CMAP_TEAL, norm, 'cabezas de atención',
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
    norm = _norma(todos)
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    d_mejor = int(MEJOR_ARQ[MEJOR_ARQ.index('--d-model') + 1]); l_mejor = int(MEJOR_ARQ[MEJOR_ARQ.index('--n-layer') + 1])
    _heatmap(ax, datos, D_GRILLA, L_GRILLA, (D_GRILLA.index(d_mejor), L_GRILLA.index(l_mejor)),
             'Encoding ordinal · cabezas: las mejores de cada d — recuadro: la ganadora',
             CMAP_TEAL, norm, 'bloques (atención + FFN)',
             [f'{d} · {hs[d]} cab.' for d in D_GRILLA], negrita=True)
    ax.set_ylabel('d_model · cabezas')
    _colorbar(fig, ax, CMAP_TEAL, norm, split_es)
    guardar(fig, nombre)


def grilla_bloques_test():
    grilla_bloques('test', 'grilla_bloques_test.png')


# ---------- 2c. barridos de la 12ª tanda: encoding x d_model, MLM, regularizacion, optimizacion ----------
def _heatmap_simple(nombre, titulo, celdas, filas, cols, elegida, xlabel, ylabel, split, figsize=(7.0, 4.3),
                    negrita=True):
    """celdas[(fila, col)] = prefijo de archivo; dibuja un heatmap tolerante a celdas faltantes."""
    split_es = {'val': 'validación', 'test': 'test'}[split]
    comunes = seeds_comunes(list(celdas.values()))
    datos = [[_stats_o_nada(celdas[(f, c)], split, comunes) for c in cols] for f in filas]
    ns = [[(len(comunes) if comunes and comunes <= _seeds(celdas[(f, c)]) else _n(celdas[(f, c)])) for c in cols]
          for f in filas]
    titulo = titulo + _nota_seeds(comunes)
    todos = [c[0] for fila in datos for c in fila if c]
    if not todos:
        print(f'  {nombre}: todavia no corrio ninguna celda')
        return
    faltan = sum(c is None for fila in datos for c in fila)
    if faltan:
        print(f'  ({nombre}: faltan {faltan} celdas, quedan en blanco)')
    norm = _norma(todos)
    fig, ax = plt.subplots(figsize=figsize)
    _heatmap(ax, datos, filas, cols, elegida, titulo, CMAP_TEAL, norm, xlabel, [str(f) for f in filas],
             negrita=negrita, ns=ns)
    ax.set_ylabel(ylabel)
    _colorbar(fig, ax, CMAP_TEAL, norm, split_es)
    guardar(fig, nombre)


VIEJA = 'features_d32_h4_l2_linear_catordinal'   # la arquitectura anterior (32·4·2): MLM, optimizacion y
                                                  # transfer se corrieron sobre ella (3 seeds) y no se repiten


def _canon(base=None, **cambios):
    """Nombre canonico de la ganadora (o de `base`) con cambios: d=..., enc='embedding'|'target'|'ordinal',
    sufijo='_mlm5', form='ing_fusion'."""
    canon = base or canon_mejor()              # p. ej. features_d32_h16_l4_linear_catordinal
    if 'd' in cambios:
        canon = re.sub(r'_d\d+_', f'_d{cambios["d"]}_', canon, count=1)
    if 'enc' in cambios:
        canon = canon.replace('_catordinal', '')
        if cambios['enc'] != 'embedding':
            canon = canon + f'_cat{cambios["enc"]}'
    if 'form' in cambios:
        canon = canon.replace('features_', cambios['form'] + '_', 1)
    return canon + cambios.get('sufijo', '')


def _grupo(**cambios):
    """El grupo ya corrido con ese nombre canonico (cualquier tag), o un prefijo que no existe."""
    canon = _canon(**cambios)
    return grupo_canonico(canon) or f'(falta)_{canon}'


def _grupo_viejo(**cambios):
    """Idem, sobre la arquitectura anterior (32·4·2)."""
    canon = _canon(base=VIEJA, **cambios)
    return grupo_canonico(canon) or f'(falta)_{canon}'


NOTA_VIEJA = ' — sobre la arquitectura anterior (32·4·2), no repetido'


ENCODINGS = [('ordinal', 'ordinal (el elegido)'), ('embedding', 'embedding'), ('target', 'target')]


def encoding(split=None, nombre='encoding.png'):
    split = split or SPLIT
    print(f'encoding x d_model ({split})')
    celdas = {(et, d): _grupo(enc=enc, d=d) for enc, et in ENCODINGS for d in D_ENC}
    d_mejor = int(MEJOR_ARQ[MEJOR_ARQ.index('--d-model') + 1])
    _heatmap_simple(nombre, 'Encoding de las categóricas × d_model (cabezas y bloques de la ganadora)', celdas,
                    [et for _, et in ENCODINGS], list(D_ENC), (0, D_ENC.index(d_mejor)), 'd_model',
                    'encoding de las categóricas', split, figsize=(7.2, 4.0))


def encoding_test():
    encoding('test', 'encoding_test.png')


def celda_mlm(enc, epocas):
    return _grupo_viejo(enc='ordinal' if enc == 'o' else 'embedding', sufijo=f'_mlm{epocas}' if epocas else '')


def mlm(split=None, nombre='mlm.png'):
    """Pre-entrenamiento MLM sobre features: PR-AUC vs epocas de MLM, una curva por encoding."""
    split = split or SPLIT
    split_es = {'val': 'validación', 'test': 'test'}[split]
    print(f'mlm ({split})')
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    algo = False
    comunes = seeds_comunes([celda_mlm(enc, e) for enc in 'oe' for e in MLM_EPOCAS])
    for enc, et, color, dy in [('o', 'ordinal (el elegido)', TEAL, 14), ('e', 'embedding aprendido', VIOLETA, -22)]:
        xs, ms, ss, ns_ = [], [], [], []
        for i, e in enumerate(MLM_EPOCAS):
            st = _stats_o_nada(celda_mlm(enc, e), split, comunes)
            if st:
                xs.append(i); ms.append(st[0]); ss.append(st[1])
                ns_.append(len(comunes) if comunes and comunes <= _seeds(celda_mlm(enc, e)) else _n(celda_mlm(enc, e)))
        if xs:
            algo = True
            ax.errorbar(xs, ms, yerr=ss, color=color, linewidth=3, marker='o', markersize=10,
                        capsize=5, capthick=2, elinewidth=2, label=et)
            for x, m, n in zip(xs, ms, ns_):
                ax.annotate(f'{m:.3f}' + (f' (n={n})' if n < 6 else ''), (x, m), xytext=(0, dy),
                            textcoords='offset points', ha='center', fontsize=13 if n < 6 else 15, color=INK)
    if not algo:
        print('  mlm: todavia no corrio ninguna celda')
        plt.close(fig)
        return
    ax.set_xticks(range(len(MLM_EPOCAS)), [str(e) for e in MLM_EPOCAS])
    ax.set_xlim(-0.4, len(MLM_EPOCAS) - 0.6)
    ax.set_xlabel('épocas de pre-entrenamiento MLM (0 = inicialización aleatoria)' + _nota_seeds(comunes))
    ax.set_ylabel(f'PR-AUC {split_es}')
    ax.set_title('Pre-entrenamiento MLM' + NOTA_VIEJA, loc='left', color=INK, fontsize=15)
    ax.grid(color='#E3E8EE'); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=15)
    guardar(fig, nombre)


def mlm_test():
    mlm('test', 'mlm_test.png')


def celda_opt(lr, bs):
    suf = (f'_lr{float(lr):g}' if lr != '0.001' else '') + (f'_bs{bs}' if bs != '256' else '')
    return _grupo_viejo(sufijo=suf)


def optimizacion(split=None, nombre='optimizacion.png'):
    split = split or SPLIT
    print(f'optimizacion ({split})')
    celdas = {(lr, bs): celda_opt(lr, bs) for lr in LRS for bs in BATCHES}
    _heatmap_simple(nombre, 'Optimización: learning rate × batch' + NOTA_VIEJA, celdas,
                    [f'{float(lr):g}' for lr in LRS], list(BATCHES),
                    (LRS.index('0.001'), BATCHES.index('256')), 'batch', 'learning rate', split, figsize=(6.4, 4.3))


def optimizacion_test():
    optimizacion('test', 'optimizacion_test.png')


# ---------- 2d. la alternativa (ingredientes) y el transfer learning: puntos ± desvio ----------
def puntos(nombre, filas, xlabel, figsize=(7.3, 4.0), xlim=None):
    """filas: (etiqueta, prefijo de archivo, rol) con rol e=elegido, a=alternativa, c=contexto."""
    comunes = seeds_comunes([g for _, g, _ in filas])
    datos = [(et, _stats_o_nada(grupo, SPLIT, comunes), rol,
              (len(comunes) if comunes and comunes <= _seeds(grupo) else _n(grupo)))
             for et, grupo, rol in filas]
    xlabel = xlabel + _nota_seeds(comunes)
    if all(st is None for _, st, _, _ in datos):
        print(f'  {nombre}: todavia no corrio ninguna fila')
        return
    fig, ax = plt.subplots(figsize=figsize)
    n = len(datos)
    for i, (et, st, rol, n) in enumerate(reversed(datos)):
        c = COLOR_ROL[rol]
        if st is None:
            ax.text(0.5, i, 'pendiente', transform=ax.get_yaxis_transform(), ha='center', va='center',
                    fontsize=14, color=MUTED)
            continue
        m, s_ = st
        ax.errorbar(m, i, xerr=s_, fmt='o', color=c, ecolor=c, elinewidth=3 if rol == 'e' else 2.2,
                    capsize=5, capthick=2, markersize=13 if rol == 'e' else 10, zorder=3)
        ax.annotate(f'{m:.3f}' + (f'  (n={n})' if n < 6 else ''), (m, i), xytext=(0, 12),
                    textcoords='offset points', ha='center', fontsize=17 if n >= 6 else 14,
                    fontweight='bold' if rol == 'e' else 'normal', color=INK if rol == 'e' else INK2)
    ax.set_yticks(range(len(datos)), [f[0] for f in reversed(datos)])
    for lab, (et, st, rol, _) in zip(ax.get_yticklabels(), reversed(datos)):
        lab.set_fontweight('bold' if rol == 'e' else 'normal')
        lab.set_color(INK if rol != 'c' else MUTED)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', color='#E3E8EE'); ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(datos) - 0.4 + 0.35)
    if xlim:
        ax.set_xlim(*xlim)
    ax.tick_params(axis='y', length=0)
    guardar(fig, nombre)


def ingredientes():
    """Ingredientes vs sin ingredientes sobre la ganadora, con tres tamanos del encoder (+ el control)."""
    print('ingredientes sobre la ganadora')
    puntos('ingredientes.png', [
        ('sin ingredientes (la ganadora)', _grupo(), 'e'),
        ('+ encoder chico · d16 · 2 cab. · 1 bloque', _grupo(form='ing_fusion', sufijo='_ingd16_ingh2'), 'a'),
        ('+ encoder base · d y cabezas del modelo · 1 bloque', _grupo(form='ing_fusion'), 'a'),
        ('+ encoder grande · d64 · 8 cab. · 2 bloques', _grupo(form='ing_fusion', sufijo='_il2_ingd64_ingh8'), 'a'),
        ('solo ingredientes', 'ing_solo_ing_d32_h4_l2_linear', 'c'),
    ], f'PR-AUC {SPLIT_ES} (media ± desvío entre seeds)', figsize=(7.6, 4.0), xlim=(0.1, 0.9))


def transfer():
    print('transfer learning (titulo preentrenado) sobre la ganadora')
    # todo sobre la arquitectura anterior (32·4·2, 3 seeds): no se repitio sobre la ganadora
    puntos('transfer.png', [
        ('sin título (32·4·2)', _grupo_viejo(), 'e'),
        ('+ MiniLM-L6 · 22M · congelado', _grupo_viejo(sufijo='_temb-titulominilm'), 'a'),
        ('+ mpnet-base · 110M · congelado', _grupo_viejo(sufijo='_temb-titulompnet'), 'a'),
        ('+ bge-large · 335M · congelado', _grupo_viejo(sufijo='_temb-titulobge'), 'a'),
        ('+ MiniLM-L6 · fine-tuning', _grupo_viejo(sufijo='_tembft-titulo'), 'a'),
        ('solo el título (bge-large)', _grupo_viejo(enc='embedding', sufijo='_temb-titulobge_sin-all'), 'c'),
    ], f'PR-AUC {SPLIT_ES} (media ± desvío entre seeds)' + NOTA_VIEJA, figsize=(7.3, 4.3), xlim=(0.1, 0.9))


def tiempo():
    """Hora y dia de la semana, ciclicas: sin tiempo vs (sin, cos) vs categorico, sobre la ganadora."""
    print('tiempo (hora, dia de la semana) sobre la ganadora')
    puntos('tiempo.png', [
        ('sin tiempo (la ganadora)', _grupo(), 'e'),
        ('+ hora y día como (sin, cos) · un token cada una', _grupo(sufijo='_tiempo-ciclico'), 'a'),
        ('+ hora y día como categóricas · 24 + 7 niveles', _grupo(sufijo='_tiempo-cat'), 'a'),
    ], f'PR-AUC {SPLIT_ES} (media ± desvío entre seeds)', figsize=(7.6, 3.4))


# ---------- 3. curva de aprendizaje ----------
def _prauc(tag_prefix, split='test'):
    vals = []
    for f in RESULTADOS.glob(f'{tag_prefix}_*seed4[2-7].json'):
        vals.append(json.loads(f.read_text())[split]['pr_auc'])
    return np.array(vals)


def curva_aprendizaje():
    print('curva de aprendizaje (test)')
    tags = [('curva_frac25_features', 25), ('curva_frac50_features', 50),
            ('curva_frac75_features', 75), ('feat_ordinal_features', 100)]
    xs, ms, ss = [], [], []
    for t, p in tags:
        v = _prauc(t)
        xs.append(p); ms.append(v.mean()); ss.append(v.std())
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.errorbar(xs, ms, yerr=ss, color=TEAL, linewidth=3, marker='o', markersize=10, capsize=5,
                capthick=2, elinewidth=2)
    for x, m in zip(xs, ms):
        ax.annotate(f'{m:.3f}', (x, m), xytext=(0, 14), textcoords='offset points', ha='center',
                    fontsize=17, color=INK)
    ax.set_xticks(xs, [f'{x}%' for x in xs])
    ax.set_xlabel('fracción de las búsquedas de train (val y test fijos)')
    ax.set_ylabel('PR-AUC test')
    ax.set_ylim(0.72, 0.87)
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
    que = sys.argv[1:] or ['grilla', 'grilla_test', 'grilla_bloques', 'grilla_bloques_test',
                           'encoding', 'encoding_test', 'mlm', 'mlm_test', 'optimizacion', 'optimizacion_test',
                           'ingredientes', 'transfer', 'tiempo', 'curva_aprendizaje',
                           'curvas_entrenamiento', 'importancia', 'atencion']
    for q in que:
        globals()[q]()
