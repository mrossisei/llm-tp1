"""Suite curada de experimentos del TP: todo lo que sostiene la presentacion.

USO EN LA MAQUINA CON GPU (dos lineas):
    .venv/bin/python experimentos.py              # corre TODA la suite (lo que falte)
    .venv/bin/python experimentos.py --resumen    # tabla comparativa: media +- desvio por config

Garantias de la suite:
- Usa la GPU automaticamente (--device auto -> cuda si esta disponible). Si la familia
  "texto" fuera a correr en CPU, ABORTA con instrucciones (evita 20+ horas de CPU por
  un torch mal instalado); para forzar CPU a proposito: --device cpu.
- Es RESUMIBLE: cada (experimento, seed) que ya tiene su JSON en salidas/resultados/ se
  saltea. Si se corta a la mitad, volver a correr la misma linea continua donde quedo.
- Guarda salidas/pesos/ por defecto (checkpoints recargables para analisis posteriores,
  p. ej. mapas de atencion); desactivable con --no-pesos.
- Si un experimento falla, sigue con el resto y lo reporta al final.

Otras opciones: --list, --plan, --only a,b (admite comodines: 'gc_*'), --familia
tabular|texto, --seeds N, --epochs N (pruebas).

Cada bloque de abajo dice que diapositiva/figura de la presentacion sostiene. Las
configuraciones exploratorias que no entraron a la presentacion (listwise, multi-task
con cart, pesos por feature, regularizacion, transfer learning, SOM/PCA/autoencoder,
MiniLM, destilacion) se sacaron del codigo; sus corridas siguen en salidas/resultados/
y su analisis en analisis.md.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from btr.train import build_parser, run_name

REPO_ROOT = Path(__file__).resolve().parent
RESULTADOS = REPO_ROOT / 'salidas' / 'resultados'

PAC = ['--patience', '20', '--epochs', '300']          # protocolo tabular (2da tanda en adelante)
ORD = ['--cat-encoding', 'ordinal', *PAC]              # la config exacta del modelo final

# nombre -> (argumentos extra, familia)  [familia: 'tabular' barata | 'texto' GPU]
EXPERIMENTOS = {
    # ---- 1ra tanda (protocolo 60 epocas / paciencia 8; la familia texto se queda con este
    # protocolo: con paciencia 20 la seleccion por validacion sobreajusta, ver analisis.md) ----
    # las formulaciones y arquitecturas (Alternativas, Exp. 6 texto, curva de formulaciones)
    'feat_base':        (['--formulation', 'features'], 'tabular'),   # embeddings, la base original
    'mlp_base':         (['--arch', 'mlp'], 'tabular'),               # baseline MLP, mismos embeddings
    'text_base':        (['--formulation', 'text'], 'texto'),         # texto crudo (chars)
    'hybrid_full':      (['--formulation', 'hybrid'], 'texto'),       # features + 256 chars
    'tower_base':       (['--arch', 'tower'], 'texto'),               # encoder de texto + MLP
    # ¿el transformer redescubre la senal del texto sin el regex?
    'hybrid_sin_regex': (['--formulation', 'hybrid', '--drop-features', 'listing_status'], 'texto'),
    # familia "producto nuevo" (sin informacion de estado: la discusion conceptual, ~0.16)
    'feat_intrinseco':  (['--formulation', 'features', '--drop-features', 'listing_status'], 'tabular'),
    'text_intrinseco':  (['--formulation', 'text', '--strip-status'], 'texto'),
    'hybrid_intrinseco': (['--formulation', 'hybrid', '--strip-status',
                           '--drop-features', 'listing_status'], 'texto'),
    # ablaciones de arquitectura (las hipotesis refutadas de "Desafios")
    'feat_bins':        (['--numeric-mode', 'bins'], 'tabular'),      # bins por cuantiles para la U del precio
    'feat_pos':         (['--positional'], 'tabular'),                # positional encoding en features
    'feat_causal':      (['--causal'], 'tabular'),                    # mascara causal (degenera: CLS en 0)
    'feat_mean':        (['--pooling', 'mean'], 'tabular'),           # mean pooling en vez de CLS
    'feat_posweight':   (['--pos-weight'], 'tabular'),                # pesar la clase positiva
}

# ---- las mismas configs tabulares de la 1ra tanda con el protocolo 300/20 (grupos pac20_*,
# corridos por Matias): son las celdas "embedding" de los graficos de cabezas, bloques,
# d_model, encoding y formulaciones ----
EXPERIMENTOS |= {
    'pac20_feat_base':       ([*PAC], 'tabular'),
    'pac20_feat_h1':         (['--n-head', '1', *PAC], 'tabular'),
    'pac20_feat_h2':         (['--n-head', '2', *PAC], 'tabular'),
    'pac20_feat_l1':         (['--n-layer', '1', *PAC], 'tabular'),
    'pac20_feat_l4':         (['--n-layer', '4', *PAC], 'tabular'),
    'pac20_feat_d64':        (['--d-model', '64', *PAC], 'tabular'),
    'pac20_feat_intrinseco': (['--drop-features', 'listing_status', *PAC], 'tabular'),
    'pac20_tower_base':      (['--arch', 'tower', *PAC], 'texto'),
    'pac20_feat_bins':       (['--numeric-mode', 'bins', *PAC], 'tabular'),
    'pac20_feat_pos':        (['--positional', *PAC], 'tabular'),
    'pac20_feat_causal':     (['--causal', *PAC], 'tabular'),
    'pac20_feat_mean':       (['--pooling', 'mean', *PAC], 'tabular'),
    'pac20_feat_posweight':  (['--pos-weight', *PAC], 'tabular'),
}

# ---- 2da y 3ra tanda: los ejes que decidieron el modelo (Exp. 2-6) ----
EXPERIMENTOS |= {
    # grilla "campeon" sobre embeddings: combinar los ganadores de las ablaciones
    # (camp_d64l4 y camp_d64h1l4 son dos de las 4 configs del empate en validacion;
    # camp_d64h1 es la celda 64x1 del heatmap de embeddings)
    'camp_d64h1':       (['--d-model', '64', '--n-head', '1', *PAC], 'tabular'),
    'camp_d64l4':       (['--d-model', '64', '--n-layer', '4', *PAC], 'tabular'),
    'camp_d64h1l4':     (['--d-model', '64', '--n-head', '1', '--n-layer', '4', *PAC], 'tabular'),
    # causal hecho bien: el feat_causal original degeneraba (CLS en posicion 0 solo
    # se ve a si mismo -> p constante, ROC 0.500 medido); con el CLS al final el
    # experimento "¿importa la bidireccionalidad?" por fin se puede responder
    'feat_causal_last': (['--causal', '--cls-position', 'last', *PAC], 'tabular'),
    # Exp. 5, el decisivo: encodings de las categoricas
    'feat_ordinal':     ([*ORD], 'tabular'),                                     # EL MODELO FINAL
    'feat_target':      (['--cat-encoding', 'target', *PAC], 'tabular'),
    'feat_freq':        (['--cat-encoding', 'freq', *PAC], 'tabular'),
    'feat_hash8':       (['--cat-encoding', 'hashing', '--hash-buckets', '8', *PAC], 'tabular'),
    'mlp_onehot':       (['--arch', 'mlp', '--cat-encoding', 'onehot', *PAC], 'tabular'),
    # el campeon ordinal combinado con los ganadores de capacidad (Exp. 2 y 3)
    'camp_ordinal_h1':  ([*ORD, '--n-head', '1'], 'tabular'),
    'camp_ordinal_l4':  ([*ORD, '--n-layer', '4'], 'tabular'),
    # el texto resumido a UN token de la secuencia tabular (fusion), con chars o con
    # palabras, y las palabras inicializadas con un skipgram propio (Exp. 6 inicializacion)
    'fusion_base':      (['--formulation', 'fusion'], 'texto'),
    'fusion_words':     (['--formulation', 'fusion', '--text-tokens', 'words',
                          '--max-text-len', '64'], 'texto'),
    'fusion_words_w2v': (['--formulation', 'fusion', '--text-tokens', 'words',
                          '--max-text-len', '64', '--w2v-init'], 'texto'),
}

# ---- 4ta tanda: robustez del MODELO FINAL (diapositivas Robustez, Modelo final, Exp. 6) ----
EXPERIMENTOS |= {
    # curva de aprendizaje (100% = feat_ordinal)
    'curva_frac25': ([*ORD, '--train-frac', '0.25'], 'tabular'),
    'curva_frac50': ([*ORD, '--train-frac', '0.5'], 'tabular'),
    'curva_frac75': ([*ORD, '--train-frac', '0.75'], 'tabular'),
    # varianza: mismo split, otra inicializacion (grilla 6 splits x 6 inits) y deep-ensemble
    'robu_init43': ([*ORD, '--init-seed', '43'], 'tabular'),
    'robu_init44': ([*ORD, '--init-seed', '44'], 'tabular'),
    'robu_init45': ([*ORD, '--init-seed', '45'], 'tabular'),
    'robu_init46': ([*ORD, '--init-seed', '46'], 'tabular'),
    'robu_init47': ([*ORD, '--init-seed', '47'], 'tabular'),
    # MLM sobre features: pre-entrenar el tronco enmascarando una columna por fila
    'feat_mlm20':         (['--pretrain-mlm', '20', *PAC], 'tabular'),
    'feat_ordinal_mlm20': ([*ORD, '--pretrain-mlm', '20'], 'tabular'),
    # GroupKFold 5: cada query pasa por test una vez por seed -> 0.821 +- 0.012
    'cv5_fold0': ([*ORD, '--cv-k', '5', '--cv-fold', '0'], 'tabular'),
    'cv5_fold1': ([*ORD, '--cv-k', '5', '--cv-fold', '1'], 'tabular'),
    'cv5_fold2': ([*ORD, '--cv-k', '5', '--cv-fold', '2'], 'tabular'),
    'cv5_fold3': ([*ORD, '--cv-k', '5', '--cv-fold', '3'], 'tabular'),
    'cv5_fold4': ([*ORD, '--cv-k', '5', '--cv-fold', '4'], 'tabular'),
}

# ---- 5ta tanda: achicar el campeon (Exp. 3 bloques y Exp. 4 d_model, sobre ordinal) ----
EXPERIMENTOS |= {
    'min_d16':   ([*ORD, '--d-model', '16'], 'tabular'),                   # 6.945 parametros
    'min_d8':    ([*ORD, '--d-model', '8'], 'tabular'),                    # 1.937
    'min_l1':    ([*ORD, '--n-layer', '1'], 'tabular'),
    'min_d16l1': ([*ORD, '--d-model', '16', '--n-layer', '1'], 'tabular'),  # 3.713: empata al campeon
}

# ---- 9na tanda: los INGREDIENTES como conjunto (Alternativas: 0.817, igual que sin ellos) ----
# Encoder de conjunto: [ING] + un token por ingrediente, sin positional encoding (la
# lista no tiene orden), atencion bidireccional. Hipotesis registrada antes de correr:
# los ingredientes co-ocurren en recetas fijas por categoria, asi que category ya lo
# captura y todo da delta ~ 0 vs feat_ordinal; ing_solo separa "no hay senal" de "la
# senal ya la tenian otras columnas" (dio 0.140 ~ azar).
EXPERIMENTOS |= {
    'ing_solo':   (['--formulation', 'ing', *PAC], 'tabular'),
    'ing_fusion': ([*ORD, '--formulation', 'ing_fusion'], 'tabular'),
    'ing_hybrid': ([*ORD, '--formulation', 'ing_hybrid'], 'tabular'),
}

# ---- 10ma tanda: grilla d_model x cabezas por encoding + la cabeza del MLP (Exp. 1 y 2) ----
# La celda d32h4 ordinal ES feat_ordinal y d32h1 es camp_ordinal_h1; en embedding,
# d32h1/h2/h4 son pac20_feat_h1/h2/base, d64h1 es camp_d64h1 y d64h4 es pac20_feat_d64:
# se reutilizan tal cual (misma clave) y no se duplican. Los MLP: "el mejor MLP que
# pudimos", ancho / profundidad / dropout sobre el encoding ordinal.
EXPERIMENTOS |= {
    **{f'gc_o_d{d}h{h}': ([*ORD, '--d-model', str(d), '--n-head', str(h)], 'tabular')
       for d in (32, 64, 128) for h in (1, 2, 4, 8) if (d, h) not in {(32, 4), (32, 1)}},
    **{f'gc_e_d{d}h{h}': ([*PAC, '--d-model', str(d), '--n-head', str(h)], 'tabular')
       for d in (32, 64, 128) for h in (1, 2, 4, 8)
       if (d, h) not in {(32, 1), (32, 2), (32, 4), (64, 1), (64, 4)}},
    'mlp_ordinal':    (['--arch', 'mlp', *ORD], 'tabular'),
    'mlp_ord_h256':   (['--arch', 'mlp', *ORD, '--mlp-hidden', '256'], 'tabular'),
    'mlp_ord_h128':   (['--arch', 'mlp', *ORD, '--mlp-hidden', '128'], 'tabular'),
    'mlp_ord_ancho':  (['--arch', 'mlp', *ORD, '--mlp-hidden', '512,256'], 'tabular'),
    'mlp_ord_prof3':  (['--arch', 'mlp', *ORD, '--mlp-hidden', '256,128,64'], 'tabular'),
    'mlp_ord_prof4':  (['--arch', 'mlp', *ORD, '--mlp-hidden', '256,128,64,32'], 'tabular'),
    'mlp_ord_do2':    (['--arch', 'mlp', *ORD, '--dropout', '0.2'], 'tabular'),
    'mlp_ord_do3':    (['--arch', 'mlp', *ORD, '--dropout', '0.3'], 'tabular'),
    'mlp_ord_grande': (['--arch', 'mlp', *ORD, '--mlp-hidden', '1024,256', '--dropout', '0.2'], 'tabular'),
    'mlp_ord_mini':   (['--arch', 'mlp', *ORD, '--mlp-hidden', '64,32'], 'tabular'),
}

# ---- 11ra tanda (02/09): la grilla de cabezas crece (h16, d256) + grilla d_model x bloques ----
# (a) Una columna mas (16 cabezas) y una fila mas (d_model 256) en la grilla d_model x cabezas
#     de la 10ma tanda, para los dos encodings: queda 4 x 5 por encoding (8 celdas nuevas c/u).
# (b) Grilla d_model {32,64,128,256} x n_layer {1,2,4,8}, encoding ordinal (el elegido), donde
#     cada d_model usa LA CANTIDAD DE CABEZAS QUE MEJOR LE DIO en (a) — la que mas da en PR-AUC
#     de validacion medio (6 seeds), no la que veniamos usando. Como eso se sabe recien cuando
#     (a) termino, --n-head lleva el centinela MEJOR_H y se resuelve AL LANZAR cada corrida
#     leyendo resultados/ (resolver_extra); la fila n_layer=2 ES la celda ganadora de (a) y no
#     se vuelve a correr. Protocolo PAC 300/20 en todo. 28 configs nuevas x 6 seeds = 168 corridas.
# Correr en la GPU (los gc_* van primero, la suite respeta el orden; lo ya hecho se saltea):
#     .venv/bin/python experimentos.py --only 'gc_*,gl_*'
# Hipotesis registradas ANTES de correr (10ma tanda: ordinal rinde con cabezas de ~8 dims,
# embedding es indiferente, d128 no supera a d32): (1) con ordinal, h16 mejora a d128 (cabeza
# de 8 dims) y empeora a d32 (cabezas de 2 dims); con embedding no mueve nada. (2) d256 no
# supera a d32/d64 en ningun encoding: la meseta sigue, con mas varianza. (3) En bloques,
# meseta o caida: l8 es peor que l2 en todos los d (10k filas, 13 tokens: no hay que aprender
# tan hondo) y ningun (d, l) supera a la elegida 32·4·2 por mas del desvio. Con 6 seeds y
# desvio ~0.03, "supera" quiere decir delta pareado > 0.02 en 5/6 seeds o mas.
H_GRILLA = (1, 2, 4, 8, 16)       # cabezas (columnas del heatmap)
D_GRILLA = (32, 64, 128, 256)     # d_model (filas)
L_GRILLA = (1, 2, 4, 8)           # bloques (columnas del heatmap de bloques)
MEJOR_H = 'MEJOR'                 # centinela de --n-head: se resuelve al lanzar (resolver_extra)
# celdas de la grilla de cabezas que ya corrieron con otro tag (misma clave canonica):
# prefijo de archivo en resultados/ (sin _seedNN), por encoding ('o' ordinal, 'e' embedding)
CELDAS_REUSADAS = {
    'o': {(32, 4): 'feat_ordinal_features_d32_h4_l2_linear_catordinal',
          (32, 1): 'camp_ordinal_h1_features_d32_h1_l2_linear_catordinal'},
    'e': {(32, 1): 'pac20_feat_h1_features_d32_h1_l2_linear', (32, 2): 'pac20_feat_h2_features_d32_h2_l2_linear',
          (32, 4): 'pac20_feat_base_features_d32_h4_l2_linear', (64, 1): 'camp_d64h1_features_d64_h1_l2_linear',
          (64, 4): 'pac20_feat_d64_features_d64_h4_l2_linear'},
}
_CELDAS_10MA = {(d, h) for d in (32, 64, 128) for h in (1, 2, 4, 8)}   # ya definidas arriba o reusadas
EXPERIMENTOS |= {
    **{f'gc_o_d{d}h{h}': ([*ORD, '--d-model', str(d), '--n-head', str(h)], 'tabular')
       for d in D_GRILLA for h in H_GRILLA if (d, h) not in _CELDAS_10MA},
    **{f'gc_e_d{d}h{h}': ([*PAC, '--d-model', str(d), '--n-head', str(h)], 'tabular')
       for d in D_GRILLA for h in H_GRILLA if (d, h) not in _CELDAS_10MA},
    **{f'gl_o_d{d}l{l}': ([*ORD, '--d-model', str(d), '--n-layer', str(l), '--n-head', MEJOR_H], 'tabular')
       for d in D_GRILLA for l in L_GRILLA if l != 2},
}


class GrillaIncompleta(RuntimeError):
    """Falta correr celdas de la grilla de cabezas (gc_*) para resolver el centinela MEJOR_H."""


def celda_grilla(enc, d, h):
    """Prefijo de archivo (sin _seedNN) de la celda (encoding, d_model, cabezas) de la grilla de cabezas."""
    fijo = CELDAS_REUSADAS[enc].get((d, h))
    if fijo:
        return fijo
    return f'gc_{enc}_d{d}h{h}_features_d{d}_h{h}_l2_linear' + ('_catordinal' if enc == 'o' else '')


def mejor_h(enc, d, seeds=6):
    """La cantidad de cabezas con mejor PR-AUC de validacion medio en la fila d_model de la grilla."""
    medias = {}
    for h in H_GRILLA:
        celda = celda_grilla(enc, d, h)
        vals = [json.loads(f.read_text())['val']['pr_auc']
                for f in RESULTADOS.glob(f'{celda}_seed4[2-7].json')]
        if len(vals) < seeds:
            raise GrillaIncompleta(f'{celda}: {len(vals)}/{seeds} corridas — correr primero gc_*')
        medias[h] = sum(vals) / len(vals)
    return max(medias, key=medias.get)


def resolver_extra(nombre, extra=None):
    """Los args del experimento con el centinela MEJOR_H reemplazado por la cantidad de cabezas
    que mejor dio en la grilla (para su encoding y d_model). Lee resultados/ en el momento."""
    extra = list(EXPERIMENTOS[nombre][0] if extra is None else extra)
    if MEJOR_H not in extra:
        return extra
    i = extra.index(MEJOR_H)
    assert extra[i - 1] == '--n-head', f'{nombre}: el centinela solo va en --n-head'
    d = int(extra[extra.index('--d-model') + 1])
    enc = 'o' if 'ordinal' in extra else 'e'
    extra[i] = str(mejor_h(enc, d))
    return extra


def resolver_device(arg):
    if arg != 'auto':
        return arg
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def chequear_gpu(device_arg, nombres):
    """Aborta si la familia texto correria en CPU sin pedirlo explicitamente."""
    device = resolver_device(device_arg)
    if device == 'cuda':
        import torch
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        return
    con_texto = [n for n in nombres if EXPERIMENTOS[n][1] == 'texto']
    if con_texto and device_arg == 'auto':
        raise SystemExit(
            "\nNO se detecto GPU (torch.cuda.is_available() = False) y la suite incluye la\n"
            f"familia 'texto' ({len(con_texto)} experimentos, 40-90 min POR CORRIDA en CPU).\n"
            "En la maquina con RTX 3070 esto suele significar que torch quedo instalado en\n"
            "version CPU. Solucion:\n"
            "    uv pip install --python .venv/bin/python --reinstall torch\n"
            "(sin el index de CPU; verificar con: .venv/bin/python -c 'import torch; print(torch.cuda.is_available())')\n"
            "Para correr en CPU a proposito: --device cpu | solo lo barato: --familia tabular"
        )
    print(f"Corriendo en {device} (familia texto excluida o CPU explicita)")


def nombre_esperado(nombre_exp, extra, seed):
    """El nombre de archivo que va a producir esta corrida (misma logica que btr.train)."""
    args = build_parser().parse_args(['--tag', nombre_exp, *extra])
    return run_name(args, seed)


def armar_plan(nombres, seeds):
    """(corridas pendientes, ya hechas). Las que llevan el centinela MEJOR_H entran siempre:
    su nombre de archivo depende de la grilla, asi que se resuelven (y se saltean) al lanzar."""
    resultados_dir = RESULTADOS
    plan, salteados = [], 0
    for nombre in nombres:
        extra, _ = EXPERIMENTOS[nombre]
        for seed in range(42, 42 + seeds):
            if MEJOR_H not in extra and (resultados_dir / f"{nombre_esperado(nombre, extra, seed)}.json").exists():
                salteados += 1
            else:
                plan.append((nombre, extra, seed))
    return plan, salteados


def correr(nombres, seeds, device, save_pesos, epochs):
    chequear_gpu(device, nombres)
    resultados_dir = RESULTADOS
    plan, salteados = armar_plan(nombres, seeds)
    print(f"Plan: {len(plan)} corridas ({salteados} ya hechas, salteadas)")

    fallidos = []
    for i, (nombre, extra, seed) in enumerate(plan, 1):
        if MEJOR_H in extra:
            try:
                extra = resolver_extra(nombre, extra)
            except GrillaIncompleta as e:
                raise SystemExit(f'\n{nombre}: no se puede resolver --n-head MEJOR: {e}')
            if (resultados_dir / f"{nombre_esperado(nombre, extra, seed)}.json").exists():
                print(f"\n[{i}/{len(plan)}] {nombre} seed {seed}: ya hecha (n_head resuelto = {extra[extra.index('--n-head') + 1]})")
                continue
        cmd = [sys.executable, '-m', 'btr.train', '--tag', nombre, '--seeds', '1',
               '--seed-start', str(seed), '--device', device, '--quiet', *extra]
        if save_pesos:
            cmd.append('--save-pesos')
        if epochs:
            cmd += ['--epochs', str(epochs)]
        print(f"\n[{i}/{len(plan)}] {nombre} seed {seed}: {' '.join(extra)}", flush=True)
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f"  !! {nombre} seed {seed} fallo (exit {result.returncode}), sigo con el resto")
            fallidos.append(f'{nombre}/seed{seed}')
    if fallidos:
        print(f"\nCorridas fallidas: {fallidos}")
    else:
        print("\nSuite completa sin errores. Ver resumen: python experimentos.py --resumen")


def resumen():
    """Agrupa salidas/resultados/*.json por configuracion y promedia entre seeds."""
    grupos = defaultdict(list)
    for path in sorted(RESULTADOS.glob('*.json')):
        data = json.loads(path.read_text())
        clave = re.sub(r'_seed\d+(_\d+)?$', '', data['nombre'])
        grupos[clave].append(data)
    if not grupos:
        print('No hay resultados todavia.')
        return
    filas = []
    for clave, runs in grupos.items():
        roc = [r['test']['roc_auc'] for r in runs]
        pr = [r['test']['pr_auc'] for r in runs]
        epocas = [len(r['historial']) for r in runs]
        filas.append((clave, len(runs), runs[0]['n_parametros'],
                      sum(roc) / len(roc), _std(roc), sum(pr) / len(pr), _std(pr),
                      sum(epocas) / len(epocas)))
    filas.sort(key=lambda f: -f[5])
    ancho = max(len(f[0]) for f in filas)
    print(f"{'configuracion':<{ancho}}  n  {'params':>9}  {'ROC-AUC test':>14}  {'PR-AUC test':>14}  epocas")
    for clave, n, params, roc_m, roc_s, pr_m, pr_s, ep in filas:
        print(f"{clave:<{ancho}}  {n}  {params:>9,}  {roc_m:.4f} ± {roc_s:.3f}  {pr_m:.4f} ± {pr_s:.3f}  {ep:5.1f}")


def _std(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def main():
    parser = argparse.ArgumentParser(description='Suite de experimentos del TP')
    parser.add_argument('--list', action='store_true', help='listar experimentos y salir')
    parser.add_argument('--resumen', action='store_true', help='tabla comparativa de resultados/')
    parser.add_argument('--only', default='', help="correr solo estos (separados por coma; admite comodines: 'gc_*,gl_*')")
    parser.add_argument('--plan', action='store_true', help='mostrar que corridas faltan y salir')
    parser.add_argument('--familia', choices=['tabular', 'texto'], help='correr solo una familia')
    parser.add_argument('--seeds', type=int, default=6)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--no-pesos', action='store_true', help='no guardar checkpoints en pesos/')
    parser.add_argument('--epochs', type=int, help='override de epocas (para pruebas rapidas)')
    args = parser.parse_args()

    if args.list:
        for nombre, (extra, familia) in EXPERIMENTOS.items():
            print(f"{nombre:<20} [{familia:7}] {' '.join(extra)}")
        return
    if args.resumen:
        resumen()
        return
    import fnmatch
    pedidos = [n.strip() for n in args.only.split(',') if n.strip()]
    nombres = []
    for n in pedidos:
        con_comodin = [m for m in EXPERIMENTOS if fnmatch.fnmatch(m, n)] if '*' in n else [n]
        nombres += [m for m in con_comodin if m not in nombres]
    nombres = nombres or list(EXPERIMENTOS)
    desconocidos = [n for n in nombres if n not in EXPERIMENTOS]
    if desconocidos:
        raise SystemExit(f'experimentos desconocidos: {desconocidos} (ver --list)')
    if args.familia:
        nombres = [n for n in nombres if EXPERIMENTOS[n][1] == args.familia]
    if args.plan:
        plan, salteados = armar_plan(nombres, args.seeds)
        por_exp = defaultdict(int)
        for nombre, _, _ in plan:
            por_exp[nombre] += 1
        for nombre, k in por_exp.items():
            print(f"{nombre:<20} {k} corridas   {' '.join(EXPERIMENTOS[nombre][0])}")
        print(f"Plan: {len(plan)} corridas ({salteados} ya hechas, salteadas)")
        return
    correr(nombres, args.seeds, args.device, not args.no_pesos, args.epochs)


if __name__ == '__main__':
    main()
