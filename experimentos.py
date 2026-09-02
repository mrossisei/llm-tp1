"""Suite curada de experimentos del TP: todo lo que sostiene la presentacion.

USO EN LA MAQUINA CON GPU (dos lineas):
    .venv/bin/python experimentos.py              # corre TODA la suite (lo que falte)
    .venv/bin/python experimentos.py --resumen    # tabla comparativa: media +- desvio por config

Garantias de la suite:
- Usa la GPU automaticamente (--device auto -> cuda si esta disponible). Si la familia
  "texto" (fine-tuning de un preentrenado) fuera a correr en CPU, ABORTA con
  instrucciones; para forzar CPU a proposito: --device cpu.
- Es RESUMIBLE: cada (experimento, seed) que ya tiene su JSON en salidas/resultados/ se
  saltea. Si se corta a la mitad, volver a correr la misma linea continua donde quedo.
- Guarda salidas/pesos/ por defecto (checkpoints recargables para analisis posteriores,
  p. ej. mapas de atencion); desactivable con --no-pesos.
- Si un experimento falla, sigue con el resto y lo reporta al final.

Otras opciones: --list, --plan, --only a,b (admite comodines: 'gc_*'), --familia
tabular|texto, --seeds N, --epochs N (pruebas).

Cada bloque dice que diapositiva/figura sostiene. Los experimentos son BARRIDOS: cada
uno mueve uno o dos ejes sobre la configuracion base (d_model 32, 4 cabezas, 2 bloques,
dropout 0.1, AdamW lr 1e-3, weight decay 0.01, batch 256, paciencia 20 / tope 300
epocas, 6 seeds) y las celdas ya corridas en tandas previas se reutilizan por su
nombre canonico, sin duplicar corridas.
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

PAC = ['--patience', '20', '--epochs', '300']          # el protocolo de entrenamiento
ORD = ['--cat-encoding', 'ordinal', *PAC]              # la config exacta del modelo final
EMB = [*PAC]                                           # idem con embeddings aprendidos

# nombre -> (argumentos extra, familia)  [familia: 'tabular' barata | 'texto' GPU]
# La suite sigue el orden de la presentacion y la regla de Fer: en cada experimento se elige la
# configuracion que mas PR-AUC de validacion da, y el experimento siguiente se corre sobre ella.
EXPERIMENTOS = {
    # ---- la base original y sus ablaciones (grupos pac20_*, corridos por Matias con el protocolo
    # 300/20): las celdas "embedding" que reutiliza la grilla de capacidad y las hipotesis
    # refutadas de "Desafios" ----
    'pac20_feat_base':       ([*EMB], 'tabular'),
    'pac20_feat_h1':         (['--n-head', '1', *EMB], 'tabular'),
    'pac20_feat_h2':         (['--n-head', '2', *EMB], 'tabular'),
    'pac20_feat_d64':        (['--d-model', '64', *EMB], 'tabular'),
    'camp_d64h1':            (['--d-model', '64', '--n-head', '1', *EMB], 'tabular'),
    'pac20_feat_intrinseco': (['--drop-features', 'listing_status', *EMB], 'tabular'),  # sin estado: ~0.16
    'pac20_feat_bins':       (['--numeric-mode', 'bins', *EMB], 'tabular'),     # bins por cuantiles
    'pac20_feat_pos':        (['--positional', *EMB], 'tabular'),               # positional encoding
    'pac20_feat_causal':     (['--causal', *EMB], 'tabular'),                   # causal: degenera (CLS en 0)
    'pac20_feat_mean':       (['--pooling', 'mean', *EMB], 'tabular'),          # mean pooling vs CLS
    'pac20_feat_posweight':  (['--pos-weight', *EMB], 'tabular'),               # pesar la clase positiva
    'feat_causal_last':      (['--causal', '--cls-position', 'last', *PAC], 'tabular'),  # causal bien hecho
    # el punto de partida ordinal (32·4·2) y su celda de 1 cabeza (grilla de capacidad)
    'feat_ordinal':     ([*ORD], 'tabular'),
    'camp_ordinal_h1':  ([*ORD, '--n-head', '1'], 'tabular'),
    # el control de ingredientes: solo la lista, sin las features (~ azar)
    'ing_solo':         (['--formulation', 'ing', *PAC], 'tabular'),
}

# ---- Robustez del modelo (diapositivas Robustez y Modelo final; corridas sobre 32·4·2 —
# se repiten sobre la ganadora cuando cierre la cadena de experimentos) ----
EXPERIMENTOS |= {
    'curva_frac25': ([*ORD, '--train-frac', '0.25'], 'tabular'),
    'curva_frac50': ([*ORD, '--train-frac', '0.5'], 'tabular'),
    'curva_frac75': ([*ORD, '--train-frac', '0.75'], 'tabular'),
    'robu_init43': ([*ORD, '--init-seed', '43'], 'tabular'),
    'robu_init44': ([*ORD, '--init-seed', '44'], 'tabular'),
    'robu_init45': ([*ORD, '--init-seed', '45'], 'tabular'),
    'robu_init46': ([*ORD, '--init-seed', '46'], 'tabular'),
    'robu_init47': ([*ORD, '--init-seed', '47'], 'tabular'),
    'cv5_fold0': ([*ORD, '--cv-k', '5', '--cv-fold', '0'], 'tabular'),
    'cv5_fold1': ([*ORD, '--cv-k', '5', '--cv-fold', '1'], 'tabular'),
    'cv5_fold2': ([*ORD, '--cv-k', '5', '--cv-fold', '2'], 'tabular'),
    'cv5_fold3': ([*ORD, '--cv-k', '5', '--cv-fold', '3'], 'tabular'),
    'cv5_fold4': ([*ORD, '--cv-k', '5', '--cv-fold', '4'], 'tabular'),
}

# ---- Exp. 1 CAPACIDAD: grilla d_model x cabezas por encoding (10ma y 11ra tandas) ----
# La celda d32h4 ordinal ES feat_ordinal y d32h1 es camp_ordinal_h1; en embedding,
# d32h1/h2/h4 son pac20_feat_h1/h2/base, d64h1 es camp_d64h1 y d64h4 es pac20_feat_d64.
H_GRILLA = (1, 2, 4, 8, 16)       # cabezas (columnas del heatmap)
D_GRILLA = (32, 64, 128, 256)     # d_model (filas)
L_GRILLA = (1, 2, 4, 8)           # bloques (columnas del heatmap de bloques)
MEJOR_H = 'MEJOR'                 # centinela de --n-head: se resuelve al lanzar (resolver_extra)
CELDAS_REUSADAS = {
    'o': {(32, 4): 'feat_ordinal_features_d32_h4_l2_linear_catordinal',
          (32, 1): 'camp_ordinal_h1_features_d32_h1_l2_linear_catordinal'},
    'e': {(32, 1): 'pac20_feat_h1_features_d32_h1_l2_linear', (32, 2): 'pac20_feat_h2_features_d32_h2_l2_linear',
          (32, 4): 'pac20_feat_base_features_d32_h4_l2_linear', (64, 1): 'camp_d64h1_features_d64_h1_l2_linear',
          (64, 4): 'pac20_feat_d64_features_d64_h4_l2_linear'},
}
EXPERIMENTOS |= {
    **{f'gc_o_d{d}h{h}': ([*ORD, '--d-model', str(d), '--n-head', str(h)], 'tabular')
       for d in D_GRILLA for h in H_GRILLA if (d, h) not in CELDAS_REUSADAS['o']},
    **{f'gc_e_d{d}h{h}': ([*EMB, '--d-model', str(d), '--n-head', str(h)], 'tabular')
       for d in D_GRILLA for h in H_GRILLA if (d, h) not in CELDAS_REUSADAS['e']},
}

# ---- Exp. 2 PROFUNDIDAD: d_model x bloques, cada d con las cabezas que mejor le dieron en el
# Exp. 1 (--n-head MEJOR se resuelve al lanzar leyendo la grilla: resolver_extra). La fila de 2
# bloques es la celda ganadora del Exp. 1 y no se repite. ----
EXPERIMENTOS |= {
    **{f'gl_o_d{d}l{l}': ([*ORD, '--d-model', str(d), '--n-layer', str(l), '--n-head', MEJOR_H], 'tabular')
       for d in D_GRILLA for l in L_GRILLA if l != 2},
}

# ---- LA MEJOR ARQUITECTURA hasta el Exp. 2 (la celda de mas PR-AUC de validacion de la grilla de
# bloques: d 32, 16 cabezas, 4 bloques, 0.841). Los experimentos que siguen se corren sobre ella;
# si alguno la supera, se actualiza esta linea y se relanza lo que sigue (la suite reutiliza las
# corridas equivalentes por nombre canonico y solo corre lo que falte). ----
MEJOR_HL = ['--n-head', '16', '--n-layer', '4', *PAC]            # cabezas y bloques de la ganadora
MEJOR_ARQ = ['--cat-encoding', 'ordinal', '--d-model', '32', *MEJOR_HL]
MEJOR_SIN_ENC = ['--d-model', '32', *MEJOR_HL]                     # la misma arquitectura, con embeddings

# ---- Exp. 3 ENCODING x d_model: {ordinal, embedding, target} x d {32, 64, 128}, con las cabezas y
# bloques de la ganadora (d 16 queda afuera: con 16 cabezas serian cabezas de 1 dimension).
# Frecuencia y hashing ya colapsaron sobre 32·4·2 (0.18 y 0.51) y no se repiten. Hipotesis:
# el orden de los encodings no depende de la capacidad; ordinal arriba en las tres columnas. ----
D_ENC = (32, 64, 128)
ENC_ARGS = {'ordinal': ['--cat-encoding', 'ordinal'], 'embedding': [], 'target': ['--cat-encoding', 'target']}
EXPERIMENTOS |= {
    f'enc_{enc}_d{d}': ([*ENC_ARGS[enc], '--d-model', str(d), *MEJOR_HL], 'tabular')
    for enc in ENC_ARGS for d in D_ENC
}

# ---- Exp. 4 PRE-ENTRENAMIENTO MLM: epocas {0, 5, 10, 20, 40} x encoding, sobre la ganadora.
# Hipotesis: nada fuera del ruido a ninguna cantidad de epocas (sobre 32·4·2 dio eso). ----
MLM_EPOCAS = (0, 5, 10, 20, 40)
EXPERIMENTOS |= {
    **{f'mlm_o_{e}': ([*MEJOR_ARQ, '--pretrain-mlm', str(e)], 'tabular') for e in MLM_EPOCAS if e},
    **{f'mlm_e_{e}': ([*MEJOR_SIN_ENC, '--pretrain-mlm', str(e)], 'tabular') for e in MLM_EPOCAS if e},
}

# ---- Exp. 5 OPTIMIZACION: learning rate {1e-4, 3e-4, 1e-3} x batch {64, 128, 256}, sobre la
# ganadora (1e-3 x 256 es la ganadora misma). Hipotesis: meseta; lr 1e-4 con batch 256 es el mas
# lento (pocos pasos por epoca) y el unico que podria no llegar dentro de la paciencia. ----
LRS = ('0.0001', '0.0003', '0.001')
BATCHES = ('64', '128', '256')
EXPERIMENTOS |= {
    f'opt_lr{lr}_bs{bs}': ([*MEJOR_ARQ, '--lr', lr, '--batch-size', bs], 'tabular')
    for lr in LRS for bs in BATCHES if (lr, bs) != ('0.001', '256')
}

# ---- Exp. 6 INGREDIENTES sobre la ganadora: el encoder de conjunto ([ING] + un token por
# ingrediente, SIN positional encoding y SIN mascara causal) en tres tamanos. Hipotesis: delta ~ 0
# (los ingredientes son la categoria disfrazada; ing_solo ~ azar); el grande empeora por varianza. ----
EXPERIMENTOS |= {
    'ing_chico':  ([*MEJOR_ARQ, '--formulation', 'ing_fusion', '--ing-d-model', '16', '--ing-head', '2',
                    '--ing-layer', '1'], 'tabular'),
    'ing_base':   ([*MEJOR_ARQ, '--formulation', 'ing_fusion'], 'tabular'),     # d y cabezas del modelo, 1 bloque
    'ing_grande': ([*MEJOR_ARQ, '--formulation', 'ing_fusion', '--ing-d-model', '64', '--ing-head', '8',
                    '--ing-layer', '2'], 'tabular'),
}

# ---- Exp. 7 TRANSFER LEARNING sobre la ganadora: el TITULO sin badge embebido por un preentrenado
# (eda/embed_titulos.py -> salidas/embeddings/, congelado), en tres tamanos, y el control
# solo-titulo. El fine-tuning (--text-emb-finetune, el encoder de 22M en el grafo) es caro y NO se
# repite: la presentacion usa las 3 seeds que corrieron sobre 32·4·2 (grupo tl_minilm_ft_*_tembft-titulo).
# Hipotesis: sobre 32·4·2 el titulo RESTO (-0.02/-0.04); se espera lo mismo. ----
EMB_TITULO = 'salidas/embeddings/titulo_{}.npy'
EXPERIMENTOS |= {
    'tl_minilm':    ([*MEJOR_ARQ, '--text-emb', EMB_TITULO.format('minilm')], 'tabular'),
    'tl_mpnet':     ([*MEJOR_ARQ, '--text-emb', EMB_TITULO.format('mpnet')], 'tabular'),
    'tl_bge':       ([*MEJOR_ARQ, '--text-emb', EMB_TITULO.format('bge')], 'tabular'),
    'tl_bge_solo':  ([*MEJOR_SIN_ENC, '--text-emb', EMB_TITULO.format('bge'), '--drop-features', 'all'], 'tabular'),
}

# ---- Exp. 8 TIEMPO sobre la ganadora: hora del dia y dia de la semana, ciclicas (btr/data.py):
# 'ciclico' = (sin, cos) del angulo, un token por variable; 'categorico' = 24 + 7 niveles con el
# encoding de las demas. Hipotesis: delta ~ 0 con las dos (el EDA mostro que el timestamp es ruido);
# si hubiera diferencia, la ciclica es la mas estable. ----
EXPERIMENTOS |= {
    'tiempo_ciclico': ([*MEJOR_ARQ, '--tiempo', 'ciclico'], 'tabular'),
    'tiempo_cat':     ([*MEJOR_ARQ, '--tiempo', 'categorico'], 'tabular'),
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
            f"familia 'texto' ({len(con_texto)} experimentos con un encoder preentrenado en el grafo).\n"
            "En la maquina con RTX 3070 esto suele significar que torch quedo instalado en\n"
            "version CPU. Solucion:\n"
            "    uv pip install --python .venv/bin/python --reinstall torch\n"
            "(sin el index de CPU; verificar con: .venv/bin/python -c 'import torch; print(torch.cuda.is_available())')\n"
            "Para correr en CPU a proposito: --device cpu | solo lo barato: --familia tabular"
        )
    print(f"Corriendo en {device} (familia texto excluida o CPU explicita)")


def canon_mejor():
    """El nombre canonico (sin tag ni seed) de la mejor arquitectura: para encontrar su corrida de
    referencia entre los grupos ya corridos, cualquiera sea su tag."""
    nombre = nombre_esperado('x', MEJOR_ARQ, 42)
    return nombre[len('x_'):-len('_seed42')]


def grupo_canonico(canon, seed=42):
    """Prefijo de archivo (con su tag) de un grupo ya corrido cuyo nombre canonico es `canon`, o None.

    Dos tags distintos con el mismo nombre canonico son la MISMA configuracion (el nombre lo
    construye run_name a partir de todos los argumentos que importan): asi la suite reutiliza
    celdas corridas en tandas previas sin duplicarlas.
    """
    for f in RESULTADOS.glob(f'*_{canon}_seed{seed}.json'):
        return f.name[:-len(f'_seed{seed}.json')]
    return None


def corrida_hecha(nombre_exp, extra, seed):
    """True si esta (config, seed) ya corrio: con este tag o con otro tag y el mismo nombre canonico."""
    esperado = nombre_esperado(nombre_exp, extra, seed)
    if (RESULTADOS / f'{esperado}.json').exists():
        return True
    canon = esperado[len(nombre_exp) + 1:-len(f'_seed{seed}')]
    return grupo_canonico(canon, seed) is not None


def nombre_esperado(nombre_exp, extra, seed):
    """El nombre de archivo que va a producir esta corrida (misma logica que btr.train)."""
    args = build_parser().parse_args(['--tag', nombre_exp, *extra])
    return run_name(args, seed)


def armar_plan(nombres, seeds):
    """(corridas pendientes, ya hechas). Las que llevan el centinela MEJOR_H entran siempre:
    su nombre de archivo depende de la grilla, asi que se resuelven (y se saltean) al lanzar."""
    plan, salteados = [], 0
    for nombre in nombres:
        extra, _ = EXPERIMENTOS[nombre]
        for seed in range(42, 42 + seeds):
            if MEJOR_H not in extra and corrida_hecha(nombre, extra, seed):
                salteados += 1
            else:
                plan.append((nombre, extra, seed))
    return plan, salteados


def correr(nombres, seeds, device, save_pesos, epochs):
    chequear_gpu(device, nombres)
    plan, salteados = armar_plan(nombres, seeds)
    print(f"Plan: {len(plan)} corridas ({salteados} ya hechas, salteadas)")

    fallidos = []
    for i, (nombre, extra, seed) in enumerate(plan, 1):
        if MEJOR_H in extra:
            try:
                extra = resolver_extra(nombre, extra)
            except GrillaIncompleta as e:
                raise SystemExit(f'\n{nombre}: no se puede resolver --n-head MEJOR: {e}')
            if corrida_hecha(nombre, extra, seed):
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
