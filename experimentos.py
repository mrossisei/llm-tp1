"""Suite curada de experimentos del TP (propuesta.md 7.4).

Uso tipico en la maquina con GPU:
    .venv/bin/python experimentos.py                  # corre toda la suite (3 seeds c/u)
    .venv/bin/python experimentos.py --list           # ver que corre cada experimento
    .venv/bin/python experimentos.py --only text_base,hybrid_sin_regex
    .venv/bin/python experimentos.py --resumen        # tabla comparativa de resultados/

Cada experimento invoca btr.train con --tag <nombre>; los JSON quedan en
resultados/ y el resumen los agrupa por configuracion (promedio +- desvio entre
seeds). La familia "texto" es la costosa (secuencia ~257): en CPU son 40-90 min
por corrida, en GPU minutos. La familia tabular corre en segundos donde sea.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# nombre -> (argumentos extra, familia)  [familia: 'tabular' barata | 'texto' GPU]
EXPERIMENTOS = {
    # formulaciones y arquitecturas (la comparacion central del TP)
    'feat_base':        (['--formulation', 'features'], 'tabular'),
    'mlp_base':         (['--arch', 'mlp'], 'tabular'),
    'listwise_base':    (['--arch', 'listwise'], 'tabular'),
    'text_base':        (['--formulation', 'text'], 'texto'),
    'hybrid_full':      (['--formulation', 'hybrid'], 'texto'),
    'tower_base':       (['--arch', 'tower'], 'texto'),
    # ¿el transformer redescubre la senal del texto sin el regex?
    'hybrid_sin_regex': (['--formulation', 'hybrid', '--drop-features', 'listing_status'], 'texto'),
    'tower_sin_regex':  (['--arch', 'tower', '--drop-features', 'listing_status'], 'texto'),
    # familia "producto nuevo" (sin informacion de estado/popularidad, ver propuesta 2.3.1)
    'feat_intrinseco':  (['--formulation', 'features', '--drop-features', 'listing_status'], 'tabular'),
    'text_intrinseco':  (['--formulation', 'text', '--strip-status'], 'texto'),
    'hybrid_intrinseco': (['--formulation', 'hybrid', '--strip-status',
                           '--drop-features', 'listing_status'], 'texto'),
    # codificacion de numericas (la U invertida del precio)
    'feat_bins':        (['--numeric-mode', 'bins'], 'tabular'),
    # ablaciones de arquitectura sobre la formulacion tabular
    'feat_pos':         (['--positional'], 'tabular'),
    'feat_causal':      (['--causal'], 'tabular'),
    'feat_mean':        (['--pooling', 'mean'], 'tabular'),
    'feat_posweight':   (['--pos-weight'], 'tabular'),
    # capacidad (d_model / capas / heads) — grilla chica estilo paper
    'feat_d8':          (['--d-model', '8'], 'tabular'),
    'feat_d16':         (['--d-model', '16'], 'tabular'),
    'feat_d64':         (['--d-model', '64'], 'tabular'),
    'feat_l1':          (['--n-layer', '1'], 'tabular'),
    'feat_l4':          (['--n-layer', '4'], 'tabular'),
    'feat_h1':          (['--n-head', '1'], 'tabular'),
    'feat_h2':          (['--n-head', '2'], 'tabular'),
    'text_d64':         (['--formulation', 'text', '--d-model', '64'], 'texto'),
}


def correr(nombres, seeds, device, save_pesos, epochs):
    fallidos = []
    for i, nombre in enumerate(nombres, 1):
        extra, familia = EXPERIMENTOS[nombre]
        cmd = [sys.executable, '-m', 'btr.train', '--tag', nombre, '--seeds', str(seeds),
               '--device', device, '--quiet', *extra]
        if save_pesos:
            cmd.append('--save-pesos')
        if epochs:
            cmd += ['--epochs', str(epochs)]
        print(f"\n[{i}/{len(nombres)}] {nombre} ({familia}): {' '.join(cmd[3:])}", flush=True)
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f"  !! {nombre} fallo (exit {result.returncode}), sigo con el resto")
            fallidos.append(nombre)
    if fallidos:
        print(f"\nExperimentos fallidos: {fallidos}")
    else:
        print("\nSuite completa sin errores. Ver resumen: python experimentos.py --resumen")


def resumen():
    """Agrupa resultados/*.json por configuracion y promedia entre seeds."""
    grupos = defaultdict(list)
    for path in sorted((REPO_ROOT / 'resultados').glob('*.json')):
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
    parser.add_argument('--only', default='', help='correr solo estos (separados por coma)')
    parser.add_argument('--familia', choices=['tabular', 'texto'], help='correr solo una familia')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--save-pesos', action='store_true')
    parser.add_argument('--epochs', type=int, help='override de epocas (para pruebas rapidas)')
    args = parser.parse_args()

    if args.list:
        for nombre, (extra, familia) in EXPERIMENTOS.items():
            print(f"{nombre:<20} [{familia:7}] {' '.join(extra)}")
        return
    if args.resumen:
        resumen()
        return
    nombres = [n.strip() for n in args.only.split(',') if n.strip()] or list(EXPERIMENTOS)
    desconocidos = [n for n in nombres if n not in EXPERIMENTOS]
    if desconocidos:
        raise SystemExit(f'experimentos desconocidos: {desconocidos} (ver --list)')
    if args.familia:
        nombres = [n for n in nombres if EXPERIMENTOS[n][1] == args.familia]
    correr(nombres, args.seeds, args.device, args.save_pesos, args.epochs)


if __name__ == '__main__':
    main()
