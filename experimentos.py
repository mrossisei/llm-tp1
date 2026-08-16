"""Suite curada de experimentos del TP (propuesta.md 7.4).

USO EN LA MAQUINA CON GPU (dos lineas):
    .venv/bin/python experimentos.py              # corre TODA la suite (24 configs x 3 seeds)
    .venv/bin/python experimentos.py --resumen    # tabla comparativa: media +- desvio por config

Garantias de la suite:
- Usa la GPU automaticamente (--device auto -> cuda si esta disponible). Si la familia
  "texto" fuera a correr en CPU, ABORTA con instrucciones (evita 20+ horas de CPU por
  un torch mal instalado); para forzar CPU a proposito: --device cpu.
- Es RESUMIBLE: cada (experimento, seed) que ya tiene su JSON en resultados/ se
  saltea. Si se corta a la mitad, volver a correr la misma linea continua donde quedo.
- Guarda pesos/ por defecto (checkpoints recargables para analisis posteriores, p. ej.
  mapas de atencion); desactivable con --no-pesos.
- Si un experimento falla, sigue con el resto y lo reporta al final.

Otras opciones: --list, --only a,b, --familia tabular|texto, --seeds N, --epochs N (pruebas).
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
    # familia "producto nuevo" (sin informacion de estado/popularidad, propuesta 2.3.1)
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


def correr(nombres, seeds, device, save_pesos, epochs):
    chequear_gpu(device, nombres)
    resultados_dir = REPO_ROOT / 'resultados'
    plan, salteados = [], 0
    for nombre in nombres:
        extra, _ = EXPERIMENTOS[nombre]
        for seed in range(42, 42 + seeds):
            if (resultados_dir / f"{nombre_esperado(nombre, extra, seed)}.json").exists():
                salteados += 1
            else:
                plan.append((nombre, extra, seed))
    print(f"Plan: {len(plan)} corridas ({salteados} ya hechas, salteadas)")

    fallidos = []
    for i, (nombre, extra, seed) in enumerate(plan, 1):
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
    nombres = [n.strip() for n in args.only.split(',') if n.strip()] or list(EXPERIMENTOS)
    desconocidos = [n for n in nombres if n not in EXPERIMENTOS]
    if desconocidos:
        raise SystemExit(f'experimentos desconocidos: {desconocidos} (ver --list)')
    if args.familia:
        nombres = [n for n in nombres if EXPERIMENTOS[n][1] == args.familia]
    correr(nombres, args.seeds, args.device, not args.no_pesos, args.epochs)


if __name__ == '__main__':
    main()
