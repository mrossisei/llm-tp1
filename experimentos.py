"""Suite curada de experimentos del TP (propuesta.md 7.4).

USO EN LA MAQUINA CON GPU (dos lineas):
    .venv/bin/python experimentos.py              # corre TODA la suite (52 configs x 6 seeds)
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

# ---- segunda tanda (16/08, disenada a partir del analisis de la primera tanda GPU:
# ver analisis.md). Las tabulares usan el protocolo paciencia 20 / tope 300 epocas,
# que gano o empato en 21/24 configs de la primera tanda (base de comparacion: los
# grupos pac20_* que corrio Matias). La familia texto queda en paciencia 8: con 20
# empeora en test (la seleccion por val sobreajusta, ver analisis.md).
PAC = ['--patience', '20', '--epochs', '300']
EXPERIMENTOS |= {
    # grilla "campeon": combinar los ganadores individuales de las ablaciones
    # (1 cabeza grande, d_model 64, 4 bloques)
    'camp_d64h1':       (['--d-model', '64', '--n-head', '1', *PAC], 'tabular'),
    'camp_d64l4':       (['--d-model', '64', '--n-layer', '4', *PAC], 'tabular'),
    'camp_h1l4':        (['--n-head', '1', '--n-layer', '4', *PAC], 'tabular'),
    'camp_d64h1l4':     (['--d-model', '64', '--n-head', '1', '--n-layer', '4', *PAC], 'tabular'),
    # causal hecho bien: el feat_causal original degeneraba (CLS en posicion 0 solo
    # se ve a si mismo -> p constante, ROC 0.500 medido); con el CLS al final el
    # experimento "¿importa la bidireccionalidad?" por fin se puede responder
    'feat_causal_last': (['--causal', '--cls-position', 'last', *PAC], 'tabular'),
    # encodings de categoricas (el "modular/columnar" del companero = hashing con
    # modulo; embedding por columna es lo que ya haciamos)
    'feat_target':      (['--cat-encoding', 'target', *PAC], 'tabular'),
    'feat_freq':        (['--cat-encoding', 'freq', *PAC], 'tabular'),
    'feat_hash8':       (['--cat-encoding', 'hashing', '--hash-buckets', '8', *PAC], 'tabular'),
    'mlp_onehot':       (['--arch', 'mlp', '--cat-encoding', 'onehot', *PAC], 'tabular'),
    # features descartados en el EDA, de vuelta (volumen, package, n_ingredients):
    # verificacion empirica de la redundancia que el EDA declaro
    'feat_extras':      (['--extra-features', 'all', *PAC], 'tabular'),
    # multi-task con cart como label auxiliar (junior_proposals.md #2)
    'feat_cartaux01':   (['--cart-aux', '0.1', *PAC], 'tabular'),
    'feat_cartaux03':   (['--cart-aux', '0.3', *PAC], 'tabular'),
    'feat_cartaux05':   (['--cart-aux', '0.5', *PAC], 'tabular'),
    # listwise enriquecido con la torre de texto (junior_proposals.md #1); paciencia
    # 20 porque listwise fue el mas beneficiado por mas paciencia (+0.041)
    'listwise_texto':   (['--arch', 'listwise', '--listwise-texto', *PAC], 'texto'),
    # texto corto: la senal vive en el sufijo del titulo (<= 81 chars); 96 chars
    # cubren el titulo entero y la atencion pasa de 257^2 a 97^2 (~7x mas barata)
    'text_len96':       (['--formulation', 'text', '--max-text-len', '96'], 'texto'),

    # ---- el estado como CAMPO separado (idea de Fer, 16/08) ----
    # (a) completar el 2x2 {token parseado si/no} x {sufijo en el texto si/no}:
    # full=ambos canales, sin_regex=solo texto, intrinseco=ninguno; faltaba
    # "solo el campo, texto limpio" — la separacion prolija que propuso Fer
    'hybrid_status_campo': (['--formulation', 'hybrid', '--strip-status'], 'texto'),
    'tower_status_campo':  (['--arch', 'tower', '--strip-status'], 'texto'),
    # (b) encoding del campo, incluso CON ORDEN. El orden defendible se deriva
    # del BTR de train (ordinal = solo rango, target = rango + magnitud); un
    # orden semantico a mano es indefendible (EDA 2.3: el wording no predice el
    # tier). --cat-feature-encoding lo aplica SOLO a listing_status.
    'feat_ordinal':        (['--cat-encoding', 'ordinal', *PAC], 'tabular'),
    'feat_status_ordinal': (['--cat-feature-encoding', 'listing_status=ordinal', *PAC], 'tabular'),
    'feat_status_target':  (['--cat-feature-encoding', 'listing_status=target', *PAC], 'tabular'),

    # ---- 3ra tanda (16/08): analisis de la 2da + revision externa (analisis.md 5-6) ----
    # el campeon nuevo es ordinal GLOBAL (0.824): ¿se combina con los ganadores de capacidad?
    'camp_ordinal_h1':      (['--cat-encoding', 'ordinal', '--n-head', '1', *PAC], 'tabular'),
    'camp_ordinal_l4':      (['--cat-encoding', 'ordinal', '--n-layer', '4', *PAC], 'tabular'),
    'camp_ordinal_d64h1l4': (['--cat-encoding', 'ordinal', '--d-model', '64', '--n-head', '1',
                              '--n-layer', '4', *PAC], 'tabular'),
    # hora/dia del timestamp (revision externa los sugirio; el EDA dice ruido -> verificar)
    'feat_tiempo':          (['--extra-features', 'hour,dow', *PAC], 'tabular'),
    # 5b de la revision externa: tokenizacion WORD-level (la que "recomendaron" en clase)
    'text_words':           (['--formulation', 'text', '--text-tokens', 'words',
                              '--max-text-len', '64'], 'texto'),
    # ...y el resumen del texto como UN token de la secuencia tabular: la atencion cruza
    # texto-features al nivel del resumen, sin que 256 chars diluyan (el mal del hybrid)
    'fusion_base':          (['--formulation', 'fusion'], 'texto'),
    'fusion_words':         (['--formulation', 'fusion', '--text-tokens', 'words',
                              '--max-text-len', '64'], 'texto'),
    # embeddings pre-entrenados (skipgram sobre el corpus de train) vs end-to-end:
    # la comparacion clase 1 vs clase 2 que pedia la revision externa
    'fusion_words_w2v':     (['--formulation', 'fusion', '--text-tokens', 'words',
                              '--max-text-len', '64', '--w2v-init'], 'texto'),
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
    nombres = [n.strip() for n in args.only.split(',') if n.strip()] or list(EXPERIMENTOS)
    desconocidos = [n for n in nombres if n not in EXPERIMENTOS]
    if desconocidos:
        raise SystemExit(f'experimentos desconocidos: {desconocidos} (ver --list)')
    if args.familia:
        nombres = [n for n in nombres if EXPERIMENTOS[n][1] == args.familia]
    correr(nombres, args.seeds, args.device, not args.no_pesos, args.epochs)


if __name__ == '__main__':
    main()
