"""Genera panel.html: el laboratorio interactivo de experimentos.

    .venv/bin/python panel.py          # regenera panel.html desde el estado real del repo

La pagina (estatica, autocontenida) permite componer una configuracion eligiendo
cada decision (arquitectura, features, encodings, capacidad, validacion) y muestra:
  - el comando exacto para correrla (o el --only de la suite si ya esta curada),
  - si la combinacion ya esta soportada por el codigo o requiere implementacion
    (en ese caso da un spec para pegar en el chat y pedirla),
  - los resultados si ya se corrio: TODAS las metricas (media +- desvio entre
    seeds) y las curvas de entrenamiento de cualquier metrica,
  - el ranking de todo lo corrido hasta ahora.

Los datos se embeben al generar: correr este script de nuevo despues de cada
tanda de experimentos (p. ej. tras la suite en la GPU) actualiza la pagina.

La identidad de una configuracion es su "clave canonica": la misma funcion existe
en Python (canon(), para suite y resultados embebidos) y en JS (canonKey(), para
lo que se elige en la pagina). La pagina se auto-verifica al cargar: recalcula en
JS la clave de cada config de la suite y compara con la embebida; si divergieran,
muestra un banner rojo (guardia contra drift entre las dos implementaciones).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from btr.data import (CAT_FEATURES, EXTRA_ALL, EXTRA_FEATURES, NUM_FEATURES, Preprocessor,
                      load_dataset, split_by_query)
from btr.train import build_parser
from experimentos import EXPERIMENTOS

REPO = Path(__file__).resolve().parent


def ffloat(x):
    return f'{float(x):.10f}'.rstrip('0').rstrip('.')


def _catfe_pares(c, catenc_global):
    """Overrides por-feature reales (modo distinto del global), desde el string del flag."""
    pares = {}
    for par in str(c.get('cat_feature_encoding', '') or '').split(','):
        par = par.strip()
        if par and '=' in par:
            f, m = (s.strip() for s in par.split('=', 1))
            if m != catenc_global:
                pares[f] = m
    return pares


def canon(c):
    """Clave canonica de una config (dict con los nombres de argparse).

    Normaliza los campos que no aplican a la arquitectura elegida (los pone en
    '-') para que dos configs que producen EL MISMO modelo tengan la misma clave
    aunque difieran en flags irrelevantes. Espejada en JS como canonKey().
    """
    arch = c['arch']
    form = c.get('formulation', 'features') if arch == 'transformer' else '-'
    lwtext = ('1' if c.get('listwise_texto') else '0') if arch == 'listwise' else '-'
    has_text = ((arch == 'transformer' and form in ('text', 'hybrid', 'fusion'))
                or arch == 'tower' or lwtext == '1')
    has_tab = not (arch == 'transformer' and form == 'text')
    if has_tab:
        drop = c.get('drop_features', '') or ''
        if isinstance(drop, str):
            drop = [f.strip() for f in drop.split(',') if f.strip()]
        drops = ','.join(sorted(set(drop)))
    else:
        drops = '-'
    if has_tab and arch != 'listwise':
        extra = c.get('extra_features', '') or ''
        if isinstance(extra, str):
            extra = [f.strip() for f in extra.split(',') if f.strip()]
        if extra == ['all']:
            extra = sorted(EXTRA_ALL)
        extras = ','.join(sorted(set(extra)))
    else:
        extras = '-'
    strip = ('1' if c.get('strip_status') else '0') if has_text else '-'
    maxlen = str(int(c.get('max_text_len', 256))) if has_text else '-'
    nmode = '-' if not has_tab else ('linear' if arch == 'listwise' else c.get('numeric_mode', 'linear'))
    nbins = str(int(c.get('n_bins', 16))) if nmode == 'bins' else '-'
    catenc = '-' if (not has_tab or arch == 'listwise') else c.get('cat_encoding', 'embedding')
    pares = {} if catenc in ('-', 'onehot') else _catfe_pares(c, catenc)
    catfe = '-' if catenc in ('-', 'onehot') else ','.join(f'{f}={pares[f]}' for f in sorted(pares))
    buckets = str(int(c.get('hash_buckets', 8))) \
        if (catenc == 'hashing' or 'hashing' in pares.values()) else '-'
    pool = c.get('pooling', 'cls') if arch == 'transformer' else '-'
    clspos = c.get('cls_position', 'first') if arch == 'transformer' else '-'
    if arch == 'transformer':
        posit = '1' if (form in ('text', 'hybrid') or c.get('positional')) else '0'
    else:
        posit = '-'
    caus = ('1' if c.get('causal') else '0') if arch in ('transformer', 'tower') else '-'
    posw = '1' if c.get('pos_weight') else '0'
    cart = '-' if arch == 'listwise' else ffloat(c.get('cart_aux', 0) or 0)
    ttok = '-' if not has_text else ('chars' if arch == 'listwise'
                                     else c.get('text_tokens', 'chars'))
    w2v = '-' if not has_text else ('1' if (ttok == 'words' and c.get('w2v_init')) else '0')
    frac = ffloat(c.get('train_frac', 1.0) or 1.0)
    init = '-' if c.get('init_seed') is None else str(int(c['init_seed']))
    mlm = str(int(c.get('pretrain_mlm', 0) or 0))
    cvk = int(c.get('cv_k', 0) or 0)
    cv = f"{cvk}.{int(c.get('cv_fold', 0) or 0)}" if cvk else '-'
    pf = c.get('per_feature', 'none') if (arch == 'transformer' and form == 'features') else '-'
    nh = '-' if arch == 'mlp' else str(int(c.get('n_head', 4)))
    nl = '-' if arch == 'mlp' else str(int(c.get('n_layer', 2)))

    # ---- 6ta tanda: regularizacion / transfer / SIA (None-safe: 0 es valor real) ----
    def num(campo, defecto):
        v = c.get(campo, defecto)
        return defecto if v is None else v
    wd = ffloat(num('weight_decay', 1e-2))
    fdrop = (ffloat(num('feature_dropout', 0.0))
             if arch == 'transformer' and form == 'features' else '-')
    lsm = '-' if arch == 'listwise' else ffloat(num('label_smoothing', 0.0))
    sinres = ('1' if c.get('sin_residual') else '0') if arch == 'transformer' else '-'
    sinln = ('1' if c.get('sin_layernorm') else '0') if arch == 'transformer' else '-'
    ifrom = (c.get('init_from') or '-') if arch == 'transformer' else '-'
    frz = ('1' if c.get('freeze_backbone') else '0') if arch == 'transformer' else '-'
    rih = ('1' if c.get('reinit_head') else '0') if ifrom != '-' else '-'
    l2sp = ffloat(num('l2sp', 0.0))
    dst = c.get('distill_from') or '-'
    dsta = ffloat(num('distill_alpha', 1.0)) if dst != '-' else '-'
    efrom = (c.get('embed_from') or '-') if arch == 'mlp' else '-'
    som = '-' if (arch in ('listwise', 'tower')
                  or (arch == 'transformer' and form == 'text')) \
        else str(int(num('som_feature', 0)))
    ae = str(int(num('pretrain_ae', 0)))
    ael = str(int(num('ae_latent', 0))) if arch == 'mlp' else '-'
    pca = str(int(num('pca', 0))) if arch == 'mlp' else '-'

    # ---- 8va tanda: transfer desde un preentrenado externo ----
    temb = c.get('text_emb') or '-'
    if temb != '-':  # canonico por nombre de archivo (la ruta puede variar)
        temb = temb.replace('\\', '/').split('/')[-1]
        temb = temb[:-4] if temb.endswith('.npy') else temb
    tembft = c.get('text_emb_finetune') or '-'
    templr = ffloat(num('text_emb_lr', 1e-5)) if tembft != '-' else '-'

    return '|'.join([
        arch, form, drops, strip, nmode, nbins, str(int(c['d_model'])), nh, nl,
        ffloat(c.get('dropout', 0.1)), pool, posit, caus, posw, maxlen,
        str(int(c.get('epochs', 60))), str(int(c.get('batch_size', 256))),
        ffloat(c.get('lr', 1e-3)), str(int(c.get('patience', 8))),
        catenc, buckets, clspos, cart, extras, lwtext, catfe, ttok, w2v,
        frac, init, mlm, cv, pf,
        wd, fdrop, lsm, sinres, sinln, ifrom, frz, rih, l2sp, dst, dsta, efrom,
        som, ae, ael, pca, temb, tembft, templr,
    ])


CFG_FIELDS = ['arch', 'formulation', 'drop_features', 'strip_status', 'max_text_len',
              'numeric_mode', 'n_bins', 'd_model', 'n_head', 'n_layer', 'dropout',
              'pooling', 'positional', 'causal', 'pos_weight', 'epochs', 'batch_size',
              'lr', 'patience', 'cat_encoding', 'hash_buckets', 'cls_position',
              'cart_aux', 'extra_features', 'listwise_texto', 'cat_feature_encoding',
              'text_tokens', 'w2v_init', 'train_frac', 'init_seed', 'pretrain_mlm',
              'cv_k', 'cv_fold', 'per_feature',
              'weight_decay', 'feature_dropout', 'label_smoothing', 'sin_residual',
              'sin_layernorm', 'init_from', 'freeze_backbone', 'reinit_head', 'l2sp',
              'distill_from', 'distill_alpha', 'embed_from', 'som_feature',
              'pretrain_ae', 'ae_latent', 'pca',
              'text_emb', 'text_emb_finetune', 'text_emb_lr']

CFG_DEFAULTS = {'cat_encoding': 'embedding', 'hash_buckets': 8, 'cls_position': 'first',
                'cart_aux': 0.0, 'listwise_texto': False, 'cat_feature_encoding': '',
                'text_tokens': 'chars', 'w2v_init': False, 'train_frac': 1.0,
                'init_seed': None, 'pretrain_mlm': 0, 'cv_k': 0, 'cv_fold': 0,
                'per_feature': 'none',
                'weight_decay': 1e-2, 'feature_dropout': 0.0, 'label_smoothing': 0.0,
                'sin_residual': False, 'sin_layernorm': False, 'init_from': '',
                'freeze_backbone': False, 'reinit_head': False, 'l2sp': 0.0,
                'distill_from': '', 'distill_alpha': 1.0, 'embed_from': '',
                'som_feature': 0, 'pretrain_ae': 0, 'ae_latent': 0, 'pca': 0,
                'text_emb': '', 'text_emb_finetune': '', 'text_emb_lr': 1e-5}


def cfg_dict(c):
    """Config reducida a los campos que definen el modelo, con drop/extra como listas."""
    out = {k: c.get(k, CFG_DEFAULTS.get(k)) for k in CFG_FIELDS}
    for campo in ('drop_features', 'extra_features'):
        v = out.get(campo) or ''
        if isinstance(v, str):
            v = [f.strip() for f in v.split(',') if f.strip()]
        if campo == 'extra_features' and v == ['all']:
            v = sorted(EXTRA_ALL)
        out[campo] = sorted(set(v))
    pares = sorted(p.strip() for p in str(out.get('cat_feature_encoding') or '').split(',')
                   if p.strip() and '=' in p)
    out['cat_feature_encoding'] = ','.join(pares)
    return out


METRICAS = [
    ('pr_auc', 'PR-AUC', 'up'), ('roc_auc', 'ROC-AUC', 'up'),
    ('loss', 'loss (criterio de entrenamiento)', 'down'), ('log_loss', 'log-loss (BCE sin pesar)', 'down'),
    ('brier', 'Brier score', 'down'), ('f1_best', 'F1 máximo', 'up'),
    ('thr_f1_best', 'umbral del F1 máximo', 'none'), ('f1', 'F1 @ 0.5', 'up'),
    ('precision', 'precisión @ 0.5', 'up'), ('recall', 'recall @ 0.5', 'up'),
    ('specificity', 'especificidad @ 0.5', 'up'), ('accuracy', 'accuracy @ 0.5', 'up'),
    ('balanced_accuracy', 'balanced accuracy @ 0.5', 'up'), ('mcc', 'MCC @ 0.5', 'up'),
    ('tasa_pred_pos', 'tasa de positivos predicha', 'none'), ('tasa_real_pos', 'tasa de positivos real', 'none'),
]


def cargar_datos():
    # cardinalidades reales (mismo fit que usan los modelos)
    df = load_dataset(REPO / 'supermarket_products.csv')
    train_df, _, _ = split_by_query(df)
    prep = Preprocessor.fit(train_df)
    features = {
        'cat': [{'name': f, 'card': len(prep.vocabs[f])} for f in CAT_FEATURES],
        'num': [{'name': f} for f in NUM_FEATURES],
    }

    # suite curada, con su clave canonica y su config completa
    suite = {}
    for nombre, (extra, familia) in EXPERIMENTOS.items():
        cfg = vars(build_parser().parse_args(extra))
        suite[nombre] = {'key': canon(cfg), 'cfg': cfg_dict(cfg),
                         'familia': familia, 'args': ' '.join(extra) or '(defaults)'}

    # resultados agrupados por clave canonica
    grupos = {}
    for path in sorted((REPO / 'resultados').glob('*.json')):
        data = json.loads(path.read_text())
        cfg = data['config']
        key = canon(cfg)
        g = grupos.setdefault(key, {'cfg': cfg_dict(cfg), 'runs': [], 'pesos': False,
                                    'nombre': data['nombre']})
        hist = {'train': {}, 'val': {}}
        for h in data['historial']:
            if 'train' not in h:  # schema viejo (1ra tanda GPU): claves planas train_loss etc.
                h = {'train': {k[6:]: v for k, v in h.items() if k.startswith('train_')},
                     'val': {k[4:]: v for k, v in h.items() if k.startswith('val_')}}
            for split in ('train', 'val'):
                for m, v in h[split].items():
                    hist[split].setdefault(m, []).append(round(v, 5))
        g['runs'].append({'seed': data['seed'], 'epochs': len(data['historial']),
                          'device': data.get('device', '?'), 'params': data['n_parametros'],
                          'val': {k: round(v, 5) for k, v in data['val'].items()},
                          'test': {k: round(v, 5) for k, v in data['test'].items()},
                          'hist': hist})
        if (REPO / 'pesos' / f"{data['nombre']}.pt").exists():
            g['pesos'] = True

    # verificacion python-side: las corridas de la suite deben matchear su clave
    suite_keys = {v['key']: n for n, v in suite.items()}
    en_suite = sum(1 for k in grupos if k in suite_keys)
    print(f'suite: {len(suite)} configs | resultados: {len(grupos)} grupos '
          f'({sum(len(g["runs"]) for g in grupos.values())} corridas), {en_suite} matchean la suite')
    for k, g in grupos.items():
        if g['nombre'].startswith(tuple(suite)) and k not in suite_keys:
            raise SystemExit(f'BUG canon(): {g["nombre"]} no matchea ninguna clave de la suite')

    defaults = cfg_dict(vars(build_parser().parse_args([])))
    return features, suite, grupos, defaults


# ----------------------------------------------------------------- pagina

CSS = """
:root{
  --bg:#EFF2F5; --card:#FFFFFF; --figbg:#F8FAFC; --grid:#DFE6EC;
  --ink:#1B2530; --ink2:#3D4B5A; --muted:#5F7183; --line:#D8DFE7;
  --tab:#1C8A76; --tab-bg:#DFF0EB; --txt:#B58117; --txt-bg:#F8ECCB;
  --cls:#7052C9; --cls-bg:#E7DEF9; --mlp:#2E68AC; --mlp-bg:#DCE9F8;
  --out:#C22B5E; --ok:#1C8A76; --warn:#A56A00; --warn-bg:#F8ECCB; --err:#C22B5E;
  --sans:"Segoe UI","Noto Sans","Liberation Sans",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"Cascadia Code","JetBrains Mono","Fira Mono","DejaVu Sans Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){ %DARK% } }
:root[data-theme="dark"]{ %DARK% }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14.5px/1.55 var(--sans)}
.wrap{max-width:1180px;margin:0 auto;padding:34px 20px 64px}
.eyebrow{font:600 10.5px var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
h1{font:700 30px/1.1 var(--sans);margin:6px 0 8px;letter-spacing:-.01em}
.lede{color:var(--ink2);max-width:78ch;margin:0 0 6px}
#selftest{display:none;background:var(--err);color:#fff;padding:10px 14px;border-radius:10px;
  margin:12px 0;font:600 13px var(--mono)}
.cols{display:grid;grid-template-columns:minmax(0,7fr) minmax(0,5fr);gap:18px;margin-top:20px;
  align-items:start}
@media (max-width:980px){.cols{grid-template-columns:1fr}.right{position:static}}
.right{position:sticky;top:14px;display:flex;flex-direction:column;gap:14px;
  max-height:calc(100vh - 28px);overflow-y:auto;padding-bottom:4px}
.card{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:16px 18px;
  margin-bottom:14px}
.right .card{margin-bottom:0}
.card h2{font:700 15px var(--sans);margin:2px 0 10px}
.card h2 .n{font:700 11px var(--mono);color:var(--muted);margin-right:6px}
.hint{color:var(--muted);font-size:12px;margin:8px 0 0;max-width:70ch}
.hint b{color:var(--ink2)}
fieldset{border:none;margin:0;padding:0}
.optgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:7px}
.opt{display:block;border:1.4px solid var(--line);border-radius:10px;padding:8px 10px;cursor:pointer;
  background:var(--figbg)}
.opt:hover{border-color:var(--muted)}
.opt input{margin-right:6px;accent-color:var(--tab)}
.opt.on{border-color:var(--tab);background:var(--tab-bg)}
.opt .t{font:600 13px var(--sans)}
.opt .d{display:block;color:var(--muted);font-size:11px;line-height:1.35;margin-top:2px}
.featgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:5px}
.feat{display:flex;align-items:center;gap:6px;font:12px var(--mono);border:1.2px solid var(--line);
  border-radius:8px;padding:5px 8px;cursor:pointer;background:var(--figbg)}
.feat input{accent-color:var(--tab);margin:0}
.feat.off{opacity:.45;text-decoration:line-through}
.feat .card-n{color:var(--muted);margin-left:auto;font-size:10.5px}
.rowc{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin:8px 0 0}
.rowc label.inl{font:12.5px var(--sans);color:var(--ink2);display:flex;align-items:center;gap:6px}
.rowc input[type=number]{width:76px;font:12.5px var(--mono);color:var(--ink);background:var(--figbg);
  border:1.2px solid var(--line);border-radius:7px;padding:4px 7px}
select{font:12.5px var(--mono);color:var(--ink);background:var(--figbg);border:1.2px solid var(--line);
  border-radius:7px;padding:4px 7px;max-width:100%}
input[type=checkbox]{accent-color:var(--tab)}
.seg{display:inline-flex;border:1.2px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{font:12px var(--mono);border:none;background:var(--figbg);color:var(--ink2);
  padding:5px 11px;cursor:pointer}
.seg button.on{background:var(--tab);color:#fff}
.preset{font:12px var(--mono);border:1.2px solid var(--line);background:var(--figbg);color:var(--ink2);
  border-radius:999px;padding:4px 12px;cursor:pointer}
.preset:hover{border-color:var(--tab);color:var(--ink)}
.pill{display:inline-block;font:11px var(--mono);border-radius:999px;padding:2px 10px;margin:2px 3px 2px 0}
.pill.ok{background:var(--tab-bg);color:var(--tab);border:1px solid var(--tab)}
.pill.warn{background:var(--warn-bg);color:var(--warn);border:1px solid var(--warn)}
.pill.err{background:transparent;color:var(--err);border:1px solid var(--err)}
.pill.info{background:var(--figbg);color:var(--muted);border:1px solid var(--line)}
.pill.cls{background:var(--cls-bg);color:var(--cls);border:1px solid var(--cls)}
pre.cmd{font:12px/1.5 var(--mono);background:var(--figbg);border:1px solid var(--line);
  border-radius:9px;padding:10px 12px;white-space:pre-wrap;word-break:break-all;margin:8px 0;
  user-select:all}
.copy{font:11px var(--mono);border:1px solid var(--line);background:var(--figbg);color:var(--ink2);
  border-radius:7px;padding:3px 10px;cursor:pointer}
.copy:hover{border-color:var(--tab);color:var(--ink)}
ul.notes{margin:8px 0 0;padding-left:18px;color:var(--ink2);font-size:12.5px}
ul.notes li{margin:3px 0}
table.mt{border-collapse:collapse;width:100%;font-size:12px;margin-top:8px}
table.mt th{font:600 10px var(--mono);letter-spacing:.07em;text-transform:uppercase;color:var(--muted);
  text-align:right;padding:4px 8px;border-bottom:1px solid var(--line)}
table.mt th:first-child{text-align:left}
table.mt td{padding:3.5px 8px;border-bottom:1px solid var(--line);text-align:right;
  font:11.5px var(--mono);font-variant-numeric:tabular-nums;color:var(--ink2)}
table.mt td:first-child{text-align:left;font-family:var(--sans);font-size:12px;color:var(--ink)}
table.mt tr.hl td{background:var(--tab-bg);color:var(--ink);font-weight:600}
.rank{display:flex;flex-direction:column;gap:4px;margin-top:8px}
.rrow{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:center;
  border:1.2px solid var(--line);border-radius:8px;padding:6px 10px;cursor:pointer;background:var(--figbg)}
.rrow:hover{border-color:var(--tab)}
.rrow.sel{border-color:var(--tab);background:var(--tab-bg)}
.rrow .nm{font:12px var(--mono);color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rrow .bar{height:4px;border-radius:2px;background:var(--tab);margin-top:4px}
.rrow .vl{font:12px var(--mono);color:var(--ink2);font-variant-numeric:tabular-nums}
.rrow .sd{color:var(--muted);font-size:10.5px}
svg.chart{width:100%;height:auto;display:block;background:var(--figbg);border:1px solid var(--line);
  border-radius:9px;margin-top:8px}
.axis{font:9.5px var(--mono);fill:var(--muted)}
.gridl{stroke:var(--line);stroke-width:1;stroke-dasharray:3 3}
.ln{fill:none;stroke-width:1.4;opacity:.55}
.ln.mean{stroke-width:2.4;opacity:1}
.ln-train{stroke:var(--tab)} .ln-val{stroke:var(--cls)}
.leg{font:11px var(--mono);color:var(--muted);display:flex;gap:14px;margin-top:6px}
.leg i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}
footer{color:var(--muted);font-size:12px;margin-top:26px}
footer code, .hint code, .lede code{font:11.5px var(--mono);background:var(--figbg);
  border:1px solid var(--line);border-radius:5px;padding:0 4px}
button:focus-visible,input:focus-visible,select:focus-visible,.rrow:focus-visible,.opt:focus-within{
  outline:2px solid var(--mlp);outline-offset:1px}
.empty{color:var(--muted);font-size:12.5px;background:var(--figbg);border:1px dashed var(--line);
  border-radius:9px;padding:12px 14px;margin-top:8px}
"""

DARK = """
  --bg:#0F141A; --card:#171E27; --figbg:#121924; --grid:#202C3B;
  --ink:#E6EBF1; --ink2:#C0CAD6; --muted:#8D9BAB; --line:#2A3543;
  --tab:#34A98F; --tab-bg:#143530; --txt:#C79A2F; --txt-bg:#3B2F12;
  --cls:#9678E8; --cls-bg:#2B2350; --mlp:#4C8BD4; --mlp-bg:#17293E;
  --out:#E85585; --ok:#34A98F; --warn:#E0B14C; --warn-bg:#3B2F12; --err:#E85585;
"""

ZOO = 'https://claude.ai/code/artifact/1464165c-3e4a-4c56-81f0-e8d467b35533'

ARCHS = [
    ('feat', 'A · Transformer tabular', 'cada feature es un token · atención entre features'),
    ('fusion', '5B · Fusión', 'el resumen del texto como token 15 · cruce sin dilución'),
    ('mlp', 'MLP (control)', 'mismos embeddings, sin atención'),
    ('text', 'C · Transformer de texto', 'cada carácter es un token · la demo adaptada'),
    ('hybrid', 'A+C · Híbrido', 'features + chars en una secuencia · atención cruzada'),
    ('tower', 'C2 · Torre de texto', 'transformer solo p/ embeddings + MLP clasificador'),
    ('listwise', 'B · Listwise', 'cada producto de la página es un token'),
]


def controles(features):
    feats = ''
    for f in features['cat']:
        feats += (f'<label class="feat" data-f="{f["name"]}"><input type="checkbox" '
                  f'name="feat" value="{f["name"]}" checked>{f["name"]}'
                  f'<span class="card-n">{f["card"]} niveles</span></label>')
    for f in features['num']:
        feats += (f'<label class="feat" data-f="{f["name"]}"><input type="checkbox" '
                  f'name="feat" value="{f["name"]}" checked>{f["name"]}'
                  f'<span class="card-n">num</span></label>')

    archs = ''
    for aid, t, d in ARCHS:
        on = ' on' if aid == 'feat' else ''
        chk = ' checked' if aid == 'feat' else ''
        archs += (f'<label class="opt{on}" data-arch="{aid}"><input type="radio" name="arch" '
                  f'value="{aid}"{chk}><span class="t">{t}</span><span class="d">{d} · '
                  f'<a href="{ZOO}#{aid}" target="_blank" rel="noopener">diagrama ↗</a></span></label>')

    numpf = ''
    for f in features['num']:
        numpf += (f'<label class="inl">{f["name"]} <select name="numpf" data-f="{f["name"]}">'
                  f'<option value="linear">lineal</option><option value="bins">bins</option>'
                  f'</select></label>')

    catpf = ''
    for f in features['cat']:
        opts = ''.join(f'<option value="{m}">{m}</option>'
                       for m in ('embedding', 'target', 'ordinal', 'freq', 'hashing'))
        catpf += (f'<label class="inl">{f["name"]} <select name="catpf" data-f="{f["name"]}">'
                  f'<option value="">(global)</option>{opts}</select></label>')

    extras = ''
    for f, tipo in sorted(EXTRA_FEATURES.items()):
        extras += (f'<label class="feat" data-x="{f}"><input type="checkbox" name="extraf" '
                   f'value="{f}">{f}<span class="card-n">{tipo}</span></label>')

    return f'''
<section class="card"><h2><span class="n">1</span>Arquitectura</h2>
  <fieldset class="optgrid" id="arch-grid">{archs}</fieldset>
  <div class="rowc" id="lwtext-wrap" style="display:none">
    <label class="inl"><input type="checkbox" id="listwise_texto"> enriquecer el token de
    producto con la torre de texto (--listwise-texto, propuesta #1 de Junior)</label>
  </div>
  <p class="hint" id="arch-hint"></p>
</section>

<section class="card"><h2><span class="n">2</span>Entrada — qué features mandamos</h2>
  <div class="rowc" style="margin:0 0 8px">
    <button class="preset" id="pre-catalogo">preset: catálogo (todo)</button>
    <button class="preset" id="pre-intrinseco">preset: intrínseco (producto nuevo)</button>
  </div>
  <fieldset class="featgrid" id="feat-grid">{feats}</fieldset>
  <div class="eyebrow" style="margin:12px 0 6px">DESCARTADOS EN EL EDA — REINTRODUCIBLES (--extra-features)</div>
  <fieldset class="featgrid" id="extra-grid">{extras}</fieldset>
  <div class="rowc">
    <label class="inl" id="strip-wrap"><input type="checkbox" id="strip">
      --strip-status: borrar del TEXTO el sufijo del título y la última oración</label>
  </div>
  <p class="hint">El preset <b>intrínseco</b> = sacar <code>listing_status</code> + strip del texto:
  simula el producto nuevo sin historial (techo GBM: 0.762 → 0.162). Destildar features acá
  se traduce a <code>--drop-features</code>. Los <b>extra</b> (volumen desde dimensions, package
  parseado, nº de ingredientes) vienen de columnas que el EDA declaró redundantes —
  <code>feat_extras</code> lo verifica empíricamente.</p>
</section>

<section class="card"><h2><span class="n">3</span>Encodings</h2>
  <div class="rowc">
    <label class="inl">categóricas <select id="catenc">
      <option value="embedding">embedding aprendido por columna (default)</option>
      <option value="onehot">one-hot crudo (solo MLP)</option>
      <option value="target">target encoding suavizado</option>
      <option value="ordinal">ordinal (rango por BTR de train)</option>
      <option value="freq">frequency encoding</option>
      <option value="hashing">hashing trick (el "modular")</option>
    </select></label>
    <label class="inl" id="buckets-wrap" style="display:none">buckets
      <input type="number" id="hash_buckets" value="8" min="2" max="64"></label>
    <label class="inl">numéricas <select id="numenc">
      <option value="linear">lineal: x·w+b por feature</option>
      <option value="bins">bins por cuantiles</option>
      <option value="mixto">elegir por feature (pedir)</option>
    </select></label>
    <label class="inl" id="nbins-wrap">n_bins <input type="number" id="n_bins" value="16" min="2" max="64"></label>
  </div>
  <div class="rowc" id="numpf-wrap" style="display:none">{numpf}</div>
  <div class="rowc" id="texttok-wrap">
    <label class="inl">texto: tokens <span class="seg" id="ttok-seg">
      <button type="button" data-v="chars" class="on">caracteres</button>
      <button type="button" data-v="words">palabras</button></span></label>
    <label class="inl" id="w2v-wrap"><input type="checkbox" id="w2v_init">
      inicializar con word2vec (skipgram sobre train)</label>
  </div>
  <details id="catpf-details" style="margin-top:8px"><summary class="hint" style="cursor:pointer">
    encoding por feature categórica (override del global) — p. ej. solo listing_status en ordinal</summary>
    <div class="rowc">{catpf}</div></details>
  <p class="hint">Los seis encodings de categóricas están <b>implementados</b>
  (<code>--cat-encoding</code>, suite <code>feat_target</code>/<code>feat_freq</code>/<code>feat_hash8</code>/<code>mlp_onehot</code>).
  El default es <b>column embeddings</b> (una tabla por columna — el "columnar/modular" que te
  mencionaron es esta familia: hashing usa el módulo). <b>one-hot + capa lineal aprende la misma
  matriz que el embedding</b> (§6.1): por eso one-hot solo existe como entrada cruda al MLP.
  <b>Target</b> = nivel → BTR promedio suavizado de train (m=50; ajustado SOLO con train);
  <b>freq</b> = frecuencia del nivel; <b>hashing</b> = nivel → hash % B con colisiones a propósito
  (útil con miles de niveles; acá mide cuánto duele). <b>Bins</b> captura la U del precio sin
  depender del FFN — la 1ª tanda dice que el FFN alcanza (bins −0.023).</p>
</section>

<section class="card"><h2><span class="n">4</span>Capacidad del modelo</h2>
  <div class="rowc">
    <label class="inl">d_model <input type="number" id="d_model" value="32" min="4" step="4"></label>
    <label class="inl" id="nh-wrap">cabezas <input type="number" id="n_head" value="4" min="1"></label>
    <label class="inl" id="nl-wrap">bloques <input type="number" id="n_layer" value="2" min="1"></label>
    <label class="inl">dropout <input type="number" id="dropout" value="0.1" min="0" max="0.9" step="0.05"></label>
  </div>
  <div class="rowc">
    <span class="pill info">grilla de la suite: d 8·16·32·64 — bloques 1·2·4 — cabezas 1·2·4</span>
  </div>
  <div class="rowc">
    <label class="inl" id="pool-wrap">pooling <span class="seg" id="pool-seg">
      <button type="button" data-v="cls" class="on">[CLS]</button>
      <button type="button" data-v="mean">promedio</button></span></label>
    <label class="inl" id="pf-wrap">pesos por feature <span class="seg" id="pf-seg">
      <button type="button" data-v="none" class="on">no</button>
      <button type="button" data-v="qkv">QKV</button>
      <button type="button" data-v="ffn">FFN</button>
      <button type="button" data-v="both">ambos</button>
      <button type="button" data-v="gate">gate</button></span></label>
    <label class="inl" id="clspos-wrap">CLS <span class="seg" id="clspos-seg">
      <button type="button" data-v="first" class="on">al inicio</button>
      <button type="button" data-v="last">al final</button></span></label>
    <label class="inl" id="pos-wrap"><input type="checkbox" id="positional"> positional encoding</label>
    <label class="inl" id="caus-wrap"><input type="checkbox" id="causal"> máscara causal (ablación)</label>
    <label class="inl"><input type="checkbox" id="pos_weight"> pos_weight en la BCE (87/13)</label>
  </div>
  <p class="hint" id="cap-hint"></p>
  <p class="hint">Medido en la 1ª tanda: <b>causal con CLS al inicio degenera</b> (el CLS solo se
  ve a sí mismo → p constante, ROC 0.500 exacto). Para que la ablación causal tenga sentido, CLS
  <b>al final</b> (<code>feat_causal_last</code>). También medido: CLS &gt; promedio (6/6 seeds) y
  pos_weight daña el test PR (−0.059).</p>
</section>

<section class="card"><h2><span class="n">5</span>Validación y entrenamiento</h2>
  <fieldset class="optgrid">
    <label class="opt on" data-val="holdout"><input type="radio" name="val" value="holdout" checked>
      <span class="t">holdout por query</span><span class="d">70/15/15 × N seeds (actual)</span></label>
    <label class="opt" data-val="gkfold"><input type="radio" name="val" value="gkfold">
      <span class="t">GroupKFold por query</span><span class="d">k folds agrupados (--cv-k; suite cv5_fold*)</span></label>
    <label class="opt" data-val="producto"><input type="radio" name="val" value="producto">
      <span class="t">split por producto</span><span class="d">verificado: da igual (§7.1) (pedir)</span></label>
  </fieldset>
  <div class="rowc">
    <label class="inl">seeds <input type="number" id="seeds" value="6" min="1" max="10"></label>
    <label class="inl" id="k-wrap" style="display:none">k <input type="number" id="kfold" value="5" min="3" max="10"></label>
    <label class="inl">épocas <input type="number" id="epochs" value="60" min="1"></label>
    <label class="inl">batch <input type="number" id="batch_size" value="256" min="16" step="16"></label>
    <label class="inl">lr <input type="number" id="lr" value="0.001" min="0" step="0.0005"></label>
    <label class="inl">patience <input type="number" id="patience" value="8" min="1"></label>
    <label class="inl" id="cart-wrap">λ cart (multi-task) <input type="number" id="cart_aux" value="0" min="0" max="1" step="0.1"></label>
  </div>
  <div class="rowc">
    <label class="inl">fracción de train <input type="number" id="train_frac" value="1" min="0.05" max="1" step="0.05"></label>
    <label class="inl">init-seed <input type="number" id="init_seed" placeholder="= seed" min="0"></label>
    <label class="inl">épocas MLM (pre-training) <input type="number" id="pretrain_mlm" value="0" min="0" max="100"></label>
  </div>
  <p class="hint">La decisión de parar y de elegir configs se toma SIEMPRE con la validación
  (PR-AUC); test solo se reporta. El protocolo vigente tras la 1ª tanda: <b>6 seeds</b>, y para
  tabulares <b>patience 20 / tope 300</b> (ganó en 21/24; en texto puro empeora el test).
  <b>λ cart</b> &gt; 0 agrega la BCE auxiliar sobre <code>cart</code> como segunda tarea
  (nunca como input — propuesta #2 de Junior; suite <code>feat_cartaux01/03/05</code>).
  <b>Fracción de train</b> = curva de aprendizaje (suite <code>curva_frac*</code>);
  <b>init-seed</b> separa la seed del modelo de la del split (varianza y deep-ensembles, suite
  <code>robu_init*</code>); <b>MLM</b> pre-entrena el tronco enmascarando una feature por fila
  (suite <code>feat_mlm20</code>, <code>feat_ordinal_mlm20</code>); GroupKFold: <code>cv5_fold0..4</code>.</p>
</section>

<section class="card"><h2><span class="n">5b</span>Regularización · Transfer learning · SIA (6ª tanda)</h2>
  <div class="eyebrow" style="margin:0 0 6px">REGULARIZACIÓN (nunca barrida hasta la 6ª tanda)</div>
  <div class="rowc">
    <label class="inl">weight decay (AdamW) <input type="number" id="weight_decay" value="0.01" min="0" max="1" step="0.001"></label>
    <label class="inl" id="fdrop-wrap">feature-dropout <input type="number" id="feature_dropout" value="0" min="0" max="0.9" step="0.05"></label>
    <label class="inl">label smoothing <input type="number" id="label_smoothing" value="0" min="0" max="0.5" step="0.05"></label>
    <label class="inl" id="sinres-wrap"><input type="checkbox" id="sin_residual"> sin residuales (ablación)</label>
    <label class="inl" id="sinln-wrap"><input type="checkbox" id="sin_layernorm"> sin LayerNorm (ablación)</label>
  </div>
  <div class="eyebrow" style="margin:12px 0 6px">TRANSFER LEARNING (clase 3: extraction / fine-tuning / distillation)</div>
  <div class="rowc">
    <button class="preset" id="pre-probe">preset: probe del campeón</button>
    <button class="preset" id="pre-dst-camp">preset: distill ← campeón</button>
    <button class="preset" id="pre-dst-ens">preset: distill ← deep-ensemble</button>
  </div>
  <div class="rowc">
    <label class="inl" style="flex:1">init-from (ckpt, admite {{seed}} y glob)
      <input type="text" id="init_from" value="" placeholder="pesos/..._seed{{seed}}.pt" style="width:100%"></label>
  </div>
  <div class="rowc">
    <label class="inl"><input type="checkbox" id="freeze_backbone"> congelar backbone (probe: solo la cabeza)</label>
    <label class="inl"><input type="checkbox" id="reinit_head"> reinicializar la cabeza</label>
    <label class="inl">λ L2-SP (ancla al pre-entrenado) <input type="number" id="l2sp" value="0" min="0" max="1" step="0.001"></label>
  </div>
  <div class="rowc">
    <label class="inl" style="flex:1">distill-from (teachers, coma/{{seed}}/glob)
      <input type="text" id="distill_from" value="" placeholder="pesos/teacher_seed{{seed}}.pt,..." style="width:100%"></label>
    <label class="inl">α soft <input type="number" id="distill_alpha" value="1" min="0" max="1" step="0.1"></label>
  </div>
  <div class="rowc" id="efrom-wrap">
    <label class="inl" style="flex:1">embed-from (MLP: + embedding congelado del transformer)
      <input type="text" id="embed_from" value="" placeholder="pesos/..._seed{{seed}}.pt" style="width:100%"></label>
  </div>
  <div class="eyebrow" style="margin:12px 0 6px">PREENTRENADO EXTERNO (8ª tanda: MiniLM sobre title+description)</div>
  <div class="rowc">
    <label class="inl">text-emb (congelado, .npy)
      <select id="text_emb">
        <option value="">— no —</option>
        <option value="embeddings/minilm.npy">minilm (texto completo)</option>
        <option value="embeddings/minilm_intr.npy">minilm_intr (sin status)</option>
      </select></label>
    <label class="inl" style="flex:1">text-emb-finetune (modelo HF, entra al grafo)
      <input type="text" id="text_emb_finetune" value="" placeholder="sentence-transformers/all-MiniLM-L6-v2" style="width:100%"></label>
    <label class="inl">lr encoder <input type="number" id="text_emb_lr" value="0.00001" min="0" max="0.01" step="0.00001"></label>
  </div>
  <div class="eyebrow" style="margin:12px 0 6px">HERRAMIENTAS DE SIA</div>
  <div class="rowc">
    <label class="inl" id="som-wrap">SOM (Kohonen) G×G, 0 = off <input type="number" id="som_feature" value="0" min="0" max="16"></label>
    <label class="inl" id="ae-wrap">épocas AE pre-training (CLS reconstruye) <input type="number" id="pretrain_ae" value="0" min="0" max="100"></label>
    <label class="inl" id="ael-wrap">AE→latente K (MLP) <input type="number" id="ae_latent" value="0" min="0" max="64"></label>
    <label class="inl" id="pca-wrap">PCA K (MLP) <input type="number" id="pca" value="0" min="0" max="64"></label>
  </div>
  <p class="hint">Las tres técnicas de la <b>clase 3</b> con nuestros propios checkpoints como base:
  <b>feature extraction</b> = congelar el tronco y entrenar solo la cabeza (probe lineal), o darle
  al MLP el embedding pooled del transformer ("el embedding más un montón de cosas");
  <b>fine-tuning</b> = --init-from (con λ&gt;0, L2-SP: la "KL penalty" — ajustarse sin alejarse del
  pre-entrenado); <b>knowledge distillation</b> = entrenar contra las PROBABILIDADES del teacher
  (el 0.8 informa más que el 1); el preset deep-ensemble usa los 6 modelos del mismo split como
  teacher (n&gt;1, "integrando información de varios modelos"). SIA: la celda BMU de un
  <b>Kohonen</b> como categórica extra, el <b>autoencoder</b> como pre-training del tronco
  (hermano con cuello de botella del MLM) y <b>PCA/AE→latente</b> como única entrada del MLP
  (representation learning puro). Suite: <code>reg_*</code>, <code>abl_*</code>, <code>tl_*</code>,
  <code>sia_*</code>. La 8ª tanda agrega el caso canónico: un preentrenado <b>externo</b>
  (MiniLM) como encoder del texto — congelado (<code>--text-emb</code>, un token extra
  proyectado 384→d) o fine-tuneado en el grafo (<code>--text-emb-finetune</code>). Suite:
  <code>bert_*</code>.</p>
</section>

<section class="card"><h2><span class="n">6</span>Métricas — se calculan SIEMPRE todas</h2>
  <div id="met-pills"></div>
  <div class="rowc">
    <label class="inl"><input type="checkbox" id="mq"> agregar métricas por página
      (top-1 de la query, NDCG) al evaluate (pedir)</label>
  </div>
  <p class="hint">Cada época guarda las 16 en train y val, y la corrida final en val y test
  (<code>compute_metrics</code> en <code>btr/train.py</code>) — el gráfico de la derecha puede
  mostrar cualquiera sin reentrenar. Las de umbral usan 0.5 salvo <b>F1 máximo</b>, que barre
  el umbral (con 13% de positivos el óptimo ronda 0.3).</p>
</section>
'''


def pagina(features, suite, grupos, defaults):
    data = {
        'gen': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'zoo': ZOO,
        'features': features,
        'extra_features': EXTRA_FEATURES,
        'suite': suite,
        'results': grupos,
        'defaults': defaults,
        'metricas': [{'k': k, 'label': l, 'dir': d} for k, l, d in METRICAS],
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    head = '''<title>Laboratorio BTR</title>
<style>''' + CSS.replace('%DARK%', DARK) + '''</style>
<div class="wrap">
<header>
  <div class="eyebrow">TP1 · 73.69 LLM · CONSOLA DE EXPERIMENTOS</div>
  <h1>Laboratorio BTR</h1>
  <p class="lede">Elegí cada decisión y el panel arma la configuración: te da el <b>comando
  exacto</b>, te dice si <b>ya está en la suite</b> o si <b>hay que implementarla</b> (con un spec
  para pegar en el chat), y si ya se corrió te muestra <b>todas las métricas y sus curvas</b>.
  La página es estática: acá no se entrena nada — el laboratorio es el repo
  (<code>panel.py</code> la regenera con los resultados frescos).</p>
  <div id="selftest">⚠ canonKey() JS ≠ canon() Python — avisale a Claude (drift de implementación)</div>
</header>
<div class="cols">
<div class="left">
''' + controles(features) + '''
<footer>Regenerar tras nuevos experimentos: <code>.venv/bin/python panel.py</code> (embebe
resultados/ y la suite) — después pedirle a Claude que republique el artifact.
Diagramas de las arquitecturas: <a href="''' + ZOO + '''" target="_blank" rel="noopener">Zoo de
arquitecturas BTR ↗</a> · <span id="gen-date"></span></footer>
</div>
<div class="right">
  <section class="card"><h2>Comando</h2>
    <div id="estado-pills"></div>
    <pre class="cmd" id="cmd"></pre>
    <button class="copy" id="copy-cmd">copiar comando</button>
    <div id="spec-wrap" style="display:none">
      <p class="hint" style="margin-top:10px"><b>Requiere implementación</b> — pegá esto en el
      chat y Claude lo agrega al código:</p>
      <pre class="cmd" id="spec"></pre>
      <button class="copy" id="copy-spec">copiar spec</button>
    </div>
    <ul class="notes" id="coerciones"></ul>
  </section>
  <section class="card"><h2>Resultados de esta configuración</h2>
    <div id="res-head"></div>
    <div id="res-chart"></div>
    <div id="res-table"></div>
  </section>
  <section class="card"><h2>Ranking de todo lo corrido</h2>
    <div class="rowc" style="margin:0">
      <label class="inl">métrica <select id="rank-metric"></select></label>
      <label class="inl">split <span class="seg" id="rank-split">
        <button type="button" data-v="test" class="on">test</button>
        <button type="button" data-v="val">val</button></span></label>
    </div>
    <div class="rank" id="rank"></div>
  </section>
</div>
</div>
</div>
<script>
const DATA = ''' + payload + ''';
'''
    return head + APP_JS + '\n</script>\n'


APP_JS = r'''
'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const ffloat = x => { let s = Number(x).toFixed(10); s = s.replace(/0+$/,'').replace(/\.$/,''); return s; };
const ARCH2CODE = {feat:{arch:'transformer',formulation:'features'}, mlp:{arch:'mlp'},
  text:{arch:'transformer',formulation:'text'}, hybrid:{arch:'transformer',formulation:'hybrid'},
  fusion:{arch:'transformer',formulation:'fusion'}, tower:{arch:'tower'}, listwise:{arch:'listwise'}};
const ALL_FEATS = DATA.features.cat.map(f=>f.name).concat(DATA.features.num.map(f=>f.name));
// teachers de la 6ta tanda (presets de transfer): el campeon por split, y el
// deep-ensemble = campeon + las 5 inits extra del MISMO split
const CKPT_CAMPEON = 'pesos/feat_ordinal_features_d32_h4_l2_linear_catordinal_seed{seed}.pt';
const CKPT_ENSEMBLE = CKPT_CAMPEON + ',pesos/robu_init4*_features_d32_h4_l2_linear_catordinal_init4*_seed{seed}.pt';

// ---- estado (mismos nombres que argparse en btr/train.py) ----
const S = Object.assign({}, DATA.defaults);
S.drop_features = (S.drop_features||[]).slice();
S.extra_features = (S.extra_features||[]).slice();
const X = {catenc:'embedding', numenc_mixto:false, numpf:{}, val:'holdout', kfold:5, seeds:3, mq:false};
DATA.features.num.forEach(f => X.numpf[f.name] = 'linear');

const hasText = s => (s.arch==='transformer' &&
  (s.formulation==='text'||s.formulation==='hybrid'||s.formulation==='fusion'))
  || s.arch==='tower' || (s.arch==='listwise' && s.listwise_texto);
const catfeRaw = str => {
  const o = {};
  for (const par of String(str||'').split(',')){
    const p = par.trim();
    if (p && p.includes('=')){ const i = p.indexOf('='); o[p.slice(0,i).trim()] = p.slice(i+1).trim(); }
  }
  return o;
};
const catfePares = (c, catencGlobal) => {
  const pares = {};
  for (const par of String(c.cat_feature_encoding||'').split(',')){
    const p = par.trim();
    if (p && p.includes('=')){
      const i = p.indexOf('=');
      const f = p.slice(0,i).trim(), m = p.slice(i+1).trim();
      if (m !== catencGlobal) pares[f] = m;
    }
  }
  return pares;
};
const hasTab  = s => !(s.arch==='transformer' && s.formulation==='text');

// ---- clave canonica: ESPEJO EXACTO de canon() en panel.py ----
function canonKey(c){
  const arch = c.arch;
  const form = arch==='transformer' ? (c.formulation||'features') : '-';
  const lwtext = arch==='listwise' ? (c.listwise_texto ? '1':'0') : '-';
  const htext = (arch==='transformer' && (form==='text'||form==='hybrid'||form==='fusion'))
    || arch==='tower' || lwtext==='1';
  const htab = !(arch==='transformer' && form==='text');
  const lista = v => {
    if (typeof v === 'string') v = v.split(',').map(s=>s.trim()).filter(Boolean);
    return v||[];
  };
  const drops = htab ? Array.from(new Set(lista(c.drop_features))).sort().join(',') : '-';
  let extras = '-';
  if (htab && arch!=='listwise'){
    let e = lista(c.extra_features);
    if (e.length===1 && e[0]==='all') e = Object.keys(DATA.extra_features).sort();
    extras = Array.from(new Set(e)).sort().join(',');
  }
  const strip = htext ? (c.strip_status ? '1':'0') : '-';
  const maxlen = htext ? String(Math.trunc(c.max_text_len??256)) : '-';
  const nmode = !htab ? '-' : (arch==='listwise' ? 'linear' : (c.numeric_mode||'linear'));
  const nbins = nmode==='bins' ? String(Math.trunc(c.n_bins??16)) : '-';
  const catenc = (!htab || arch==='listwise') ? '-' : (c.cat_encoding||'embedding');
  const pares = (catenc==='-'||catenc==='onehot') ? {} : catfePares(c, catenc);
  const catfe = (catenc==='-'||catenc==='onehot') ? '-'
    : Object.keys(pares).sort().map(f=>`${f}=${pares[f]}`).join(',');
  const buckets = (catenc==='hashing' || Object.values(pares).includes('hashing'))
    ? String(Math.trunc(c.hash_buckets??8)) : '-';
  const pool = arch==='transformer' ? (c.pooling||'cls') : '-';
  const clspos = arch==='transformer' ? (c.cls_position||'first') : '-';
  const posit = arch==='transformer' ? ((form==='text'||form==='hybrid'||c.positional) ? '1':'0') : '-';
  const caus = (arch==='transformer'||arch==='tower') ? (c.causal ? '1':'0') : '-';
  const posw = c.pos_weight ? '1':'0';
  const cart = arch==='listwise' ? '-' : ffloat(c.cart_aux||0);
  const ttok = !htext ? '-' : (arch==='listwise' ? 'chars' : (c.text_tokens||'chars'));
  const w2v = !htext ? '-' : ((ttok==='words' && c.w2v_init) ? '1' : '0');
  const frac = ffloat(c.train_frac||1.0);
  const init = (c.init_seed===null||c.init_seed===undefined) ? '-' : String(Math.trunc(c.init_seed));
  const mlm = String(Math.trunc(c.pretrain_mlm||0));
  const cvk = Math.trunc(c.cv_k||0);
  const cv = cvk ? `${cvk}.${Math.trunc(c.cv_fold||0)}` : '-';
  const pf = (arch==='transformer' && form==='features') ? (c.per_feature||'none') : '-';
  const nh = arch==='mlp' ? '-' : String(Math.trunc(c.n_head??4));
  const nl = arch==='mlp' ? '-' : String(Math.trunc(c.n_layer??2));

  // ---- 6ta tanda: regularizacion / transfer / SIA (None-safe: 0 es valor real) ----
  const numo = (v,d) => (v===null||v===undefined) ? d : v;
  const wd = ffloat(numo(c.weight_decay, 0.01));
  const fdrop = (arch==='transformer' && form==='features') ? ffloat(numo(c.feature_dropout,0)) : '-';
  const lsm = arch==='listwise' ? '-' : ffloat(numo(c.label_smoothing,0));
  const sinres = arch==='transformer' ? (c.sin_residual ? '1':'0') : '-';
  const sinln = arch==='transformer' ? (c.sin_layernorm ? '1':'0') : '-';
  const ifrom = arch==='transformer' ? (c.init_from || '-') : '-';
  const frz = arch==='transformer' ? (c.freeze_backbone ? '1':'0') : '-';
  const rih = ifrom!=='-' ? (c.reinit_head ? '1':'0') : '-';
  const l2sp = ffloat(numo(c.l2sp,0));
  const dst = c.distill_from || '-';
  const dsta = dst!=='-' ? ffloat(numo(c.distill_alpha,1)) : '-';
  const efrom = arch==='mlp' ? (c.embed_from || '-') : '-';
  const som = (arch==='listwise'||arch==='tower'||(arch==='transformer'&&form==='text'))
    ? '-' : String(Math.trunc(numo(c.som_feature,0)));
  const ae = String(Math.trunc(numo(c.pretrain_ae,0)));
  const ael = arch==='mlp' ? String(Math.trunc(numo(c.ae_latent,0))) : '-';
  const pca = arch==='mlp' ? String(Math.trunc(numo(c.pca,0))) : '-';

  // ---- 8va tanda: transfer desde un preentrenado externo ----
  let temb = c.text_emb || '-';
  if (temb!=='-'){ temb = temb.replace(/\\/g,'/').split('/').pop().replace(/\.npy$/,''); }
  const tembft = c.text_emb_finetune || '-';
  const templr = tembft!=='-' ? ffloat(numo(c.text_emb_lr, 1e-5)) : '-';

  return [arch, form, drops, strip, nmode, nbins, String(Math.trunc(c.d_model)), nh, nl,
    ffloat(c.dropout??0.1), pool, posit, caus, posw, maxlen,
    String(Math.trunc(c.epochs??60)), String(Math.trunc(c.batch_size??256)),
    ffloat(c.lr??0.001), String(Math.trunc(c.patience??8)),
    catenc, buckets, clspos, cart, extras, lwtext, catfe, ttok, w2v,
    frac, init, mlm, cv, pf,
    wd, fdrop, lsm, sinres, sinln, ifrom, frz, rih, l2sp, dst, dsta, efrom,
    som, ae, ael, pca, temb, tembft, templr].join('|');
}

// auto-test contra las claves generadas en Python (guardia anti-drift)
(function(){
  let bad = 0;
  for (const [n,v] of Object.entries(DATA.suite)) if (canonKey(v.cfg) !== v.key) bad++;
  for (const [k,g] of Object.entries(DATA.results)) if (canonKey(g.cfg) !== k) bad++;
  if (bad) $('selftest').style.display = 'block';
})();

const suiteByKey = {};
for (const [n,v] of Object.entries(DATA.suite)) suiteByKey[v.key] = n;

// ---- coerciones: que ajusta el codigo solo, segun la arquitectura ----
function coerciones(s){
  const out = [];
  if (s.arch==='transformer' && s.formulation==='text' && s.drop_features.length)
    out.push('formulación text: los features tabulares no entran al modelo — el drop no aplica');
  if (s.arch==='transformer' && (s.formulation==='text'||s.formulation==='hybrid'))
    out.push('con texto el positional encoding va SIEMPRE (el orden de los chars importa)');
  if (!hasText(s) && s.strip_status)
    out.push('strip-status solo aplica a arquitecturas que ven el texto — ignorado');
  if (s.arch==='listwise' && s.numeric_mode==='bins')
    out.push('listwise no implementa bins → usa lineal');
  if (s.arch==='listwise' && s.cat_encoding!=='embedding')
    out.push('listwise no implementa encodings alternativos → embedding');
  if (s.arch==='listwise' && Object.keys(catfeRaw(s.cat_feature_encoding)).length)
    out.push('listwise no implementa encoding por feature → ignorado');
  if (s.cat_encoding==='onehot' && Object.keys(catfeRaw(s.cat_feature_encoding)).length)
    out.push('onehot es global del MLP: los overrides por feature se ignoran');
  if (s.arch==='listwise' && Number(s.cart_aux)>0)
    out.push('cart-aux no implementado para listwise → ignorado');
  if (s.arch==='listwise' && s.extra_features.length)
    out.push('extra-features no implementado para listwise → ignorado');
  if (s.arch==='transformer' && s.causal && s.cls_position==='first' && s.pooling==='cls')
    out.push('⚠ causal + CLS al inicio DEGENERA (medido: p constante, ROC 0.500) — usá CLS al final');
  if (s.arch==='mlp') out.push('MLP: cabezas y bloques no aplican (no hay atención)');
  if (s.arch==='listwise') out.push('listwise: sin positional (no hay orden de página) y batch en queries');
  if (s.arch==='tower') out.push('tower: la torre lleva su PE propio; pooling fijo [CLS] del texto');
  if (s.w2v_init && s.text_tokens!=='words')
    out.push('w2v-init requiere tokens de palabras — ignorado con caracteres');
  if (s.arch==='listwise' && s.listwise_texto && s.text_tokens==='words')
    out.push('listwise-texto es char-level (words no implementado ahí) → chars');
  if (s.arch==='transformer' && s.formulation==='fusion')
    out.push('fusion: el texto entra como UN token-resumen (torre interna); sin PE (sigue siendo un set)');
  if (Number(s.pretrain_mlm)>0 && !(s.arch==='transformer' && s.formulation==='features' && s.cls_position==='first'))
    out.push('⚠ pretrain-mlm es solo para transformer features con CLS al inicio (el comando fallaría)');
  if ((s.per_feature||'none')!=='none' && s.arch==='transformer' && s.formulation==='features' && s.causal)
    out.push('⚠ per-feature no se combina con causal — el comando fallaría');
  if ((Number(s.cv_k)>0 || Number(s.train_frac)<1) && s.arch==='listwise')
    out.push('cv / train-frac no implementados para listwise — el comando fallaría');
  if (Number(s.feature_dropout)>0 && !(s.arch==='transformer' && s.formulation==='features'))
    out.push('feature-dropout es solo para transformer features — ignorado');
  if ((s.sin_residual||s.sin_layernorm) && s.arch!=='transformer')
    out.push('sin-residual / sin-layernorm son ablaciones del transformer — ignoradas');
  if (Number(s.label_smoothing)>0 && (s.arch==='listwise'||Number(s.cart_aux)>0))
    out.push('⚠ label-smoothing no se combina con listwise ni multi-task — el comando fallaría');
  if (s.distill_from && (s.arch==='listwise'||Number(s.cart_aux)>0||Number(s.label_smoothing)>0))
    out.push('⚠ distill no se combina con listwise, cart-aux ni label-smoothing — el comando fallaría');
  if ((s.init_from||s.freeze_backbone) && s.arch!=='transformer')
    out.push('init-from / freeze-backbone están implementados solo para el transformer — ignorados');
  if (Number(s.l2sp)>0 && !(s.init_from||Number(s.pretrain_mlm)>0||Number(s.pretrain_ae)>0))
    out.push('⚠ l2sp ancla a pesos PRE-entrenados: requiere init-from, MLM o AE — el comando fallaría');
  if (s.embed_from && s.arch!=='mlp')
    out.push('embed-from es feature extraction PARA el MLP — ignorado');
  if (s.text_emb && s.text_emb_finetune)
    out.push('⚠ text-emb y text-emb-finetune son excluyentes (congelado O fine-tuning) — el comando fallaría');
  if (s.text_emb && !((s.arch==='transformer'&&s.formulation==='features')||s.arch==='mlp'))
    out.push('⚠ text-emb: transformer features (token extra) o mlp (numéricas extra) — el comando fallaría');
  if (s.text_emb_finetune && !(s.arch==='transformer'&&s.formulation==='features'))
    out.push('⚠ text-emb-finetune: solo transformer formulation features — el comando fallaría');
  if ((Number(s.ae_latent)>0||Number(s.pca)>0) && s.arch!=='mlp')
    out.push('ae-latent / pca reemplazan la entrada del MLP — ignorados');
  if (Number(s.ae_latent)>0 && Number(s.pca)>0)
    out.push('⚠ ae-latent y pca son excluyentes — el comando fallaría');
  if (Number(s.som_feature)>0 && (s.arch==='listwise'||s.arch==='tower'
      ||(s.arch==='transformer'&&s.formulation==='text')||s.cat_encoding==='onehot'))
    out.push('⚠ som-feature necesita la rama tabular sin onehot — el comando fallaría');
  if (Number(s.pretrain_ae)>0 && !(s.arch==='transformer' && s.formulation==='features' && s.cls_position==='first'))
    out.push('⚠ pretrain-ae es solo transformer features con CLS al inicio — el comando fallaría');
  if (Number(s.pretrain_ae)>0 && Number(s.pretrain_mlm)>0)
    out.push('⚠ pretrain-ae y pretrain-mlm: elegir UN pre-entrenamiento — el comando fallaría');
  return out;
}

// ---- pendientes de implementacion (los ejes 🟡) ----
function pendientes(s){
  const out = [];
  if (X.numenc_mixto){
    const mix = Object.entries(X.numpf).filter(([f,m])=>m!=='linear');
    out.push({k:'encoding_numericas_por_feature', v:Object.fromEntries(Object.entries(X.numpf)),
      n:'modo por feature: hoy --numeric-mode es global ('+(mix.map(([f,m])=>f+'→'+m).join(', ')||'todo lineal')+')'});
  }
  if (X.val==='producto') out.push({k:'validacion', v:'split por producto',
    n:'agrupar por producto en vez de query (ya verificado que no cambia métricas)'});
  if (X.mq) out.push({k:'metricas_por_pagina', v:true,
    n:'top-1 por query y NDCG de página en compute_metrics (requiere pasar query_id al evaluate)'});
  return out;
}

// ---- comando CLI (solo flags no-default, espejo de build_parser) ----
function comando(s){
  const p = ['.venv/bin/python','-m','btr.train'];
  if (s.arch!=='transformer') p.push('--arch', s.arch);
  else if (s.formulation!=='features') p.push('--formulation', s.formulation);
  if (s.arch==='listwise' && s.listwise_texto) p.push('--listwise-texto');
  if (hasTab(s) && s.drop_features.length) p.push('--drop-features', Array.from(new Set(s.drop_features)).sort().join(','));
  if (hasTab(s) && s.arch!=='listwise' && s.extra_features.length)
    p.push('--extra-features', Array.from(new Set(s.extra_features)).sort().join(','));
  if (hasText(s) && s.strip_status) p.push('--strip-status');
  if (hasText(s) && s.max_text_len!==256) p.push('--max-text-len', String(s.max_text_len));
  if (hasText(s) && s.arch!=='listwise' && s.text_tokens==='words'){
    p.push('--text-tokens','words');
    if (s.w2v_init) p.push('--w2v-init');
  }
  if (hasTab(s) && s.arch!=='listwise' && s.cat_encoding!=='embedding'){
    p.push('--cat-encoding', s.cat_encoding);
  }
  if (hasTab(s) && s.arch!=='listwise' && s.cat_encoding!=='onehot'){
    const pares = catfePares(s, s.cat_encoding||'embedding');
    const fs = Object.keys(pares).sort();
    if (fs.length) p.push('--cat-feature-encoding', fs.map(f=>`${f}=${pares[f]}`).join(','));
    const usaHash = s.cat_encoding==='hashing' || Object.values(pares).includes('hashing');
    if (usaHash && s.hash_buckets!==8) p.push('--hash-buckets', String(s.hash_buckets));
  }
  if (s.d_model!==32) p.push('--d-model', String(s.d_model));
  if (s.arch!=='mlp' && s.n_head!==4) p.push('--n-head', String(s.n_head));
  if (s.arch!=='mlp' && s.n_layer!==2) p.push('--n-layer', String(s.n_layer));
  if (Number(s.dropout)!==0.1) p.push('--dropout', ffloat(s.dropout));
  if (hasTab(s) && s.arch!=='listwise' && s.numeric_mode==='bins'){
    p.push('--numeric-mode','bins');
    if (s.n_bins!==16) p.push('--n-bins', String(s.n_bins));
  }
  if (s.arch==='transformer' && s.pooling==='mean') p.push('--pooling','mean');
  if (s.arch==='transformer' && s.cls_position==='last') p.push('--cls-position','last');
  if (s.arch==='transformer' && s.formulation==='features' && s.positional) p.push('--positional');
  if ((s.arch==='transformer'||s.arch==='tower') && s.causal) p.push('--causal');
  if (s.pos_weight) p.push('--pos-weight');
  if (s.arch!=='listwise' && Number(s.cart_aux)>0) p.push('--cart-aux', ffloat(s.cart_aux));
  if (Number(s.train_frac)>0 && Number(s.train_frac)<1) p.push('--train-frac', ffloat(s.train_frac));
  if (s.init_seed!==null && s.init_seed!==undefined && s.init_seed!=='') p.push('--init-seed', String(s.init_seed));
  if (Number(s.pretrain_mlm)>0) p.push('--pretrain-mlm', String(s.pretrain_mlm));
  if (Number(s.cv_k)>0){ p.push('--cv-k', String(s.cv_k)); p.push('--cv-fold', String(s.cv_fold||0)); }
  if (s.arch==='transformer' && s.formulation==='features' && (s.per_feature||'none')!=='none')
    p.push('--per-feature', s.per_feature);
  if (Number(s.weight_decay)!==0.01) p.push('--weight-decay', ffloat(s.weight_decay));
  if (s.arch==='transformer' && s.formulation==='features' && Number(s.feature_dropout)>0)
    p.push('--feature-dropout', ffloat(s.feature_dropout));
  if (Number(s.label_smoothing)>0) p.push('--label-smoothing', ffloat(s.label_smoothing));
  if (s.arch==='transformer' && s.sin_residual) p.push('--sin-residual');
  if (s.arch==='transformer' && s.sin_layernorm) p.push('--sin-layernorm');
  if (s.arch==='transformer' && s.init_from){
    p.push('--init-from', `'${s.init_from}'`);
    if (s.reinit_head) p.push('--reinit-head');
  }
  if (s.arch==='transformer' && s.freeze_backbone) p.push('--freeze-backbone');
  if (Number(s.l2sp)>0) p.push('--l2sp', ffloat(s.l2sp));
  if (s.distill_from){
    p.push('--distill-from', `'${s.distill_from}'`);
    if (Number(s.distill_alpha)!==1) p.push('--distill-alpha', ffloat(s.distill_alpha));
  }
  if (s.arch==='mlp' && s.embed_from) p.push('--embed-from', `'${s.embed_from}'`);
  if (s.text_emb) p.push('--text-emb', `'${s.text_emb}'`);
  if (s.text_emb_finetune){
    p.push('--text-emb-finetune', `'${s.text_emb_finetune}'`);
    if (Number(s.text_emb_lr)!==1e-5) p.push('--text-emb-lr', ffloat(s.text_emb_lr));
  }
  if (Number(s.som_feature)>0) p.push('--som-feature', String(s.som_feature));
  if (Number(s.pretrain_ae)>0) p.push('--pretrain-ae', String(s.pretrain_ae));
  if (s.arch==='mlp' && Number(s.ae_latent)>0) p.push('--ae-latent', String(s.ae_latent));
  if (s.arch==='mlp' && Number(s.pca)>0) p.push('--pca', String(s.pca));
  if (s.epochs!==60) p.push('--epochs', String(s.epochs));
  if (s.batch_size!==256) p.push('--batch-size', String(s.batch_size));
  if (Number(s.lr)!==0.001) p.push('--lr', ffloat(s.lr));
  if (s.patience!==8) p.push('--patience', String(s.patience));
  if (X.seeds!==1) p.push('--seeds', String(X.seeds));
  p.push('--save-pesos');
  return p.join(' ');
}

// ---- resultados ----
const mean = a => a.reduce((x,y)=>x+y,0)/a.length;
const std = a => { const m = mean(a); return Math.sqrt(mean(a.map(x=>(x-m)*(x-m)))); };
const f4 = x => (Math.round(x*10000)/10000).toFixed(4);

function chartSVG(g, metric, showTrain, showVal){
  const W=560, H=250, L=52, R=10, T=12, B=26;
  const series = [];
  for (const run of g.runs){
    if (showTrain && run.hist.train[metric]) series.push({v:run.hist.train[metric], c:'ln-train'});
    if (showVal && run.hist.val[metric]) series.push({v:run.hist.val[metric], c:'ln-val'});
  }
  if (!series.length) return '<div class="empty">sin datos para esa métrica</div>';
  const all = series.flatMap(s=>s.v);
  let y0 = Math.min(...all), y1 = Math.max(...all);
  const pad = (y1-y0)||1e-6; y0 -= pad*0.06; y1 += pad*0.06;
  const maxEp = Math.max(...series.map(s=>s.v.length));
  const xs = i => L + (W-L-R)*(maxEp>1 ? i/(maxEp-1) : 0.5);
  const ys = v => T + (H-T-B)*(1-(v-y0)/(y1-y0));
  const flabel = v => { const s = f4(v).replace(/0+$/,'').replace(/\.$/,''); return (s===''||s==='-') ? '0' : s; };
  let out = '';
  for (let k=0;k<=4;k++){
    const v = y0 + (y1-y0)*k/4, y = ys(v);
    out += `<line class="gridl" x1="${L}" y1="${y}" x2="${W-R}" y2="${y}"/>`;
    out += `<text class="axis" x="${L-6}" y="${y+3}" text-anchor="end">${flabel(v)}</text>`;
  }
  for (const t of [0, Math.round((maxEp-1)/2), maxEp-1]){
    out += `<text class="axis" x="${xs(t)}" y="${H-8}" text-anchor="middle">${t}</text>`;
  }
  const line = (vals, cls) => `<polyline class="${cls}" points="${vals.map((v,i)=>xs(i).toFixed(1)+','+ys(v).toFixed(1)).join(' ')}"/>`;
  for (const s of series) out += line(s.v, 'ln '+s.c);
  for (const [sp,cls] of [['train','ln-train'],['val','ln-val']]){
    if ((sp==='train'&&!showTrain)||(sp==='val'&&!showVal)) continue;
    const runs = g.runs.filter(r=>r.hist[sp][metric]);
    if (runs.length>1){
      const m=[]; for (let i=0;i<maxEp;i++){
        const vs = runs.map(r=>r.hist[sp][metric][i]).filter(v=>v!==undefined);
        if (vs.length) m.push(mean(vs));
      }
      out += line(m, 'ln mean '+cls);
    }
  }
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img">${out}</svg>` +
    '<div class="leg">' + (showTrain?'<span><i class="ln-train" style="background:var(--tab)"></i>train (submuestra 4k)</span>':'') +
    (showVal?'<span><i style="background:var(--cls)"></i>val</span>':'') +
    (g.runs.length>1?'<span>línea gruesa = media entre seeds</span>':'') + '</div>';
}

let chartMetric = 'pr_auc', chartTrain = true, chartVal = true;
function renderResultados(key){
  const g = DATA.results[key];
  const head = $('res-head'), chart = $('res-chart'), table = $('res-table');
  if (!g){
    head.innerHTML = '<div class="empty">Sin resultados todavía para esta configuración exacta. '+
      'Corré el comando de arriba (o la suite completa en la 3070) y regenerá el panel con '+
      '<b>python panel.py</b>.</div>';
    chart.innerHTML = ''; table.innerHTML = ''; return;
  }
  const seeds = g.runs.map(r=>r.seed).join(', ');
  const dev = Array.from(new Set(g.runs.map(r=>r.device))).join('/');
  head.innerHTML = `<span class="pill ok">${g.runs.length} corrida(s) · seeds ${esc(seeds)}</span>`+
    `<span class="pill info">${g.runs[0].params.toLocaleString('es-AR')} parámetros · ${esc(dev)}</span>`+
    (g.pesos?'<span class="pill cls">checkpoint en pesos/ ✓</span>':'');
  // el historial de la 1ra tanda GPU (schema viejo) solo trae loss/roc/pr por epoca
  const avail = new Set([...Object.keys(g.runs[0].hist.train||{}), ...Object.keys(g.runs[0].hist.val||{})]);
  if (!avail.has(chartMetric)) chartMetric = avail.has('pr_auc') ? 'pr_auc' : [...avail][0];
  const opts = DATA.metricas.filter(m=>avail.has(m.k))
    .map(m=>`<option value="${m.k}"${m.k===chartMetric?' selected':''}>${esc(m.label)}</option>`).join('');
  const nota = avail.size<6 ? ' <span class="sd">(historial de la 1ª tanda: solo estas 3 por época; las 16 están en la tabla)</span>' : '';
  chart.innerHTML = `<div class="rowc" style="margin:8px 0 0">
      <label class="inl">métrica <select id="chart-metric">${opts}</select></label>
      <label class="inl"><input type="checkbox" id="chart-train"${chartTrain?' checked':''}> train</label>
      <label class="inl"><input type="checkbox" id="chart-val"${chartVal?' checked':''}> val</label>${nota}
    </div><div id="chart-box">${chartSVG(g, chartMetric, chartTrain, chartVal)}</div>`;
  $('chart-metric').onchange = e => { chartMetric = e.target.value; update(); };
  $('chart-train').onchange = e => { chartTrain = e.target.checked; update(); };
  $('chart-val').onchange = e => { chartVal = e.target.checked; update(); };
  let rows = '';
  for (const m of DATA.metricas){
    const va = g.runs.map(r=>r.val[m.k]).filter(v=>v!==undefined);
    const te = g.runs.map(r=>r.test[m.k]).filter(v=>v!==undefined);
    const cell = a => !a.length ? '—'
      : f4(mean(a)) + (a.length>1 ? ` <span class="sd">±${f4(std(a))}</span>` : '');
    rows += `<tr${m.k==='pr_auc'?' class="hl"':''}><td>${esc(m.label)}</td><td>${cell(va)}</td><td>${cell(te)}</td></tr>`;
  }
  table.innerHTML = `<table class="mt"><thead><tr><th>métrica</th><th>val</th><th>test</th></tr></thead><tbody>${rows}</tbody></table>`;
}

let rankMetric = 'pr_auc', rankSplit = 'test';
function renderRanking(currentKey){
  const sel = $('rank-metric');
  if (!sel.options.length){
    sel.innerHTML = DATA.metricas.map(m=>`<option value="${m.k}">${esc(m.label)}</option>`).join('');
    sel.value = rankMetric;
    sel.onchange = e => { rankMetric = e.target.value; update(); };
    $('rank-split').querySelectorAll('button').forEach(b=>b.onclick = () => {
      rankSplit = b.dataset.v;
      $('rank-split').querySelectorAll('button').forEach(x=>x.classList.toggle('on', x===b));
      update();
    });
  }
  const dir = (DATA.metricas.find(m=>m.k===rankMetric)||{}).dir || 'up';
  const rows = Object.entries(DATA.results).map(([k,g])=>{
    const vals = g.runs.map(r=>r[rankSplit][rankMetric]).filter(v=>v!==undefined);
    if (!vals.length) return null;
    return {k, g, m: mean(vals), s: vals.length>1?std(vals):null,
      nm: suiteByKey[k] ? suiteByKey[k]+' (suite)' : g.nombre.replace(/_seed\d+(_\d+)?$/,'')};
  }).filter(Boolean);
  if (!rows.length){ $('rank').innerHTML = '<div class="empty">Todavía no hay corridas en resultados/.</div>'; return; }
  rows.sort((a,b)=> dir==='down' ? a.m-b.m : b.m-a.m);
  const lo = Math.min(...rows.map(r=>r.m)), hi = Math.max(...rows.map(r=>r.m));
  $('rank').innerHTML = rows.map(r=>{
    const w = hi>lo ? 8+92*(dir==='down' ? (hi-r.m)/(hi-lo) : (r.m-lo)/(hi-lo)) : 100;
    return `<div class="rrow${r.k===currentKey?' sel':''}" data-key="${esc(r.k)}" tabindex="0" role="button">
      <div><div class="nm">${esc(r.nm)}</div><div class="bar" style="width:${w.toFixed(0)}%"></div></div>
      <div class="vl">${f4(r.m)}${r.s!==null?` <span class="sd">±${f4(r.s)}</span>`:''}</div></div>`;
  }).join('');
  $('rank').querySelectorAll('.rrow').forEach(el=>{
    const go = () => { applyCfg(DATA.results[el.dataset.key].cfg); };
    el.onclick = go;
    el.onkeydown = e => { if (e.key==='Enter'||e.key===' ') { e.preventDefault(); go(); } };
  });
}

// ---- estado / update ----
function update(){
  // coherencia de ejes segun arquitectura
  const htext = hasText(S), htab = hasTab(S);
  document.querySelectorAll('#arch-grid .opt').forEach(el=>{
    const c = ARCH2CODE[el.dataset.arch];
    el.classList.toggle('on', c.arch===S.arch && (c.arch!=='transformer' || c.formulation===S.formulation));
  });
  $('lwtext-wrap').style.display = S.arch==='listwise' ? '' : 'none';
  $('strip-wrap').style.opacity = htext ? 1 : .45;
  $('strip').disabled = !htext;
  document.querySelectorAll('#feat-grid .feat').forEach(el=>{
    const f = el.dataset.f;
    const on = !S.drop_features.includes(f);
    el.querySelector('input').checked = on;
    el.classList.toggle('off', !on || !htab);
  });
  const extrasOk = htab && S.arch!=='listwise';
  document.querySelectorAll('#extra-grid .feat').forEach(el=>{
    const f = el.dataset.x;
    const on = S.extra_features.includes(f);
    el.querySelector('input').checked = on;
    el.classList.toggle('off', !extrasOk);
  });
  $('nh-wrap').style.opacity = $('nl-wrap').style.opacity = S.arch==='mlp' ? .45 : 1;
  $('pool-wrap').style.display = S.arch==='transformer' ? '' : 'none';
  $('clspos-wrap').style.display = S.arch==='transformer' ? '' : 'none';
  $('pf-wrap').style.display = (S.arch==='transformer' && S.formulation==='features') ? '' : 'none';
  $('pos-wrap').style.display = (S.arch==='transformer' && S.formulation==='features') ? '' : 'none';
  $('caus-wrap').style.display = (S.arch==='transformer'||S.arch==='tower') ? '' : 'none';
  $('nbins-wrap').style.display = S.numeric_mode==='bins' ? '' : 'none';
  const catfeVals = Object.values(catfeRaw(S.cat_feature_encoding));
  $('buckets-wrap').style.display = (S.cat_encoding==='hashing' || catfeVals.includes('hashing')) ? '' : 'none';
  $('catpf-details').style.opacity = (S.arch==='listwise' || S.cat_encoding==='onehot' || !hasTab(S)) ? .45 : 1;
  $('cart-wrap').style.opacity = S.arch==='listwise' ? .45 : 1;
  $('texttok-wrap').style.display = (hasText(S) && S.arch!=='listwise') ? '' : 'none';
  $('w2v-wrap').style.opacity = S.text_tokens==='words' ? 1 : .45;
  $('w2v_init').disabled = S.text_tokens!=='words';
  $('numpf-wrap').style.display = X.numenc_mixto ? '' : 'none';
  $('k-wrap').style.display = X.val==='gkfold' ? '' : 'none';
  const archHints = {
    mlp:'El control sin atención: la atención le saca +0.048 de PR-AUC (gana en 5/6 seeds).',
    listwise:'0.740 test PR con paciencia 20 (fue el más beneficiado: +0.041). listwise_texto responde si pierde por la idea o por no ver el texto.',
    tower:'La mejor textual (0.775) — pero su embedding único NO recupera la señal del regex (sin_regex: −0.04).',
    text:'La familia cara. 0.652: redescubre el tier desde los chars (>> techo intrínseco 0.16) pero no alcanza al tabular. Ojo: paciencia 20 acá EMPEORA test.',
    hybrid:'0.705: los 256 chars diluyen lo tabular. Pero sin_regex ≈ full: recupera desde el texto lo que saca la regex.',
    feat:'La base: 0.794. Campeón global: ordinal + paciencia 20 = 0.824 (feat_ordinal); con embedding, 0.816 (pac20_feat_h1).',
    fusion:'5B de la revisión externa: la torre resume el texto a UN token de la secuencia tabular — cruce texto↔features sin la dilución medida del híbrido (3ª tanda).'};
  const aid = Object.entries(ARCH2CODE).find(([id,c])=>c.arch===S.arch &&
    (c.arch!=='transformer'||c.formulation===S.formulation))[0];
  $('arch-hint').textContent = archHints[aid]||'';
  $('cap-hint').textContent = (S.arch!=='mlp' && S.d_model % S.n_head !== 0)
    ? '⚠ d_model debe ser múltiplo del número de cabezas (head_size = d_model / cabezas)' : '';

  // estado + comando + spec
  const pend = pendientes(S);
  const key = canonKey(S);
  const inSuite = suiteByKey[key];
  const badOnehot = S.cat_encoding==='onehot' && S.arch!=='mlp' && hasTab(S) && S.arch!=='listwise';
  const invalid = (S.arch!=='mlp' && S.d_model % S.n_head !== 0) || badOnehot;
  let pills = '';
  if (invalid) pills += badOnehot
    ? '<span class="pill err">one-hot es solo para MLP: en el transformer, one-hot + lineal ≡ embedding (§6.1)</span>'
    : '<span class="pill err">inválida: d_model % cabezas ≠ 0</span>';
  else if (pend.length) pills += '<span class="pill warn">requiere implementación</span>';
  else if (inSuite) pills += `<span class="pill ok">en la suite: ${esc(inSuite)}</span>`;
  else pills += '<span class="pill ok">soportada por el código</span>';
  if (!pend.length && !invalid && !inSuite)
    pills += '<span class="pill info">no está en la suite — pedile a Claude que la agregue si va en serio</span>';
  $('estado-pills').innerHTML = pills;
  const cmd = invalid ? (badOnehot ? '— elegí arquitectura MLP para one-hot —' : '— arreglá d_model / cabezas —') :
    (inSuite && X.seeds===6 && !pend.length)
      ? `.venv/bin/python experimentos.py --only ${inSuite}\n# (equivale a: ${comando(S)})`
      : comando(S);
  $('cmd').textContent = cmd;
  $('spec-wrap').style.display = pend.length ? '' : 'none';
  if (pend.length){
    $('spec').textContent = 'IMPLEMENTAR ' + JSON.stringify(
      {pedidos: pend.map(p=>({[p.k]: p.v, nota: p.n})), sobre: cmd}, null, 1);
  }
  const notas = coerciones(S).map(c=>`<li>${esc(c)}</li>`).join('');
  $('coerciones').innerHTML = notas;

  renderResultados(key);
  renderRanking(key);
}

function applyCfg(cfg){
  for (const k of Object.keys(DATA.defaults)) if (cfg[k]!==undefined) S[k] = cfg[k];
  S.drop_features = (cfg.drop_features||[]).slice();
  S.extra_features = (cfg.extra_features||[]).slice();
  syncUI(); update();
  window.scrollTo({top:0, behavior:'smooth'});
}

function syncUI(){
  const aid = Object.entries(ARCH2CODE).find(([id,c])=>c.arch===S.arch &&
    (c.arch!=='transformer'||c.formulation===(S.formulation||'features')));
  document.querySelectorAll('#arch-grid input[name=arch]').forEach(r=>r.checked = r.value===aid[0]);
  $('strip').checked = !!S.strip_status;
  $('listwise_texto').checked = !!S.listwise_texto;
  ['d_model','n_head','n_layer','dropout','epochs','batch_size','lr','patience','n_bins',
   'hash_buckets','cart_aux','train_frac','pretrain_mlm',
   'weight_decay','feature_dropout','label_smoothing','l2sp','distill_alpha',
   'som_feature','pretrain_ae','ae_latent','pca'].forEach(k=>{ $(k).value = S[k]; });
  $('init_seed').value = (S.init_seed===null||S.init_seed===undefined) ? '' : S.init_seed;
  $('init_from').value = S.init_from||''; $('distill_from').value = S.distill_from||'';
  $('embed_from').value = S.embed_from||'';
  $('text_emb').value = S.text_emb||''; $('text_emb_finetune').value = S.text_emb_finetune||'';
  $('text_emb_lr').value = S.text_emb_lr;
  $('sin_residual').checked = !!S.sin_residual; $('sin_layernorm').checked = !!S.sin_layernorm;
  $('freeze_backbone').checked = !!S.freeze_backbone; $('reinit_head').checked = !!S.reinit_head;
  $('catenc').value = S.cat_encoding||'embedding';
  const catfeObj = catfeRaw(S.cat_feature_encoding);
  document.querySelectorAll('select[name=catpf]').forEach(sel=>{ sel.value = catfeObj[sel.dataset.f]||''; });
  $('numenc').value = X.numenc_mixto ? 'mixto' : S.numeric_mode;
  $('pool-seg').querySelectorAll('button').forEach(b=>b.classList.toggle('on', b.dataset.v===(S.pooling||'cls')));
  $('ttok-seg').querySelectorAll('button').forEach(b=>b.classList.toggle('on', b.dataset.v===(S.text_tokens||'chars')));
  $('w2v_init').checked = !!S.w2v_init;
  $('clspos-seg').querySelectorAll('button').forEach(b=>b.classList.toggle('on', b.dataset.v===(S.cls_position||'first')));
  $('pf-seg').querySelectorAll('button').forEach(b=>b.classList.toggle('on', b.dataset.v===(S.per_feature||'none')));
  $('positional').checked = !!S.positional; $('causal').checked = !!S.causal;
  $('pos_weight').checked = !!S.pos_weight;
  $('seeds').value = X.seeds;
}

// ---- wiring ----
document.querySelectorAll('#arch-grid input[name=arch]').forEach(r=>r.onchange = () => {
  Object.assign(S, {formulation:'features'}, ARCH2CODE[r.value]);
  if (!ARCH2CODE[r.value].formulation) S.formulation = 'features';
  update();
});
document.querySelectorAll('#feat-grid input[name=feat]').forEach(cb=>cb.onchange = () => {
  const f = cb.value;
  S.drop_features = cb.checked ? S.drop_features.filter(x=>x!==f)
                               : S.drop_features.concat([f]);
  update();
});
document.querySelectorAll('#extra-grid input[name=extraf]').forEach(cb=>cb.onchange = () => {
  const f = cb.value;
  S.extra_features = cb.checked ? S.extra_features.concat([f])
                                : S.extra_features.filter(x=>x!==f);
  update();
});
$('listwise_texto').onchange = e => { S.listwise_texto = e.target.checked; update(); };
$('pre-catalogo').onclick = () => { S.drop_features = []; S.strip_status = false; syncUI(); update(); };
$('pre-intrinseco').onclick = () => { S.drop_features = ['listing_status']; S.strip_status = true; syncUI(); update(); };
$('strip').onchange = e => { S.strip_status = e.target.checked; update(); };
$('catenc').onchange = e => { S.cat_encoding = e.target.value; update(); };
document.querySelectorAll('select[name=catpf]').forEach(sel=>sel.onchange = () => {
  const o = catfeRaw(S.cat_feature_encoding);
  if (sel.value==='') delete o[sel.dataset.f]; else o[sel.dataset.f] = sel.value;
  S.cat_feature_encoding = Object.keys(o).sort().map(f=>f+'='+o[f]).join(',');
  update();
});
$('clspos-seg').querySelectorAll('button').forEach(b=>b.onclick = () => { S.cls_position = b.dataset.v; syncUI(); update(); });
$('pf-seg').querySelectorAll('button').forEach(b=>b.onclick = () => { S.per_feature = b.dataset.v; syncUI(); update(); });
$('cart_aux').onchange = e => { S.cart_aux = parseFloat(e.target.value||0); update(); };
$('ttok-seg').querySelectorAll('button').forEach(b=>b.onclick = () => { S.text_tokens = b.dataset.v; syncUI(); update(); });
$('w2v_init').onchange = e => { S.w2v_init = e.target.checked; update(); };
$('hash_buckets').onchange = e => { S.hash_buckets = parseInt(e.target.value||8,10); update(); };
$('numenc').onchange = e => {
  X.numenc_mixto = e.target.value==='mixto';
  if (!X.numenc_mixto) S.numeric_mode = e.target.value;
  update();
};
document.querySelectorAll('select[name=numpf]').forEach(s=>s.onchange = e => {
  X.numpf[s.dataset.f] = e.target.value; update();
});
[['d_model',1],['n_head',1],['n_layer',1],['dropout',0],['epochs',1],['batch_size',1],
 ['lr',0],['patience',1],['n_bins',1]].forEach(([k,isInt])=>{
  $(k).onchange = e => { S[k] = isInt ? parseInt(e.target.value||0,10) : parseFloat(e.target.value||0); update(); };
});
$('seeds').onchange = e => { X.seeds = parseInt(e.target.value||1,10); update(); };
$('kfold').onchange = e => { X.kfold = parseInt(e.target.value||5,10);
  if (X.val==='gkfold') S.cv_k = X.kfold; update(); };
$('pool-seg').querySelectorAll('button').forEach(b=>b.onclick = () => { S.pooling = b.dataset.v; syncUI(); update(); });
$('positional').onchange = e => { S.positional = e.target.checked; update(); };
$('causal').onchange = e => { S.causal = e.target.checked; update(); };
$('pos_weight').onchange = e => { S.pos_weight = e.target.checked; update(); };
document.querySelectorAll('input[name=val]').forEach(r=>r.onchange = () => {
  X.val = r.value;
  S.cv_k = r.value==='gkfold' ? (X.kfold||5) : 0;
  S.cv_fold = 0;
  document.querySelectorAll('[data-val]').forEach(el=>el.classList.toggle('on', el.dataset.val===X.val));
  update();
});
$('train_frac').onchange = e => { S.train_frac = parseFloat(e.target.value||1); update(); };
$('init_seed').onchange = e => { S.init_seed = e.target.value==='' ? null : parseInt(e.target.value,10); update(); };
$('pretrain_mlm').onchange = e => { S.pretrain_mlm = parseInt(e.target.value||0,10); update(); };
// ---- 6ta tanda: regularizacion / transfer / SIA ----
[['weight_decay',0],['feature_dropout',0],['label_smoothing',0],['l2sp',0],['distill_alpha',0],
 ['som_feature',1],['pretrain_ae',1],['ae_latent',1],['pca',1]].forEach(([k,isInt])=>{
  $(k).onchange = e => { S[k] = isInt ? parseInt(e.target.value||0,10) : parseFloat(e.target.value||0); update(); };
});
['sin_residual','sin_layernorm','freeze_backbone','reinit_head'].forEach(k=>{
  $(k).onchange = e => { S[k] = e.target.checked; update(); };
});
$('text_emb_lr').onchange = e => { S.text_emb_lr = parseFloat(e.target.value||1e-5); update(); };
['init_from','distill_from','embed_from','text_emb','text_emb_finetune'].forEach(k=>{
  $(k).onchange = e => { S[k] = e.target.value.trim(); update(); };
});
$('pre-probe').onclick = () => {
  Object.assign(S, {arch:'transformer', formulation:'features', cat_encoding:'ordinal',
    init_from:CKPT_CAMPEON, freeze_backbone:true, reinit_head:true, dropout:0,
    patience:20, epochs:300, distill_from:''});
  syncUI(); update();
};
$('pre-dst-camp').onclick = () => {
  Object.assign(S, {arch:'transformer', formulation:'features', cat_encoding:'ordinal',
    distill_from:CKPT_CAMPEON, distill_alpha:1, init_from:'', freeze_backbone:false,
    reinit_head:false, patience:20, epochs:300});
  syncUI(); update();
};
$('pre-dst-ens').onclick = () => {
  Object.assign(S, {arch:'transformer', formulation:'features', cat_encoding:'ordinal',
    distill_from:CKPT_ENSEMBLE, distill_alpha:1, init_from:'', freeze_backbone:false,
    reinit_head:false, patience:20, epochs:300});
  syncUI(); update();
};
$('mq').onchange = e => { X.mq = e.target.checked; update(); };
const copiar = (btn, src) => {
  const txt = $(src).textContent;
  const done = ok => { btn.textContent = ok ? '¡copiado!' : 'seleccioná el texto y copialo'; setTimeout(()=>btn.textContent = btn.dataset.t, 1600); };
  btn.dataset.t = btn.textContent;
  if (navigator.clipboard && navigator.clipboard.writeText)
    navigator.clipboard.writeText(txt).then(()=>done(true), ()=>done(false));
  else done(false);
};
$('copy-cmd').onclick = () => copiar($('copy-cmd'), 'cmd');
$('copy-spec').onclick = () => copiar($('copy-spec'), 'spec');
$('met-pills').innerHTML = DATA.metricas.map(m=>{
  const d = m.dir==='up' ? '↑' : m.dir==='down' ? '↓' : '·';
  return `<span class="pill info">${esc(m.label)} ${d}</span>`;
}).join('');
$('gen-date').textContent = 'datos embebidos: ' + DATA.gen.slice(0,16).replace('T',' ') + ' UTC';

syncUI();
update();
'''


def main():
    features, suite, grupos, defaults = cargar_datos()
    html = pagina(features, suite, grupos, defaults)
    out = REPO / 'panel.html'
    out.write_text(html)
    print(f'escrito {out.name} ({len(html) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
