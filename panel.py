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

from btr.data import CAT_FEATURES, NUM_FEATURES, Preprocessor, load_dataset, split_by_query
from btr.train import build_parser
from experimentos import EXPERIMENTOS

REPO = Path(__file__).resolve().parent


def ffloat(x):
    return f'{float(x):.10f}'.rstrip('0').rstrip('.')


def canon(c):
    """Clave canonica de una config (dict con los nombres de argparse).

    Normaliza los campos que no aplican a la arquitectura elegida (los pone en
    '-') para que dos configs que producen EL MISMO modelo tengan la misma clave
    aunque difieran en flags irrelevantes. Espejada en JS como canonKey().
    """
    arch = c['arch']
    form = c.get('formulation', 'features') if arch == 'transformer' else '-'
    has_text = (arch == 'transformer' and form in ('text', 'hybrid')) or arch == 'tower'
    has_tab = not (arch == 'transformer' and form == 'text')
    if has_tab:
        drop = c.get('drop_features', '') or ''
        if isinstance(drop, str):
            drop = [f.strip() for f in drop.split(',') if f.strip()]
        drops = ','.join(sorted(set(drop)))
    else:
        drops = '-'
    strip = ('1' if c.get('strip_status') else '0') if has_text else '-'
    maxlen = str(int(c.get('max_text_len', 256))) if has_text else '-'
    nmode = '-' if not has_tab else ('linear' if arch == 'listwise' else c.get('numeric_mode', 'linear'))
    nbins = str(int(c.get('n_bins', 16))) if nmode == 'bins' else '-'
    pool = c.get('pooling', 'cls') if arch == 'transformer' else '-'
    if arch == 'transformer':
        posit = '1' if (form in ('text', 'hybrid') or c.get('positional')) else '0'
    else:
        posit = '-'
    caus = ('1' if c.get('causal') else '0') if arch in ('transformer', 'tower') else '-'
    posw = '1' if c.get('pos_weight') else '0'
    nh = '-' if arch == 'mlp' else str(int(c.get('n_head', 4)))
    nl = '-' if arch == 'mlp' else str(int(c.get('n_layer', 2)))
    return '|'.join([
        arch, form, drops, strip, nmode, nbins, str(int(c['d_model'])), nh, nl,
        ffloat(c.get('dropout', 0.1)), pool, posit, caus, posw, maxlen,
        str(int(c.get('epochs', 60))), str(int(c.get('batch_size', 256))),
        ffloat(c.get('lr', 1e-3)), str(int(c.get('patience', 8))),
    ])


CFG_FIELDS = ['arch', 'formulation', 'drop_features', 'strip_status', 'max_text_len',
              'numeric_mode', 'n_bins', 'd_model', 'n_head', 'n_layer', 'dropout',
              'pooling', 'positional', 'causal', 'pos_weight', 'epochs', 'batch_size',
              'lr', 'patience']


def cfg_dict(c):
    """Config reducida a los campos que definen el modelo, con drop como lista."""
    out = {k: c.get(k) for k in CFG_FIELDS}
    drop = out.get('drop_features') or ''
    if isinstance(drop, str):
        drop = [f.strip() for f in drop.split(',') if f.strip()]
    out['drop_features'] = sorted(set(drop))
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

    return f'''
<section class="card"><h2><span class="n">1</span>Arquitectura</h2>
  <fieldset class="optgrid" id="arch-grid">{archs}</fieldset>
  <p class="hint" id="arch-hint"></p>
</section>

<section class="card"><h2><span class="n">2</span>Entrada — qué features mandamos</h2>
  <div class="rowc" style="margin:0 0 8px">
    <button class="preset" id="pre-catalogo">preset: catálogo (todo)</button>
    <button class="preset" id="pre-intrinseco">preset: intrínseco (producto nuevo)</button>
  </div>
  <fieldset class="featgrid" id="feat-grid">{feats}</fieldset>
  <div class="rowc">
    <label class="inl" id="strip-wrap"><input type="checkbox" id="strip">
      --strip-status: borrar del TEXTO el sufijo del título y la última oración</label>
  </div>
  <p class="hint">El preset <b>intrínseco</b> = sacar <code>listing_status</code> + strip del texto:
  simula el producto nuevo sin historial (techo GBM: 0.762 → 0.162). Destildar features acá
  se traduce a <code>--drop-features</code>.</p>
</section>

<section class="card"><h2><span class="n">3</span>Encodings</h2>
  <div class="rowc">
    <label class="inl">categóricas <select id="catenc">
      <option value="embedding">embedding aprendido por columna (actual)</option>
      <option value="onehot">one-hot directo al MLP (pedir)</option>
      <option value="target">target encoding por nivel (pedir)</option>
    </select></label>
    <label class="inl">numéricas <select id="numenc">
      <option value="linear">lineal: x·w+b por feature</option>
      <option value="bins">bins por cuantiles</option>
      <option value="mixto">elegir por feature (pedir)</option>
    </select></label>
    <label class="inl" id="nbins-wrap">n_bins <input type="number" id="n_bins" value="16" min="2" max="64"></label>
  </div>
  <div class="rowc" id="numpf-wrap" style="display:none">{numpf}</div>
  <p class="hint">Lo que el código hace hoy para categóricas es exactamente <b>column embeddings</b>
  (una tabla por columna — seguramente el "columnar" que te dijeron). <b>one-hot + capa lineal
  aprende la misma matriz</b> (propuesta §6.1), por eso no existe como opción aparte del
  transformer; sí tiene sentido como entrada cruda a un MLP, para medir qué aporta embeber.
  <b>Target encoding</b> (nivel → BTR promedio en train) es fuerte pero con riesgo de leakage:
  se ajusta por fold. <b>Bins</b> captura la U invertida del precio sin depender del FFN.</p>
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
    <label class="inl" id="pos-wrap"><input type="checkbox" id="positional"> positional encoding</label>
    <label class="inl" id="caus-wrap"><input type="checkbox" id="causal"> máscara causal (ablación)</label>
    <label class="inl"><input type="checkbox" id="pos_weight"> pos_weight en la BCE (87/13)</label>
  </div>
  <p class="hint" id="cap-hint"></p>
</section>

<section class="card"><h2><span class="n">5</span>Validación y entrenamiento</h2>
  <fieldset class="optgrid">
    <label class="opt on" data-val="holdout"><input type="radio" name="val" value="holdout" checked>
      <span class="t">holdout por query</span><span class="d">70/15/15 × N seeds (actual)</span></label>
    <label class="opt" data-val="gkfold"><input type="radio" name="val" value="gkfold">
      <span class="t">GroupKFold por query</span><span class="d">k folds agrupados (pedir)</span></label>
    <label class="opt" data-val="producto"><input type="radio" name="val" value="producto">
      <span class="t">split por producto</span><span class="d">verificado: da igual (§7.1) (pedir)</span></label>
  </fieldset>
  <div class="rowc">
    <label class="inl">seeds <input type="number" id="seeds" value="3" min="1" max="10"></label>
    <label class="inl" id="k-wrap" style="display:none">k <input type="number" id="kfold" value="5" min="3" max="10"></label>
    <label class="inl">épocas <input type="number" id="epochs" value="60" min="1"></label>
    <label class="inl">batch <input type="number" id="batch_size" value="256" min="16" step="16"></label>
    <label class="inl">lr <input type="number" id="lr" value="0.001" min="0" step="0.0005"></label>
    <label class="inl">patience <input type="number" id="patience" value="8" min="1"></label>
  </div>
  <p class="hint">La decisión de parar y de elegir configs se toma SIEMPRE con la validación
  (PR-AUC); test solo se reporta. Con 2.012 queries el holdout × 3 seeds ya da desvíos chicos;
  GroupKFold es el paso siguiente si queremos intervalos más finos.</p>
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
  tower:{arch:'tower'}, listwise:{arch:'listwise'}};
const ALL_FEATS = DATA.features.cat.map(f=>f.name).concat(DATA.features.num.map(f=>f.name));

// ---- estado (mismos nombres que argparse en btr/train.py) ----
const S = Object.assign({}, DATA.defaults);
S.drop_features = (S.drop_features||[]).slice();
const X = {catenc:'embedding', numenc_mixto:false, numpf:{}, val:'holdout', kfold:5, seeds:3, mq:false};
DATA.features.num.forEach(f => X.numpf[f.name] = 'linear');

const hasText = s => (s.arch==='transformer' && (s.formulation==='text'||s.formulation==='hybrid')) || s.arch==='tower';
const hasTab  = s => !(s.arch==='transformer' && s.formulation==='text');

// ---- clave canonica: ESPEJO EXACTO de canon() en panel.py ----
function canonKey(c){
  const arch = c.arch;
  const form = arch==='transformer' ? (c.formulation||'features') : '-';
  const htext = (arch==='transformer' && (form==='text'||form==='hybrid')) || arch==='tower';
  const htab = !(arch==='transformer' && form==='text');
  let drop = c.drop_features||[];
  if (typeof drop === 'string') drop = drop.split(',').map(s=>s.trim()).filter(Boolean);
  const drops = htab ? Array.from(new Set(drop)).sort().join(',') : '-';
  const strip = htext ? (c.strip_status ? '1':'0') : '-';
  const maxlen = htext ? String(Math.trunc(c.max_text_len??256)) : '-';
  const nmode = !htab ? '-' : (arch==='listwise' ? 'linear' : (c.numeric_mode||'linear'));
  const nbins = nmode==='bins' ? String(Math.trunc(c.n_bins??16)) : '-';
  const pool = arch==='transformer' ? (c.pooling||'cls') : '-';
  const posit = arch==='transformer' ? ((form==='text'||form==='hybrid'||c.positional) ? '1':'0') : '-';
  const caus = (arch==='transformer'||arch==='tower') ? (c.causal ? '1':'0') : '-';
  const posw = c.pos_weight ? '1':'0';
  const nh = arch==='mlp' ? '-' : String(Math.trunc(c.n_head??4));
  const nl = arch==='mlp' ? '-' : String(Math.trunc(c.n_layer??2));
  return [arch, form, drops, strip, nmode, nbins, String(Math.trunc(c.d_model)), nh, nl,
    ffloat(c.dropout??0.1), pool, posit, caus, posw, maxlen,
    String(Math.trunc(c.epochs??60)), String(Math.trunc(c.batch_size??256)),
    ffloat(c.lr??0.001), String(Math.trunc(c.patience??8))].join('|');
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
  if (s.arch==='mlp') out.push('MLP: cabezas y bloques no aplican (no hay atención)');
  if (s.arch==='listwise') out.push('listwise: sin positional (no hay orden de página) y batch en queries');
  if (s.arch==='tower') out.push('tower: la torre lleva su PE propio; pooling fijo [CLS] del texto');
  return out;
}

// ---- pendientes de implementacion (los ejes 🟡) ----
function pendientes(s){
  const out = [];
  if (X.catenc==='onehot') out.push({k:'encoding_categoricas', v:'one-hot directo al MLP',
    n: s.arch==='mlp' ? 'one-hot crudo al MLP (medir qué aporta embeber)' :
       'one-hot: para el transformer equivale al embedding (§6.1) — elegila con arquitectura MLP'});
  if (X.catenc==='target') out.push({k:'encoding_categoricas', v:'target encoding',
    n:'nivel → BTR promedio de train (ajustado por fold para no filtrar el target)'});
  if (X.numenc_mixto){
    const mix = Object.entries(X.numpf).filter(([f,m])=>m!=='linear');
    out.push({k:'encoding_numericas_por_feature', v:Object.fromEntries(Object.entries(X.numpf)),
      n:'modo por feature: hoy --numeric-mode es global ('+(mix.map(([f,m])=>f+'→'+m).join(', ')||'todo lineal')+')'});
  }
  if (X.val==='gkfold') out.push({k:'validacion', v:'GroupKFold k='+X.kfold,
    n:'k-fold agrupado por query en lugar del holdout'});
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
  if (hasTab(s) && s.drop_features.length) p.push('--drop-features', Array.from(new Set(s.drop_features)).sort().join(','));
  if (hasText(s) && s.strip_status) p.push('--strip-status');
  if (hasText(s) && s.max_text_len!==256) p.push('--max-text-len', String(s.max_text_len));
  if (s.d_model!==32) p.push('--d-model', String(s.d_model));
  if (s.arch!=='mlp' && s.n_head!==4) p.push('--n-head', String(s.n_head));
  if (s.arch!=='mlp' && s.n_layer!==2) p.push('--n-layer', String(s.n_layer));
  if (Number(s.dropout)!==0.1) p.push('--dropout', ffloat(s.dropout));
  if (hasTab(s) && s.arch!=='listwise' && s.numeric_mode==='bins'){
    p.push('--numeric-mode','bins');
    if (s.n_bins!==16) p.push('--n-bins', String(s.n_bins));
  }
  if (s.arch==='transformer' && s.pooling==='mean') p.push('--pooling','mean');
  if (s.arch==='transformer' && s.formulation==='features' && s.positional) p.push('--positional');
  if ((s.arch==='transformer'||s.arch==='tower') && s.causal) p.push('--causal');
  if (s.pos_weight) p.push('--pos-weight');
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
  const opts = DATA.metricas.map(m=>`<option value="${m.k}"${m.k===chartMetric?' selected':''}>${esc(m.label)}</option>`).join('');
  chart.innerHTML = `<div class="rowc" style="margin:8px 0 0">
      <label class="inl">métrica <select id="chart-metric">${opts}</select></label>
      <label class="inl"><input type="checkbox" id="chart-train"${chartTrain?' checked':''}> train</label>
      <label class="inl"><input type="checkbox" id="chart-val"${chartVal?' checked':''}> val</label>
    </div><div id="chart-box">${chartSVG(g, chartMetric, chartTrain, chartVal)}</div>`;
  $('chart-metric').onchange = e => { chartMetric = e.target.value; update(); };
  $('chart-train').onchange = e => { chartTrain = e.target.checked; update(); };
  $('chart-val').onchange = e => { chartVal = e.target.checked; update(); };
  let rows = '';
  for (const m of DATA.metricas){
    const va = g.runs.map(r=>r.val[m.k]), te = g.runs.map(r=>r.test[m.k]);
    const cell = a => f4(mean(a)) + (a.length>1 ? ` <span class="sd">±${f4(std(a))}</span>` : '');
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
    const vals = g.runs.map(r=>r[rankSplit][rankMetric]);
    return {k, g, m: mean(vals), s: vals.length>1?std(vals):null,
      nm: suiteByKey[k] ? suiteByKey[k]+' (suite)' : g.nombre.replace(/_seed\d+(_\d+)?$/,'')};
  });
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
  $('strip-wrap').style.opacity = htext ? 1 : .45;
  $('strip').disabled = !htext;
  document.querySelectorAll('#feat-grid .feat').forEach(el=>{
    const f = el.dataset.f;
    const on = !S.drop_features.includes(f);
    el.querySelector('input').checked = on;
    el.classList.toggle('off', !on || !htab);
  });
  $('nh-wrap').style.opacity = $('nl-wrap').style.opacity = S.arch==='mlp' ? .45 : 1;
  $('pool-wrap').style.display = S.arch==='transformer' ? '' : 'none';
  $('pos-wrap').style.display = (S.arch==='transformer' && S.formulation==='features') ? '' : 'none';
  $('caus-wrap').style.display = (S.arch==='transformer'||S.arch==='tower') ? '' : 'none';
  $('nbins-wrap').style.display = S.numeric_mode==='bins' ? '' : 'none';
  $('numpf-wrap').style.display = X.numenc_mixto ? '' : 'none';
  $('k-wrap').style.display = X.val==='gkfold' ? '' : 'none';
  const archHints = {mlp:'El control sin atención: si el transformer no le gana, la atención no se justifica.',
    listwise:'Ve la página completa; único capaz de modelar competencia entre productos.',
    tower:'Tu propuesta: transformer como encoder de texto, MLP como clasificador.',
    text:'La familia cara (atención 257²): corre en la 3070.',
    hybrid:'La atención puede cruzar texto ↔ features en la misma capa.',
    feat:'La ganadora hasta ahora en CPU: PR-AUC 0.766 vs GBM 0.762.'};
  const aid = Object.entries(ARCH2CODE).find(([id,c])=>c.arch===S.arch &&
    (c.arch!=='transformer'||c.formulation===S.formulation))[0];
  $('arch-hint').textContent = archHints[aid]||'';
  $('cap-hint').textContent = (S.arch!=='mlp' && S.d_model % S.n_head !== 0)
    ? '⚠ d_model debe ser múltiplo del número de cabezas (head_size = d_model / cabezas)' : '';

  // estado + comando + spec
  const pend = pendientes(S);
  const key = canonKey(S);
  const inSuite = suiteByKey[key];
  const invalid = S.arch!=='mlp' && S.d_model % S.n_head !== 0;
  let pills = '';
  if (invalid) pills += '<span class="pill err">inválida: d_model % cabezas ≠ 0</span>';
  else if (pend.length) pills += '<span class="pill warn">requiere implementación</span>';
  else if (inSuite) pills += `<span class="pill ok">en la suite: ${esc(inSuite)}</span>`;
  else pills += '<span class="pill ok">soportada por el código</span>';
  if (!pend.length && !invalid && !inSuite)
    pills += '<span class="pill info">no está en la suite — pedile a Claude que la agregue si va en serio</span>';
  $('estado-pills').innerHTML = pills;
  const cmd = invalid ? '— arreglá d_model / cabezas —' :
    (inSuite && X.seeds===3 && !pend.length)
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
  syncUI(); update();
  window.scrollTo({top:0, behavior:'smooth'});
}

function syncUI(){
  const aid = Object.entries(ARCH2CODE).find(([id,c])=>c.arch===S.arch &&
    (c.arch!=='transformer'||c.formulation===(S.formulation||'features')));
  document.querySelectorAll('#arch-grid input[name=arch]').forEach(r=>r.checked = r.value===aid[0]);
  $('strip').checked = !!S.strip_status;
  ['d_model','n_head','n_layer','dropout','epochs','batch_size','lr','patience','n_bins'].forEach(k=>{ $(k).value = S[k]; });
  $('numenc').value = X.numenc_mixto ? 'mixto' : S.numeric_mode;
  $('pool-seg').querySelectorAll('button').forEach(b=>b.classList.toggle('on', b.dataset.v===(S.pooling||'cls')));
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
$('pre-catalogo').onclick = () => { S.drop_features = []; S.strip_status = false; syncUI(); update(); };
$('pre-intrinseco').onclick = () => { S.drop_features = ['listing_status']; S.strip_status = true; syncUI(); update(); };
$('strip').onchange = e => { S.strip_status = e.target.checked; update(); };
$('catenc').onchange = e => { X.catenc = e.target.value; update(); };
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
$('kfold').onchange = e => { X.kfold = parseInt(e.target.value||5,10); update(); };
$('pool-seg').querySelectorAll('button').forEach(b=>b.onclick = () => { S.pooling = b.dataset.v; syncUI(); update(); });
$('positional').onchange = e => { S.positional = e.target.checked; update(); };
$('causal').onchange = e => { S.causal = e.target.checked; update(); };
$('pos_weight').onchange = e => { S.pos_weight = e.target.checked; update(); };
document.querySelectorAll('input[name=val]').forEach(r=>r.onchange = () => {
  X.val = r.value;
  document.querySelectorAll('[data-val]').forEach(el=>el.classList.toggle('on', el.dataset.val===X.val));
  update();
});
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
