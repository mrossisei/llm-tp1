# llm-tp1 — Predicción de Buy Through Rate con Transformers

Trabajo práctico 1 de 73.69 Large Language Models (ITBA). Predecimos `p(bought | producto,
búsqueda)` por impresión con un **FT-Transformer sobre features**: cada feature del producto es un
token, la atención corre entre las 13 features y un `[CLS]` aprendido, y la salida es una
sigmoide. El BTR de un producto es el promedio de esas probabilidades.

**Modelo final:** transformer de features con encoding ordinal de las categóricas — d_model 32,
4 cabezas, 2 bloques pre-LN, **26.177 parámetros**. PR-AUC de test **0.824 ± 0.018** (6 seeds),
GroupKFold 5×6 0.821 ± 0.012, ensemble 0.834. Varas: GBM 0.762, mejor MLP 0.810, logística
0.698, azar 0.131.

- [`propuesta.md`](propuesta.md): el diseño — formulación del problema, EDA, decisiones y plan de experimentos.
- [`analisis.md`](analisis.md): el análisis de cada tanda de experimentos, con la evidencia y las decisiones.

## Estructura

```
├── supermarket_products.csv   # dataset de eventos de búsqueda (10.000 impresiones, 2.012 búsquedas)
├── btr/                       # el laboratorio: todas las arquitecturas que sostienen la presentación
│   ├── data.py                #   carga, features derivados (listing_status, price_rel), split por query, tensores
│   ├── model.py               #   bloques del transformer (demo de la cátedra) + FT-Transformer, MLP, torre de texto
│   └── train.py               #   entrenamiento, early stopping por PR-AUC de validación, 16 métricas, multi-seed
├── experimentos.py            # la suite curada (118 configs × 6 seeds), resumible, con --resumen y --plan
├── modelo_final/              # SOLO el modelo final, autocontenido: data.py, model.py, train.py, predecir.py, pesos/
├── eda/                       # los scripts que producen los números y figuras de la presentación
│   ├── verificaciones.py      #   reproduce el EDA y los baselines (logística, GBM, azar)
│   ├── graficos.py            #   todas las figuras de la presentación -> salidas/graficos/
│   ├── atencion.py            #   mapas de atención del modelo final (graficos.py los usa)
│   ├── calibracion.py         #   reliability diagram + ECE
│   ├── ensemble.py            #   ensemble de configuraciones (0.834)
│   ├── deep_ensemble.py       #   ensemble de inicializaciones y descomposición de la varianza
│   ├── cross_manual.py        #   logística + cruce precio×tier a mano (¿la atención solo descubrió eso?)
│   └── metricas_pagina.py     #   top-1 por página: el producto que el modelo pone primero
└── salidas/
    ├── resultados/            # un JSON por corrida: config + curvas train/val por época + val/test finales
    ├── pesos/                 # checkpoints .pt recargables (btr.model.load_checkpoint)
    └── graficos/              # las figuras de la presentación
```

`salidas/resultados/` conserva **todas** las corridas del proyecto (1.030 al cierre de la 10ª tanda),
incluidas las de variantes que no entraron a la presentación (listwise, multi-task con cart,
pesos por feature, regularización, transfer learning, SOM/PCA/autoencoder, MiniLM, destilación);
el código de esas variantes se retiró de `btr/` y `experimentos.py`, y su análisis está en
`analisis.md`.

## Setup

```bash
uv venv .venv
# CPU:
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
# GPU (máquina con NVIDIA): instalar torch SIN el index de CPU
#   uv pip install --python .venv/bin/python torch
#   verificar: .venv/bin/python -c "import torch; print(torch.cuda.is_available())"  -> True
uv pip install --python .venv/bin/python -r requirements.txt
```

## El modelo final

```bash
cd modelo_final
python predecir.py                      # carga el checkpoint, métricas de test + top-10 productos por BTR
python train.py --seeds 6               # reproduce el entrenamiento (GPU: minutos; CPU: ~1-2 min por seed)
```

## El laboratorio

```bash
.venv/bin/python -m btr.train --cat-encoding ordinal --patience 20 --epochs 300   # el modelo final, seed 42
.venv/bin/python -m btr.train --help                                             # arquitecturas, formulaciones y ejes
```

Cada corrida escribe `salidas/resultados/<nombre>.json` y, con `--save-pesos`,
`salidas/pesos/<nombre>.pt`. Disciplina: los hiperparámetros se eligen mirando **validación**;
test se reporta solo para las configuraciones finales.

```python
from btr.model import load_checkpoint
model, prep = load_checkpoint('salidas/pesos/feat_ordinal_features_d32_h4_l2_linear_catordinal_seed42.pt')
x_cat, x_num, x_text, _ = prep.transform(df_nuevo)  # mismas transformaciones que en train
probs = model.predict_proba(x_cat, x_num, x_text)   # p(bought) por fila
```

### La suite de experimentos (correr en la máquina con GPU)

```bash
.venv/bin/python experimentos.py                     # corre TODO lo que falte (resumible)
.venv/bin/python experimentos.py --plan              # qué falta, sin correr
.venv/bin/python experimentos.py --only 'gc_*,gl_*'  # solo esos (admite comodines)
.venv/bin/python experimentos.py --resumen           # media ± desvío por configuración
```

Usa la GPU automáticamente y aborta con instrucciones si la familia texto fuera a correr en CPU.
Cada bloque de `experimentos.py` dice qué diapositiva sostiene. Al terminar una tanda: commitear
`salidas/resultados/` y `salidas/pesos/` y pushear.

### Las figuras y los números de la presentación

```bash
.venv/bin/python eda/graficos.py          # todas las figuras -> salidas/graficos/
.venv/bin/python eda/verificaciones.py    # EDA y baselines
.venv/bin/python eda/calibracion.py       # ECE del modelo final
.venv/bin/python eda/metricas_pagina.py   # top-1 por página
```

| diapositiva | de dónde sale |
|---|---|
| EDA, baselines, familia intrínseca | `eda/verificaciones.py`; `feat_intrinseco` / `pac20_feat_intrinseco` (~0.16) |
| Exp. 1 — ¿la atención aporta? | `feat_base` vs `mlp_base`, `mlp_onehot`, `mlp_ordinal` + `mlp_ord_*` (`graficos/mlp.png`), `eda/cross_manual.py` |
| Exp. 2 — cabezas y d_model × cabezas | `gc_o_*`, `gc_e_*`, `feat_ordinal`, `camp_ordinal_h1`, `pac20_feat_h1/h2/base/d64`, `camp_d64h1` (`grilla.png`, `cabezas.png`) |
| Exp. 3 — bloques | `camp_ordinal_l4`, `min_l1`, `pac20_feat_l1/l4` (`bloques.png`); `gl_o_*` (`grilla_bloques.png`) |
| Exp. 4 — d_model | `min_d8`, `min_d16`, `min_d16l1`, `pac20_feat_d64` (`dmodel.png`) |
| Exp. 5 — encoding de categóricas | `feat_ordinal`, `feat_target`, `feat_freq`, `feat_hash8`, `mlp_onehot`, `pac20_feat_base` (`encoding.png`) |
| Exp. 6 — inicialización | `feat_mlm20`, `feat_ordinal_mlm20`, `fusion_words`, `fusion_words_w2v` (`init.png`) |
| Exp. 7 — texto / alternativas | `text_base`, `hybrid_full`, `hybrid_sin_regex`, `fusion_base`, `pac20_tower_base`, `ing_*` (`formulacion.png`) |
| Modelo final, robustez | `cv5_fold*`, `curva_frac*`, `robu_init*`, `eda/ensemble.py`, `eda/deep_ensemble.py`, `eda/calibracion.py` |
| Interpretabilidad | `eda/graficos.py` (`atencion.png`, `importancia.png` por permutación), `eda/metricas_pagina.py` |
| Desafíos (hipótesis refutadas) | `feat_causal` / `feat_causal_last`, `feat_bins`, `feat_pos`, `feat_mean`, `feat_posweight` |

## Cobertura del enunciado

| pide el enunciado | dónde está |
|---|---|
| Ej. 1 — variable objetivo, EDA, features y preprocesamiento | `propuesta.md` §1–3 y §6; `eda/verificaciones.py`; `btr/data.py` |
| Ej. 2 — dónde va el transformer y por qué; alternativas | features-como-tokens (`btr/model.py`); texto, híbrido, fusión, torre e ingredientes implementados y corridos |
| Ej. 2 — partición train/valid/test | por `query_id` 70/15/15 (`btr/data.py`), GroupKFold 5×6 al cierre |
| Ej. 2 — experimentos y ablaciones | `experimentos.py` (6 seeds cada uno), análisis en `analisis.md` |
| Ej. 2 — evaluación (PR-AUC / ROC-AUC, over/underfitting) | 16 métricas por época y por split en cada JSON de `salidas/resultados/` |
| Ej. 3 — personalización | una diapositiva de la presentación |
