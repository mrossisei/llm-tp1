# llm-tp1 — Predicción de Buy Through Rate con Transformers

Trabajo práctico 1 de 73.69 Large Language Models (ITBA). Predecimos `p(bought | producto,
búsqueda)` por impresión con un **FT-Transformer sobre features**: cada feature del producto es un
token, la atención corre entre las 13 features y un `[CLS]` aprendido, y la salida es una
sigmoide. El BTR de un producto es el promedio de esas probabilidades.

**Modelo final:** transformer de features con encoding ordinal de las categóricas — d_model 32,
4 cabezas, 2 bloques pre-LN, **26.177 parámetros**. PR-AUC de test **0.824 ± 0.018** (6 seeds),
GroupKFold 5×6 0.821 ± 0.012, ensemble 0.834. Varas: regresión logística 0.698, azar 0.131.

- [`propuesta.md`](propuesta.md): el diseño — formulación del problema, EDA, decisiones y plan de experimentos.
- [`analisis.md`](analisis.md): el análisis de cada tanda de experimentos, con la evidencia y las decisiones.

## Estructura

```
├── supermarket_products.csv   # dataset de eventos de búsqueda (10.000 impresiones, 2.012 búsquedas)
├── btr/                       # el laboratorio
│   ├── data.py                #   carga, features derivados (listing_status, price_rel), split por query, tensores
│   ├── model.py               #   bloques del transformer (demo de la cátedra), FT-Transformer, encoder de ingredientes
│   └── train.py               #   entrenamiento, early stopping por PR-AUC de validación, 16 métricas, multi-seed
├── experimentos.py            # la suite curada (138 configs × 6 seeds), resumible, con --resumen y --plan
├── modelo_final/              # SOLO el modelo final, autocontenido: data.py, model.py, train.py, predecir.py, pesos/
├── eda/                       # los scripts que producen los números y figuras de la presentación
│   ├── verificaciones.py      #   reproduce el EDA y los baselines (logística, azar)
│   ├── graficos.py            #   todas las figuras de la presentación -> salidas/graficos/
│   ├── embed_titulos.py       #   transfer learning: embeddings del título con modelos preentrenados
│   ├── atencion.py            #   mapas de atención del modelo final (graficos.py los usa)
│   ├── calibracion.py         #   reliability diagram + ECE
│   ├── ensemble.py            #   ensemble de configuraciones (0.834)
│   ├── deep_ensemble.py       #   ensemble de inicializaciones y descomposición de la varianza
│   └── metricas_pagina.py     #   top-1 por página: el producto que el modelo pone primero
└── salidas/
    ├── resultados/            # un JSON por corrida: config + curvas train/val por época + val/test finales
    ├── pesos/                 # checkpoints .pt recargables (btr.model.load_checkpoint)
    ├── graficos/              # las figuras de la presentación
    └── embeddings/            # embeddings del título (float16) de los 3 modelos preentrenados
```

`salidas/resultados/` conserva **todas** las corridas del proyecto, incluidas las de variantes que
no entraron a la presentación (MLP y GBM como baselines, texto crudo / híbrido / fusión / torre,
listwise, multi-task con cart, pesos por feature, transfer interno, SOM/PCA/autoencoder); el
código de esas variantes se retiró de `btr/` y `experimentos.py`, y su análisis está en `analisis.md`.

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
.venv/bin/python -m btr.train --help                                             # formulaciones y ejes
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

Los experimentos son **barridos** encadenados: en cada uno nos quedamos con la configuración de
mayor PR-AUC de validación y el siguiente parte de ella (`MEJOR_ARQ`, una sola línea de
`experimentos.py`; hoy d_model 32, 16 cabezas, 4 bloques, ordinal). Una corrida equivalente ya
hecha (mismo nombre canónico, aunque tenga otro tag) se reutiliza y no se repite. Al terminar una
tanda: commitear `salidas/resultados/` y pushear (los checkpoints de `salidas/pesos/` quedan solo
en la máquina local).

| experimento | barrido | configs | figura |
|---|---|---|---|
| Capacidad | d_model {32, 64, 128, 256} × cabezas {1, 2, 4, 8, 16}, por encoding (ordinal / embedding) | `gc_o_*`, `gc_e_*` + celdas previas | `grilla.png` |
| Profundidad | d_model {32, 64, 128, 256} × bloques {1, 2, 4, 8}, con las cabezas que mejor dieron | `gl_o_*` | `grilla_bloques.png` |
| Encoding de las categóricas | {ordinal, embedding, target} × d_model {32, 64, 128}, con las cabezas y bloques de la ganadora | `enc_*` | `encoding.png` |
| Pre-entrenamiento MLM | épocas {0, 5, 10, 20, 40} × encoding, sobre la ganadora | `mlm_*` | `mlm.png` |
| Optimización | learning rate {1e-4, 3e-4, 1e-3} × batch {64, 128, 256}, sobre la ganadora | `opt_*` | `optimizacion.png` |
| Ingredientes (la alternativa) | encoder de conjunto chico / base / grande vs sin ingredientes, sobre la ganadora; control solo ingredientes | `ing_*` | `ingredientes.png` |
| Transfer learning | título sin badge embebido por MiniLM-L6 / mpnet-base / bge-large (congelados), MiniLM fine-tuneado, solo el título; sobre la ganadora | `tl_*` | `transfer.png` |
| Tiempo | hora y día de la semana como (sin, cos) o como categóricas, vs sin tiempo; sobre la ganadora | `tiempo_*` | `tiempo.png` |
| Robustez del modelo final | curva de aprendizaje, 6 splits × 6 inits, GroupKFold 5×6, calibración, ensembles | `curva_*`, `robu_*`, `cv5_*` | `curva_aprendizaje.png` |
| Hipótesis refutadas | causal (y causal con CLS al final), bins, positional, mean pooling, pos_weight | `pac20_feat_*`, `feat_causal_last` | — |

### Transfer learning con el título

Los tres modelos preentrenados **no se entregan**: `eda/embed_titulos.py` los baja de Hugging Face
(`sentence-transformers`) y guarda en `salidas/embeddings/` el embedding del título de cada
producto, sin el sufijo de estado. Las corridas `tl_*` congeladas leen esas matrices; el
fine-tuning (`--text-emb-finetune`) necesita `transformers` y GPU.

### Las figuras y los números de la presentación

```bash
.venv/bin/python eda/graficos.py          # todas las figuras -> salidas/graficos/
.venv/bin/python eda/verificaciones.py    # EDA y baselines
.venv/bin/python eda/calibracion.py       # ECE del modelo final
.venv/bin/python eda/metricas_pagina.py   # top-1 por página
```

## Cobertura del enunciado

| pide el enunciado | dónde está |
|---|---|
| Ej. 1 — variable objetivo, EDA, features y preprocesamiento | `propuesta.md` §1–3 y §6; `eda/verificaciones.py`; `btr/data.py` |
| Ej. 2 — dónde va el transformer y por qué; alternativas | features-como-tokens (`btr/model.py`); encoder de ingredientes; título preentrenado como token |
| Ej. 2 — partición train/valid/test | por `query_id` 70/15/15 (`btr/data.py`), GroupKFold 5×6 al cierre |
| Ej. 2 — experimentos y ablaciones | `experimentos.py` (barridos, 6 seeds cada uno), análisis en `analisis.md` |
| Ej. 2 — evaluación (PR-AUC / ROC-AUC, over/underfitting) | 16 métricas por época y por split en cada JSON de `salidas/resultados/` |
| Ej. 3 — personalización | una diapositiva de la presentación |
