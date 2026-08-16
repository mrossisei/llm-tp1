# llm-tp1

Trabajo práctico 1 de la materia LLM (73.69): predicción de Buy Through Rate en un
e-commerce de supermercado con un modelo basado en Transformers.

**➡️ Ver [`propuesta.md`](propuesta.md):** formulación del problema, hallazgos del EDA,
decisiones de diseño con alternativas, plan de experimentos y ablaciones.
En particular §5.1 tiene el recorrido de punta a punta (qué entra, qué sale) con un ejemplo real.

## Estructura

```
├── propuesta.md               # análisis del problema y plan (leer primero)
├── supermarket_products.csv   # dataset de eventos de búsqueda
├── btr/
│   ├── data.py                # carga, features derivados, split por query, tensores
│   ├── model.py               # bloques del transformer + FeatureTokenizer + BTRTransformer
│   └── train.py               # entrenamiento, early stopping, métricas, multi-seed, guardado
├── resultados/                # un JSON por corrida (config + curvas train/val + val/test finales)
├── pesos/                     # checkpoints .pt (solo con --save-pesos)
└── notebooks/                 # (próximamente) EDA y experimentos
```

## Setup

```bash
uv venv .venv
# CPU:
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
# GPU (máquina con NVIDIA, ej. RTX 3070): instalar torch sin el index de CPU
#   uv pip install --python .venv/bin/python torch
uv pip install --python .venv/bin/python -r requirements.txt
```

## Uso

```bash
.venv/bin/python -m btr.train                          # formulación tabular (seed 42, device auto)
.venv/bin/python -m btr.train --seeds 3                # promedio de 3 corridas
.venv/bin/python -m btr.train --save-pesos --tag final # guarda checkpoint en pesos/
.venv/bin/python -m btr.train --help                   # flags de ablación (positional, causal, bins, ...)
```

### La suite de experimentos (correr en la máquina con GPU)

Todas las arquitecturas y ablaciones del TP están codificadas en `experimentos.py`
(24 configuraciones, ver `--list`). En la máquina con GPU:

```bash
.venv/bin/python experimentos.py                     # toda la suite, 3 seeds por config
.venv/bin/python experimentos.py --familia texto     # solo la familia cara (GPU)
.venv/bin/python experimentos.py --only text_base,hybrid_sin_regex --save-pesos
.venv/bin/python experimentos.py --resumen           # tabla comparativa de resultados/
```

Arquitecturas (`--arch` en `btr.train`): `transformer` (formulaciones `features`/`text`/`hybrid`),
`mlp` (baseline sin atención), `tower` (transformer solo como encoder de texto → embedding + MLP),
`listwise` (los productos de la página como tokens). Ejes: `--drop-features listing_status`
(sin el estado parseado) y `--strip-status` (texto sin estado: el modelo "producto nuevo", §2.3.1
de la propuesta).

Costos medidos: familia tabular ≈ 30 s por corrida en CPU; familia texto (secuencia ~257) ≈ 74
s/época en CPU (40–90 min por corrida) → en GPU queda en minutos. La suite entera es inviable en
CPU por la familia texto: para eso está la RTX 3070 (plan B: Colab).

Cada corrida escribe `resultados/<nombre>.json` (config, historial por época en train/val,
métricas finales). Disciplina: los hiperparámetros se eligen mirando **validación**; test se
reporta solo para las configuraciones finales. Con `--save-pesos` el checkpoint se recarga así:

```python
from btr.model import load_checkpoint
model, prep = load_checkpoint('pesos/features_d32_h4_l2_linear_seed42.pt')
x_cat, x_num, x_text, _ = prep.transform(df_nuevo)  # mismas transformaciones que en train
probs = model.predict_proba(x_cat, x_num, x_text)   # p(bought) por fila
```

Referencia rápida (test, split por query): regresión logística PR-AUC ≈ 0.72 ·
transformer tabular (3 seeds) ROC-AUC 0.972 ± 0.004, PR-AUC 0.815 ± 0.037 ·
MLP baseline 0.715 y listwise 0.664 (seed 42) · sin información de estado el techo se
desploma (GBM: PR-AUC 0.167 — ver §2.3.1 de la propuesta) · `text`/`hybrid`/`tower`:
pendientes de entrenar en GPU.
