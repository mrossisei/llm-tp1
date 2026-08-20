# llm-tp1

Trabajo práctico 1 de la materia LLM (73.69): predicción de Buy Through Rate en un
e-commerce de supermercado con un modelo basado en Transformers.

**➡️ Ver [`propuesta.md`](propuesta.md):** formulación del problema, hallazgos del EDA,
decisiones de diseño con alternativas, plan de experimentos y ablaciones.
En particular §5.1 tiene el recorrido de punta a punta (qué entra, qué sale) con un ejemplo real.

## Estructura

```
├── propuesta.md               # documento de diseño: análisis del problema y plan (leer primero)
├── bitacora.md                # registro cronológico de discusiones y decisiones
├── diagramas.md               # diagramas de las arquitecturas (GitHub los renderiza)
├── analisis.md                # análisis de las tandas GPU: hallazgos y decisiones, con evidencia
├── zoo.py / zoo.html          # genera el Zoo (los diagramas SVG, publicable como artifact)
├── supermarket_products.csv   # dataset de eventos de búsqueda
├── btr/
│   ├── data.py                # carga, features derivados, split por query, tensores
│   ├── model.py               # bloques del transformer + las 4 arquitecturas
│   └── train.py               # entrenamiento, early stopping, métricas, multi-seed, guardado
├── experimentos.py            # suite completa (24 configs × seeds), resumible, con --resumen
├── panel.py                   # genera panel.html: laboratorio interactivo de experimentos
├── panel.html                 # ese laboratorio, con la suite y los resultados embebidos
├── eda/verificaciones.py      # reproduce todos los números del EDA y los baselines
├── resultados/                # un JSON por corrida (config + curvas train/val + val/test finales)
└── pesos/                     # checkpoints .pt recargables (la suite los guarda por defecto)
```

## Setup

```bash
uv venv .venv
# CPU:
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
# GPU (máquina con NVIDIA, ej. RTX 3070): instalar torch SIN el index de CPU
#   uv pip install --python .venv/bin/python torch
#   verificar: .venv/bin/python -c "import torch; print(torch.cuda.is_available())"  -> True
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
(67 configuraciones, ver `--list`). En la máquina con la RTX 3070, **estas dos líneas hacen todo**
(resume automático: solo corre lo que falte):

```bash
.venv/bin/python experimentos.py                     # corre TODA la suite (configs × 6 seeds) en la GPU
.venv/bin/python experimentos.py --resumen           # tabla comparativa: media ± desvío por config
```

Garantías: usa la GPU automáticamente (imprime el nombre de la placa al arrancar) y **aborta con
instrucciones si detectara que la familia texto correría en CPU** (típicamente torch instalado en
versión CPU); es **resumible** — si se corta, relanzar la misma línea continúa donde quedó
(saltea toda corrida cuyo JSON ya esté en `resultados/`); guarda los checkpoints en `pesos/` por
defecto (`--no-pesos` para no hacerlo); si una corrida falla, sigue con el resto y lo reporta al
final. Al terminar: commitear `resultados/` y `pesos/` y pushear.

Otras variantes: `--familia texto` (solo lo caro), `--only text_base,hybrid_sin_regex`, `--list`.

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

Referencia rápida (PR-AUC test, 6 seeds — análisis completo en [`analisis.md`](analisis.md)):
**MODELO FINAL: `feat_ordinal` 0.824 ± 0.018** (ensemble: **0.834** por dos rutas, §9.1/§10.4 · CV 5×6: 0.821 ± 0.012, §10.5) (transformer tabular d32/h4/l2 + categóricas como
su rango de BTR de train, 26k parámetros; elegido por empate en validación + parsimonia, §8.2) ·
`pac20_feat_h1` 0.816 · mlp_onehot 0.797 · tabular base 0.794 · fusión = tower 0.775 · GBM 0.762 ·
listwise_texto 0.749 · listwise 0.740 · hybrid 0.705–0.735 · logística 0.660 · text 0.652 ·
sin información de estado el techo se desploma a ~0.16 (familia intrínseca, §2.3.1). Baselines
reproducibles con `eda/verificaciones.py`.

## El laboratorio interactivo (`panel.html`)

La consola del proyecto: se compone una configuración eligiendo cada decisión (arquitectura,
features, encodings, capacidad, validación) y la página muestra el **comando exacto**, si la
combinación **ya está en la suite** o si **requiere implementación** (da un spec para pedirla), y
— si ya se corrió — **todas las métricas** (media ± desvío entre seeds), las **curvas por época**
de cualquier métrica y el **ranking** de todo lo corrido. Es HTML estático autocontenido: se abre
localmente o como artifact. Después de cada tanda de experimentos:

```bash
.venv/bin/python panel.py     # re-embebe resultados/ y la suite en panel.html
```

Cada corrida guarda las 16 métricas por época (`compute_metrics` en `btr/train.py`), así que el
panel grafica cualquiera sin reentrenar.
