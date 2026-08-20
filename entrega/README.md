# Entrega — TP1: Predicción de Buy Through Rate con Transformers

**73.69 Large Language Models · 2026** — Fer Rossi · Junior Rambau · Matias Rossi Seifert

Esta carpeta contiene la entrega curada: el modelo final con su código, los experimentos con su
análisis, y la presentación con su guión. El resto del repositorio es el laboratorio completo
(544 corridas crudas en `/resultados`, todas las arquitecturas alternativas en `/btr`, el panel
interactivo `/panel.html`, la bitácora de decisiones) — está disponible para auditar cualquier
número, pero lo que se defiende es lo que está acá.

> **Hash del commit de la entrega**: el que figura en Campus (`git log -1` sobre `main`).

## Resumen ejecutivo

- **Formulación**: clasificación binaria por impresión — `p(bought | producto, búsqueda)`; el
  BTR por producto es la agregación de esas probabilidades (ranking, sin umbral).
- **Modelo final**: **FT-Transformer con encoding ordinal de categóricas** — 13 feature-tokens +
  [CLS], d_model 32, 4 cabezas, 2 bloques, sin positional encoding, **26.177 parámetros**.
- **Resultados** (PR-AUC de test, 6 seeds): **0.824 ± 0.018** · GroupKFold 5×6: 0.821 ± 0.012 ·
  ensemble: 0.834 · ROC-AUC 0.975 · F1 máx 0.784 @ umbral 0.40. Referencias: GBM 0.762 ·
  mejor MLP 0.797 · logística 0.698 · azar 0.131.
- **Auditable**: la atención del [CLS] se concentra 0.75 en el estado del listing y la
  importancia por permutación lo confirma; calibración ECE ≈ 0.01 (el promedio de p por producto
  es directamente el BTR estimado); el top-1 de cada página resulta comprado el 91% de las veces.

## Estructura

```
entrega/
├── modelo/                  # SOLO la arquitectura final, autocontenida
│   ├── data.py              # carga + features derivados + split por query + encoding ordinal
│   ├── model.py             # el transformer (bloques de la demo + tokenizador ordinal)
│   ├── train.py             # métricas (las 16) + entrenamiento; reproduce los pesos
│   ├── predecir.py          # uso de negocio: p(bought) y ranking de productos por BTR
│   └── pesos/               # 6 checkpoints (uno por seed), formato plano sin pickles de clases
├── experimentos/
│   ├── analisis.md          # el análisis completo: cada experimento, su porqué y su conclusión
│   ├── resumen_resultados.csv  # las 91 configuraciones agregadas (media ± desvío, 6 seeds)
│   └── graficos/            # curva de aprendizaje, atención, importancia, calibración, resumen
└── presentacion/
    ├── presentacion.html    # 20 diapositivas autocontenidas (abrir en el navegador; ← →)
    ├── guion.md             # guión hablado con tiempos (~27 min) + apéndice para preguntas
    └── generar.py           # regenera las diapositivas
```

## Cómo correr

```bash
# desde entrega/modelo/ (requiere el venv del repo: torch, pandas, numpy, scikit-learn)
python predecir.py                      # carga el mejor checkpoint, métricas + top-10 por BTR
python train.py --seeds 6               # reproduce el entrenamiento completo (GPU: minutos)
python train.py --seeds 1 --device cpu  # una corrida en CPU (~1-2 min)
```

Nota de reproducibilidad: los pesos entregados provienen de la suite corrida en GPU (verificados
equivalentes en predicción contra el pipeline del repo, tolerancia 1e-6). Reentrenar produce
trayectorias distintas corrida a corrida — la varianza de entrenamiento tiene cola izquierda
(medida en la grilla de robustez: min 0.718, max 0.870) y es exactamente la razón del protocolo
de **promediar 6 seeds** en lugar de reportar corridas sueltas.

## Cobertura del enunciado

| pide el enunciado | dónde está |
|---|---|
| **Ej. 1** — variable objetivo | formulación por impresión + agregación (`analisis.md`, slides 2) |
| Ej. 1 — características / distribución / calidad | EDA: tiers del sufijo, U del precio, funnel de cart, timestamps rotos, redundancias (slides 3–4) |
| Ej. 1 — qué features y qué preprocesamiento (sug.: one-hot) | tabla feature→preprocesamiento (slide 5); la codificación de categóricas fue un experimento dedicado con 6 técnicas comparadas — one-hot incluida y su equivalencia con embedding demostrada (slide 13) |
| **Ej. 2** — dónde va el transformer y por qué | features-como-tokens porque la señal es relacional; alternativas implementadas y medidas (slides 8–9) |
| Ej. 2 — partición train/valid/test | por query_id 70/15/15, con la justificación de por qué no fila/temporal/producto (slide 7) |
| Ej. 2 — experimentos / ablación (base chica, d_model<100) | base d=32; grillas de capacidad, encodings, texto, y ablaciones (pooling, PE, causal, pos_weight, bins) — 544 corridas, 6 seeds c/u (`resumen_resultados.csv`) |
| Ej. 2 — evaluación con PR-AUC / ROC-AUC, over/underfitting, sin umbral | PR-AUC principal + ROC + 14 más, curvas train/val por época en cada corrida, early stopping por validación; sin umbral (F1 máx @ 0.40 lo justifica) (slide 7) |
| Ej. 2 — "uno o más modelos", comparación de alternativas de módulos | 7 arquitecturas + variantes corridas; comparación apareada por seed en `analisis.md` |
| Ej. 2 — conexión con la teoría (clases 1 y 2) | bloques de la demo con 2 adaptaciones justificadas; [CLS] estilo BERT; word2vec con negative sampling probado como init (clase 1→2); tokenizador chico vindicado |
| **Ej. 3** — personalización (1 diapositiva, <5 min) | slide 19: historial como tokens con cross-attention (BST/SASRec) + embeddings con negative sampling + cold-start (2.5 min de guión) |
| Presentación 25–30 min: problema, decisiones, resultados, desafíos, conclusiones | `presentacion/` — 20 diapositivas, guión de ~27 min con desafíos (slide 15) y conclusiones (slide 20) |
| Entrega: repo + README + hash + presentación | este README; hash en Campus |
