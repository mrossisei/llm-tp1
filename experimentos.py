"""Suite curada de experimentos del TP (propuesta.md 7.4).

USO EN LA MAQUINA CON GPU (dos lineas):
    .venv/bin/python experimentos.py              # corre TODA la suite (121 configs x 6 seeds)
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

# ---- 4ta tanda (18/08): robustez y caracterizacion del MODELO FINAL ----
# La busqueda de arquitectura ya convergio (feat_ordinal, analisis.md 8.2); esta
# tanda no intenta superarlo (salvo MLM): lo interroga. Todo tabular = barato.
ORD = ['--cat-encoding', 'ordinal', *PAC]  # la config exacta del modelo final
EXPERIMENTOS |= {
    # curva de aprendizaje: ¿el 0.824 esta saturado o mas datos ayudarian?
    # (100% = feat_ordinal, ya corrido)
    'curva_frac25': ([*ORD, '--train-frac', '0.25'], 'tabular'),
    'curva_frac50': ([*ORD, '--train-frac', '0.5'], 'tabular'),
    'curva_frac75': ([*ORD, '--train-frac', '0.75'], 'tabular'),
    # varianza: mismo split, otra inicializacion -> ¿cuanto del ±0.018 es del
    # split y cuanto del modelo? Ademas habilita el deep-ensemble puro (promediar
    # las 6 inits de cada split). Grilla: 6 splits x (init original + estas 5).
    'robu_init43': ([*ORD, '--init-seed', '43'], 'tabular'),
    'robu_init44': ([*ORD, '--init-seed', '44'], 'tabular'),
    'robu_init45': ([*ORD, '--init-seed', '45'], 'tabular'),
    'robu_init46': ([*ORD, '--init-seed', '46'], 'tabular'),
    'robu_init47': ([*ORD, '--init-seed', '47'], 'tabular'),
    # MLM sobre features (revision externa): pre-entrenar el tronco enmascarando
    # una columna por fila. ¿El pre-training regulariza como el ordinal?
    'feat_mlm20':         (['--pretrain-mlm', '20', *PAC], 'tabular'),
    'feat_ordinal_mlm20': ([*ORD, '--pretrain-mlm', '20'], 'tabular'),
    # GroupKFold 5: cada query pasa por test una vez por seed -> intervalos finos
    'cv5_fold0': ([*ORD, '--cv-k', '5', '--cv-fold', '0'], 'tabular'),
    'cv5_fold1': ([*ORD, '--cv-k', '5', '--cv-fold', '1'], 'tabular'),
    'cv5_fold2': ([*ORD, '--cv-k', '5', '--cv-fold', '2'], 'tabular'),
    'cv5_fold3': ([*ORD, '--cv-k', '5', '--cv-fold', '3'], 'tabular'),
    'cv5_fold4': ([*ORD, '--cv-k', '5', '--cv-fold', '4'], 'tabular'),
}

# ---- 5ta tanda (21/08): pesos POR FEATURE dentro del transformer (idea de Fer) ----
# En texto, compartir W_q/W_k/W_v y la FFN entre posiciones es el sesgo inductivo
# correcto (posiciones intercambiables). Aca la posicion ES el feature: desatar
# los pesos (cada feature con su propio W_q/W_k/W_v y/o su propia FFN) es la
# extension natural de la identidad-por-parametros que ya usamos en la ENTRADA.
# Hipotesis registrada ANTES de correr: multiplica parametros (26k -> 106k/245k/
# 323k) y todo el TP dice que en 10k filas gana el prior simple -> lo esperable
# es que NO supere a feat_ordinal por overfitting; correrlo cierra la pregunta
# "¿hace falta el weight-tying del transformer cuando el conjunto es fijo?".
EXPERIMENTOS |= {
    'pf_qkv':      ([*ORD, '--per-feature', 'qkv'], 'tabular'),
    'pf_ffn':      ([*ORD, '--per-feature', 'ffn'], 'tabular'),
    'pf_full':     ([*ORD, '--per-feature', 'both'], 'tabular'),
    # sobre embeddings: ¿el desatado suple la identidad que el ordinal ya inyecta?
    'pf_full_emb': (['--per-feature', 'both', *PAC], 'tabular'),

    # ---- el contrapeso (idea de Fer, 21/08): reducir complejidad ----
    # (a) minimalismo sobre el campeon: nunca probamos ACHICAR sobre ordinal
    # (la grilla d8/d16 vieja era sobre embeddings). ¿Hasta donde aguanta el
    # prior simple? min_d16 tiene 6.945 parametros.
    'min_d16':    ([*ORD, '--d-model', '16'], 'tabular'),
    'min_d8':     ([*ORD, '--d-model', '8'], 'tabular'),
    'min_l1':     ([*ORD, '--n-layer', '1'], 'tabular'),
    'min_d16l1':  ([*ORD, '--d-model', '16', '--n-layer', '1'], 'tabular'),
    # (b) especializacion BARATA: compuertas diagonales por posicion sobre los
    # W compartidos (init=1 -> arranca siendo exactamente el campeon); ~+11k params
    'pf_gate':    ([*ORD, '--per-feature', 'gate'], 'tabular'),
    # (c) desatado COMPENSADO: per-feature qkv pero d16 -> 26.913 parametros,
    # la misma escala que el campeon (26.177). Comparacion controlada:
    # misma cantidad de parametros, ¿especializar o compartir?
    'pf_qkv_d16': ([*ORD, '--per-feature', 'qkv', '--d-model', '16'], 'tabular'),
}

# ---- 6ta tanda (22/08): regularizacion + transfer learning (clase 3) + SIA ----
# La regularizacion EFECTIVA del TP hasta aca fue early stopping + capacidad
# (min_*) + el prior del ordinal: weight decay (1e-2, default de AdamW) y
# dropout (0.1) quedaron FIJOS en todas las corridas y nunca se barrieron;
# residuales y LayerNorm vienen de los bloques de la demo y nunca se
# ablacionaron. La parte de transfer implementa las TRES tecnicas de la clase 3
# (feature extraction / fine-tuning / knowledge distillation) con NUESTROS
# propios modelos como base (el enunciado pide entrenar el transformer, no
# bajar uno preentrenado). Hipotesis registradas ANTES de correr: analisis.md §13.
CH = 'pesos/feat_ordinal_features_d32_h4_l2_linear_catordinal_seed{seed}.pt'
ENS = CH + ',pesos/robu_init4*_features_d32_h4_l2_linear_catordinal_init4*_seed{seed}.pt'
EXPERIMENTOS |= {
    # (a) regularizacion sobre el campeon — ¿estabamos en el punto justo sin saberlo?
    'reg_wd0':     ([*ORD, '--weight-decay', '0'], 'tabular'),
    'reg_wd1e3':   ([*ORD, '--weight-decay', '0.001'], 'tabular'),
    'reg_wd1e1':   ([*ORD, '--weight-decay', '0.1'], 'tabular'),
    'reg_do0':     ([*ORD, '--dropout', '0'], 'tabular'),
    'reg_do03':    ([*ORD, '--dropout', '0.3'], 'tabular'),
    # sin NINGUNA regularizacion explicita: ¿alcanza el early stopping solo?
    'reg_nada':    ([*ORD, '--weight-decay', '0', '--dropout', '0'], 'tabular'),
    # dropout a nivel TOKEN (features enteras anuladas, primo del MLM)
    'reg_fdrop01': ([*ORD, '--feature-dropout', '0.1'], 'tabular'),
    'reg_fdrop02': ([*ORD, '--feature-dropout', '0.2'], 'tabular'),
    # label smoothing = distillation con teacher uniforme (conexion clase 3)
    'reg_ls01':    ([*ORD, '--label-smoothing', '0.1'], 'tabular'),
    # ablaciones de lo que la demo trae "de fabrica": ¿que aportan a esta escala?
    'abl_sinres':  ([*ORD, '--sin-residual'], 'tabular'),
    'abl_sinln':   ([*ORD, '--sin-layernorm'], 'tabular'),

    # (b) transfer learning (clase 3) con nuestros checkpoints como teachers
    # feature extraction: tronco del campeon CONGELADO + cabeza nueva (probe
    # lineal): ¿la representacion aprendida es linealmente separable?
    'tl_probe':        ([*ORD, '--init-from', CH, '--freeze-backbone',
                         '--reinit-head', '--dropout', '0'], 'tabular'),
    # probe sobre tronco pre-entrenado SOLO self-supervised (MLM, sin ver labels):
    # sobre embeddings (donde el MLM aporto +0.011) y sobre ordinal
    'tl_mlm_probe':     (['--pretrain-mlm', '20', '--freeze-backbone',
                          '--dropout', '0', *PAC], 'tabular'),
    'tl_mlm_probe_ord': ([*ORD, '--pretrain-mlm', '20', '--freeze-backbone',
                          '--dropout', '0'], 'tabular'),
    # fine-tuning ANCLADO: L2 hacia los pesos post-MLM (la "KL penalty" de la
    # clase 3 en version L2-SP) — ¿retiene lo pre-entrenado sin frenar la tarea?
    'tl_mlm_l2sp':      (['--pretrain-mlm', '20', '--l2sp', '0.001', *PAC], 'tabular'),
    # knowledge distillation: entrenar contra las PROBABILIDADES del teacher
    # (soft labels > labels duras, el 0.8 informa mas que el 1) — self-distill,
    # compresion al modelo de 3.7k params, y el deep-ensemble (0.833) como
    # teacher n=6 ("integrando informacion de varios modelos")
    'tl_distill_same':    ([*ORD, '--distill-from', CH], 'tabular'),
    'tl_distill_min':     ([*ORD, '--d-model', '16', '--n-layer', '1',
                            '--distill-from', CH], 'tabular'),
    'tl_distill_ens':     ([*ORD, '--distill-from', ENS], 'tabular'),
    'tl_distill_ens_min': ([*ORD, '--d-model', '16', '--n-layer', '1',
                            '--distill-from', ENS], 'tabular'),
    'tl_distill_mix':     ([*ORD, '--distill-from', ENS, '--distill-alpha', '0.5'], 'tabular'),
    # "el embedding mas un monton de cosas": pooled del campeon congelado como
    # 32 numericas extra del MLP — ¿el MLP alcanza al transformer con su embedding?
    'tl_emb_mlp':         (['--arch', 'mlp', *ORD, '--embed-from', CH], 'tabular'),

    # (c) herramientas de SIA en este problema
    # Kohonen: celda BMU del SOM (train, no supervisado) como categorica extra
    'sia_som16':     ([*ORD, '--som-feature', '4'], 'tabular'),
    'sia_som64':     ([*ORD, '--som-feature', '8'], 'tabular'),
    # autoencoder como PRE-ENTRENAMIENTO del tronco (CLS reconstruye la fila);
    # el hermano con cuello de botella del MLM — misma comparacion emb/ordinal
    'sia_ae_cls':     ([*ORD, '--pretrain-ae', '20'], 'tabular'),
    'sia_ae_cls_emb': (['--pretrain-ae', '20', *PAC], 'tabular'),
    # representation learning puro (clase 3): PCA (la version cerrada de Oja) y
    # el espacio latente de un AE como UNICA entrada del MLP
    'sia_pca_mlp':    (['--arch', 'mlp', '--pca', '16', *PAC], 'tabular'),
    'sia_ae_mlp':     (['--arch', 'mlp', '--ae-latent', '16', *PAC], 'tabular'),
}

# ---- 7ma mini-tanda (23/08): el piso de la compresion, con y sin teacher ----
# Cierra la UNICA pregunta abierta que dejaron la 5ta y la 6ta juntas. Puntos
# que ya tenemos: sin teacher 26.177 -> 0.8239 | 3.713 -> 0.8254 | 1.937 ->
# 0.8142 (aca empieza a degradar); con teacher (deep-ensemble 0.833) solo
# 26.177 -> 0.8272 y 3.713 -> 0.8274. Esta tanda completa la curva "PR vs
# parametros" en dos ramas (plain vs destilada) bajando hasta 353 parametros:
# min_d8l1 1.089 / min_d4l1 353 + las versiones destiladas de 1.937/1.089/353.
# Hipotesis registrada ANTES de correr (analisis.md 13.2): las soft labels
# corren el piso ~un nivel hacia abajo (d8 con teacher ~= d16 sin), porque
# regularizan justo donde la capacidad empieza a faltar; si NO lo corren, el
# "dark knowledge" no compra compresion en este problema — ambas salidas
# cierran la figura. La seleccion sigue cerrada: esto es la curva final de
# "conocimiento vs parametros", no una busqueda de campeon.
EXPERIMENTOS |= {
    'min_d8l1':            ([*ORD, '--d-model', '8', '--n-layer', '1'], 'tabular'),
    'min_d4l1':            ([*ORD, '--d-model', '4', '--n-layer', '1'], 'tabular'),
    'tl_distill_ens_d8':   ([*ORD, '--d-model', '8', '--distill-from', ENS], 'tabular'),
    'tl_distill_ens_d8l1': ([*ORD, '--d-model', '8', '--n-layer', '1',
                             '--distill-from', ENS], 'tabular'),
    'tl_distill_ens_d4l1': ([*ORD, '--d-model', '4', '--n-layer', '1',
                             '--distill-from', ENS], 'tabular'),
}

# ---- 8va tanda (23/08): transfer desde un preentrenado EXTERNO (idea de Fer) ----
# Todo el transfer previo fue con NUESTROS checkpoints; esta tanda agrega el caso
# canonico de la clase 3: MiniLM (sentence-transformers, 22M params) como encoder
# de title+description. El transformer sigue siendo propio — el preentrenado solo
# aporta el embedding del texto, que entra como UN token extra (proyeccion
# aprendida 384->32), el mismo mecanismo de fusion pero con encoder ajeno.
# Dos regimenes: feature extraction (embeddings/*.npy precomputados con
# eda/embed_texto.py, CONGELADOS — los .npy ya estan commiteados) y fine-tuning
# (el encoder entra al grafo con lr 1e-5; requiere `uv pip install transformers`
# en la 3070 y baja el modelo de HF la primera vez). Hipotesis: analisis.md §15.
TEMB = 'embeddings/minilm.npy'        # texto completo (el sufijo de estado incluido)
TEMBI = 'embeddings/minilm_intr.npy'  # texto SIN estado (strip_status_from_text)
HF = 'sentence-transformers/all-MiniLM-L6-v2'
EXPERIMENTOS |= {
    # ¿cuanto ve el preentrenado por si solo? (referencias: logistica cruda
    # 0.660, text_base 0.652, tower 0.775, campeon 0.824)
    'bert_solo':      (['--arch', 'mlp', '--drop-features', 'all',
                        '--text-emb', TEMB, *PAC], 'tabular'),
    # ...y sin el sufijo de estado (referencia: techo intrinseco 0.16)
    'bert_solo_intr': (['--arch', 'mlp', '--drop-features', 'all',
                        '--text-emb', TEMBI, *PAC], 'tabular'),
    # el embedding como 384 numericas extra del MLP (hermano de tl_emb_mlp)
    'bert_mlp':       (['--arch', 'mlp', *ORD, '--text-emb', TEMB], 'tabular'),
    # campeon + token BERT congelado (feature extraction pura)
    'bert_token':     ([*ORD, '--text-emb', TEMB], 'tabular'),
    # ¿el preentrenado CONGELADO reemplaza al regex? (hybrid lo logro end-to-end)
    'bert_token_sin': ([*ORD, '--drop-features', 'listing_status',
                        '--text-emb', TEMB], 'tabular'),
    # fine-tuning: el encoder se ACTUALIZA (lr 1e-5, batch 128 por memoria);
    # familia 'texto' para que el guard de GPU los proteja
    'bert_ft':        ([*ORD, '--text-emb-finetune', HF,
                        '--batch-size', '128'], 'texto'),
    'bert_ft_sin':    ([*ORD, '--drop-features', 'listing_status',
                        '--text-emb-finetune', HF, '--batch-size', '128'], 'texto'),
}


# ---- 9na tanda (31/08): el transformer de INGREDIENTES como pieza (idea de Fer) ----
# ingredients quedo afuera de la v1 porque la CANTIDAD no mostro senal (EDA: corr
# 0.02 con bought; feat_extras la reintrodujo como numerica y dio delta ~ 0), pero
# la IDENTIDAD y las interacciones ingrediente x ingrediente nunca se midieron.
# Ademas es el caso de libro de "transformer como pieza": un encoder de CONJUNTO
# ([ING] + un token por ingrediente, vocabulario de TRAIN con UNK, embeddings
# aprendidos, atencion bidireccional todos-contra-todos y SIN positional encoding,
# porque la lista no tiene orden conocido) cuya salida entra a otra arquitectura.
# Hipotesis registrada ANTES de correr: hay spread por ingrediente (BTR 0.06
# Seafood ... 0.19 Baby-safe, base 0.13), pero los ingredientes co-ocurren en
# recetas fijas por categoria (Milk+Cream+Cultures n=1003 = dairy; Yeast+Wheat
# flour+Water+Sugar n=917 = bakery), asi que lo esperable es que category/allergens
# ya lo capturen y todo de delta ~ 0 vs feat_ordinal (0.824); ing_solo separa "no
# hay senal" de "la senal ya la tenian otras columnas". La seleccion sigue cerrada
# (4ta tanda): esto caracteriza, no busca campeon.
EXPERIMENTOS |= {
    # ¿cuanto predicen los ingredientes POR SI SOLOS? (referencia: tasa base 0.13)
    'ing_solo':      (['--formulation', 'ing', *PAC], 'tabular'),
    # la salida del encoder de conjunto como UN token mas de NUESTRO transformer
    # (el mecanismo de fusion, con encoder propio de ingredientes)
    'ing_fusion':    ([*ORD, '--formulation', 'ing_fusion'], 'tabular'),
    # sin encoder aparte: un token POR ingrediente en la secuencia tabular
    # (la atencion cruza ingrediente x feature directamente)
    'ing_hybrid':    ([*ORD, '--formulation', 'ing_hybrid'], 'tabular'),
    # la salida del encoder de conjunto entra a un MLP (espejo del tower de texto)
    'ing_tower':     ([*ORD, '--arch', 'ing_tower'], 'tabular'),
    # ¿profundidad del encoder? (default 1 bloque: la lista tiene <= 5 items)
    'ing_fusion_l2': ([*ORD, '--formulation', 'ing_fusion', '--ing-layer', '2'], 'tabular'),
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
