"""Entrenamiento y evaluacion de los modelos de BTR.

Protocolo de propuesta.md (seccion 7): split por query, AdamW, early stopping
por PR-AUC de validacion, promedio de corridas con seeds distintas. Cada epoca
y cada split final guardan TODAS las metricas (ver compute_metrics): las
decisiones se toman con PR-AUC (desbalance 13%) pero el resto queda registrado
para poder graficar cualquiera despues. Cada corrida escribe resultados/<nombre>.json
y, con --save-pesos, pesos/<nombre>.pt (recargable con btr.model.load_checkpoint).

Arquitecturas (--arch) y formulaciones (--formulation), ver propuesta.md 4:
  transformer + features   cada feature tabular es un token (FT-Transformer)
  transformer + text       cada caracter de title+description es un token (demo)
  transformer + hybrid     [CLS] + features + caracteres en una secuencia
  mlp                      baseline sin atencion (mismos embeddings, MLP denso)
  tower                    transformer SOLO como encoder de texto -> embedding
                           que se concatena con lo tabular y clasifica un MLP
  listwise                 los tokens son los productos de la misma pagina
  transformer + ing        SOLO los ingredientes como tokens (conjunto, 9na tanda)
  transformer + ing_fusion encoder de conjunto de ingredientes -> su [ING] entra
                           como un token mas de la secuencia tabular
  transformer + ing_hybrid un token POR ingrediente en la secuencia tabular
  ing_tower                encoder de ingredientes -> embedding que se concatena
                           con lo tabular y clasifica un MLP (espejo de tower)

Ejes transversales: --drop-features listing_status (modelo sin el estado
parseado), --strip-status (texto sin sufijo ni oracion de estado: la variante
"producto nuevo"), --numeric-mode bins, --positional, --causal, --pooling,
--pos-weight. La suite curada de experimentos esta en experimentos.py.

Disciplina: las decisiones (hiperparametros, early stopping) se toman con
VALIDACION; test se mira solo para reportar las configuraciones finales.
"""

import argparse
import glob
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, log_loss, matthews_corrcoef, precision_recall_curve, roc_auc_score,
)

from .data import (CAT_FEATURES, EXTRA_ALL, EXTRA_FEATURES, MAX_INGREDIENTS, MAX_TEXT_LEN,
                   NUM_FEATURES, TARGET, _palabras, load_dataset, prepare, prepare_listwise,
                   split_by_query, strip_status_from_text)
from .model import (ING_FORMULATIONS, BTRTransformer, IngredientTowerModel,
                    ListwiseTransformer, MLPBaseline, TextTowerModel)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_BATCH = 1024        # con secuencias largas no entra todo el split en un forward
TRAIN_EVAL_ROWS = 4000   # submuestra fija de train para las metricas por epoca


def compute_metrics(y_true, probs, loss):
    """TODAS las metricas con sentido para clasificacion binaria desbalanceada (13%).

    Se calculan siempre todas y se guardan en el JSON de la corrida, para poder
    graficar cualquiera despues sin reentrenar. Dos grupos:
      - sin umbral (las que importan para rankear): roc_auc, pr_auc, log_loss, brier
      - con umbral 0.5 (informativas): accuracy, balanced_accuracy, precision,
        recall, f1, specificity, mcc — mas f1_best y su umbral optimo, porque con
        13% de positivos el 0.5 es arbitrario.
    'loss' es el criterio de entrenamiento del modelo (BCE, con pos_weight si se
    pidio); 'log_loss' es la BCE sin pesar, comparable entre configuraciones.
    """
    pred = probs >= 0.5
    prec_c, rec_c, thr_c = precision_recall_curve(y_true, probs)
    f1_c = 2 * prec_c * rec_c / np.clip(prec_c + rec_c, 1e-12, None)
    i_best = int(np.nanargmax(f1_c))
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # mcc/f1 avisan si pred es todo-negativos
        return {
            'loss': float(loss),
            'roc_auc': float(roc_auc_score(y_true, probs)),
            'pr_auc': float(average_precision_score(y_true, probs)),
            'log_loss': float(log_loss(y_true, probs, labels=[0, 1])),
            'brier': float(brier_score_loss(y_true, probs)),
            'accuracy': float(accuracy_score(y_true, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
            'precision': float(tp / (tp + fp)) if tp + fp else 0.0,
            'recall': float(tp / (tp + fn)) if tp + fn else 0.0,
            'f1': float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            'specificity': float(tn / (tn + fp)) if tn + fp else 0.0,
            'mcc': float(matthews_corrcoef(y_true, pred)),
            'f1_best': float(f1_c[i_best]),
            'thr_f1_best': float(thr_c[i_best]) if i_best < len(thr_c) else 1.0,
            'tasa_pred_pos': float(pred.mean()),
            'tasa_real_pos': float(y_true.mean()),
        }


def evaluate(model, split, max_rows=None):
    """Dict con TODAS las metricas del split (compute_metrics), en eval mode y por lotes.

    Los splits son 4-tuplas posicionales que se pasan tal cual al modelo:
      filas:    (x_cat, x_num, x_text, y)     -> model(x_cat, x_num, x_text, y)
      listwise: (x_cat, x_num, prod_mask, y)  -> model(x_cat, x_num, prod_mask, y)
    En listwise (tensores 3D) las metricas se calculan solo sobre los slots reales.
    """
    a, b, c, y = split
    if max_rows is not None and a.shape[0] > max_rows:
        # submuestra fija (generador con seed propia) para abaratar la metrica de train
        idx = torch.randperm(a.shape[0], generator=torch.Generator().manual_seed(0))[:max_rows]
        idx = idx.to(a.device)
        a, b, c, y = a[idx], b[idx], c[idx], y[idx]
    model.eval()
    logits = []
    with torch.no_grad():
        for s in range(0, a.shape[0], EVAL_BATCH):
            lg, _ = model(a[s:s + EVAL_BATCH], b[s:s + EVAL_BATCH], c[s:s + EVAL_BATCH])
            logits.append(lg)
        logits = torch.cat(logits)
        if a.dim() == 3:  # listwise: c es la mascara (o x_text, de la que se deriva)
            mask = c if c.dtype == torch.bool else (c != 0).any(-1)
            logits, y = logits[mask], y[mask]
        elif y.dim() == 2:  # multi-task (--cart-aux): metricas SIEMPRE sobre bought
            y = y[:, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=model.pos_weight
        ).item()
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = y.cpu().numpy()
    return compute_metrics(y_true, probs, loss)


def train_model(model, splits, epochs=60, batch_size=256, lr=1e-3, patience=8, verbose=True,
                weight_decay=1e-2, train_targets=None, l2sp=0.0, l2sp_ref=None,
                text_emb_lr=1e-5):
    """Entrena con early stopping por PR-AUC de validacion; restaura el mejor estado.

    weight_decay: el de AdamW (1e-2 era el default implicito de TODAS las corridas
      previas; recien en la 6ta tanda se barre).
    train_targets: reemplazo del target SOLO para los pasos de gradiente (labels
      suavizadas o probabilidades del teacher en distillation); las metricas de
      train/val/test se calculan siempre contra las labels duras originales.
    l2sp / l2sp_ref: penalidad L2 hacia los pesos de referencia (post pre-training),
      el analogo simple de la KL penalty de la clase 3: ajustarse a la tarea sin
      alejarse del modelo pre-entrenado (L2-SP, Li et al. 2018).
    """
    a, b, c, y = splits['train']
    entrenables = [p for p in model.parameters() if p.requires_grad]
    if getattr(model, 'hf_encoder', None) is not None:
        # fine-tuning del preentrenado (clase 3): lr chico y separado para el
        # encoder HF (no destruir lo aprendido), lr normal para el resto
        hf_ids = {id(p) for p in model.hf_encoder.parameters()}
        grupos = [{'params': [p for p in entrenables if id(p) not in hf_ids]},
                  {'params': [p for p in entrenables if id(p) in hf_ids],
                   'lr': text_emb_lr}]
        optimizer = torch.optim.AdamW(grupos, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(entrenables, lr=lr, weight_decay=weight_decay)
    history, best_pr, best_state, since_best = [], -1.0, None, 0
    y_grad = y if train_targets is None else train_targets

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(a.shape[0], device=a.device)
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            _, loss = model(a[idx], b[idx], c[idx], y_grad[idx])
            if l2sp > 0 and l2sp_ref is not None:
                ancla = sum(((p - l2sp_ref[n]) ** 2).sum()
                            for n, p in model.named_parameters()
                            if p.requires_grad and n in l2sp_ref)
                loss = loss + l2sp * ancla
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        tr = evaluate(model, splits['train'], max_rows=TRAIN_EVAL_ROWS)
        va = evaluate(model, splits['val'])
        # historial con TODAS las metricas por epoca, para graficar cualquiera despues
        history.append({'epoch': epoch, 'train': tr, 'val': va})
        val_pr = va['pr_auc']
        if verbose:
            print(f"epoch {epoch:3d} | loss train {tr['loss']:.4f} val {va['loss']:.4f} | "
                  f"PR-AUC train {tr['pr_auc']:.4f} val {val_pr:.4f} | ROC-AUC val {va['roc_auc']:.4f}")

        if val_pr > best_pr:
            best_pr, since_best = val_pr, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                if verbose:
                    print(f"early stopping en epoch {epoch} (mejor val PR-AUC: {best_pr:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def pretrain_w2v(train_df, prep, d_model, device, epochs=3, window=2, k_neg=5, verbose=True):
    """Skipgram con negative sampling sobre el corpus de TRAIN (--w2v-init).

    La conexion clase 1 -> clase 2 de la revision externa: pre-entrenar los
    embeddings de palabras de forma no supervisada (predecir el contexto) y
    usarlos como INICIALIZACION del encoder de texto, que luego se ajusta
    end-to-end. Comparar contra la inicializacion aleatoria mide cuanto vale el
    pre-entrenamiento en un corpus tan chico (10k documentos).
    """
    import torch.nn as nn
    from .data import strip_status_from_text
    docs = []
    for t, d in zip(train_df['title'], train_df['description']):
        if prep.strip_status:
            t, d = strip_status_from_text(t, d)
        docs.append([prep.char_vocab.get(w, 1) for w in _palabras(t + ' ' + d)])
    centros, contextos = [], []
    for doc in docs:
        for i, c in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if j != i:
                    centros.append(c)
                    contextos.append(doc[j])
    centros = torch.tensor(centros, device=device)
    contextos = torch.tensor(contextos, device=device)
    vocab = prep.char_vocab_size
    emb_c = nn.Embedding(vocab, d_model).to(device)
    emb_o = nn.Embedding(vocab, d_model).to(device)
    opt = torch.optim.Adam(list(emb_c.parameters()) + list(emb_o.parameters()), lr=5e-3)
    lote = 8192
    for ep in range(epochs):
        perm = torch.randperm(len(centros), device=device)
        total = 0.0
        for s in range(0, len(perm), lote):
            idx = perm[s:s + lote]
            c, o = emb_c(centros[idx]), emb_o(contextos[idx])
            neg = emb_o(torch.randint(2, vocab, (len(idx), k_neg), device=device))
            pos = torch.nn.functional.logsigmoid((c * o).sum(-1))
            negs = torch.nn.functional.logsigmoid(-(neg @ c.unsqueeze(-1)).squeeze(-1)).sum(-1)
            loss = -(pos + negs).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose:
            print(f"  w2v epoch {ep}: loss {total / len(centros):.4f} "
                  f"({len(centros):,} pares, vocab {vocab})")
    return emb_c.weight.detach()


def pretrain_mlm(model, split_train, cardinalities, epochs, batch_size, lr, device,
                 verbose=True):
    """Pre-entrenamiento "MLM sobre features" (revision externa, opcional).

    Analogia con el MLM de BERT llevada a lo tabular: en cada fila se enmascara
    UNA columna al azar (su token se reemplaza por un vector [MASK] aprendido) y
    el modelo debe predecirla mirando las demas — clasificacion para categoricas
    (nivel original), regresion para numericas (valor z-scoreado). Se entrena el
    tronco (tokenizer + bloques) con cabezas temporarias que luego se descartan;
    despues arranca el entrenamiento supervisado normal con ese tronco ya
    "conocedor" de las correlaciones entre features.
    """
    import torch.nn as nn
    x_cat, x_num, _, _ = split_train
    tok = model.tokenizer
    n_tok, n_cat = tok.n_tokens, tok.n_cat
    d = model.cls.shape[-1]
    mask_vec = nn.Parameter(torch.empty(1, d, device=device))
    nn.init.normal_(mask_vec, mean=0.0, std=0.02)
    heads = nn.ModuleList(
        [nn.Linear(d, c) for c in cardinalities]
        + [nn.Linear(d, 1) for _ in range(n_tok - n_cat)]
    ).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + [mask_vec] + list(heads.parameters()),
                            lr=lr)
    n = x_cat.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total, cuenta = 0.0, 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xc, xn = x_cat[idx], x_num[idx]
            b = xc.shape[0]
            tokens = tok(xc, xn)                                   # (b, T, d)
            pos = torch.randint(0, n_tok, (b,), device=device)
            tokens[torch.arange(b, device=device), pos] = mask_vec
            x = torch.cat([model.cls.expand(b, -1, -1), tokens], dim=1)
            if model.position_embedding_table is not None:
                x = x + model.position_embedding_table(torch.arange(x.shape[1], device=device))
            for blk in model.blocks:
                x = blk(x)
            h = model.ln_f(x)[torch.arange(b, device=device), pos + 1]  # +1 por el CLS
            loss = x.new_zeros(())
            for f in range(n_tok):
                filas = pos == f
                if not filas.any():
                    continue
                if f < n_cat:
                    loss = loss + torch.nn.functional.cross_entropy(
                        heads[f](h[filas]), xc[filas, f])
                else:
                    loss = loss + torch.nn.functional.mse_loss(
                        heads[f](h[filas]).squeeze(-1), xn[filas, f - n_cat])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * b
            cuenta += b
        if verbose:
            print(f"  mlm epoch {ep}: loss {total / cuenta:.4f}")
    # las cabezas y el [MASK] se descartan; queda el tronco pre-entrenado


def pretrain_ae(model, split_train, cardinalities, epochs, batch_size, lr, device,
                verbose=True):
    """Pre-entrenamiento autoencoder con cuello de botella en el CLS (--pretrain-ae).

    Representation learning de la clase 3 via el autoencoder de SIA (TP5): el
    vector pooled del CLS (d_model) debe RECONSTRUIR todas las features de la
    fila — clasificacion para categoricas, regresion para numericas. Se difiere
    del MLM en el cuello: MLM predice UNA feature enmascarada mirando las demas
    (las posiciones conservan su informacion); aca TODO pasa por un unico vector
    de d_model, como el espacio latente del AE. Las cabezas se descartan.
    """
    import torch.nn as nn
    x_cat, x_num, _, _ = split_train
    tok = model.tokenizer
    n_cat, n_num = tok.n_cat, tok.n_tokens - tok.n_cat
    d = model.cls.shape[-1]
    heads = nn.ModuleList(
        [nn.Linear(d, c) for c in cardinalities]
        + [nn.Linear(d, 1) for _ in range(n_num)]
    ).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()), lr=lr)
    n = x_cat.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total, cuenta = 0.0, 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xc, xn = x_cat[idx], x_num[idx]
            h = model._pooled(xc, xn)  # (b, d): el cuello de botella
            loss = h.new_zeros(())
            for f in range(n_cat):
                loss = loss + torch.nn.functional.cross_entropy(heads[f](h), xc[:, f])
            for f in range(n_num):
                loss = loss + torch.nn.functional.mse_loss(
                    heads[n_cat + f](h).squeeze(-1), xn[:, f])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * xc.shape[0]
            cuenta += xc.shape[0]
        if verbose:
            print(f"  ae epoch {ep}: loss {total / cuenta:.4f}")


def entrenar_som(x, grid, seed, epochs=10):
    """Mapa de Kohonen (SIA, TP4) minimo: grid x grid sobre las numericas de TRAIN.

    Online clasico: BMU + vecindad gaussiana con lr y sigma decrecientes.
    Devuelve los pesos (grid^2, d); la celda BMU de cada fila entra al modelo
    como UNA feature categorica mas (--som-feature): ¿agrega senal un
    clustering topologico no supervisado que el transformer no saque solo?
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n, d = x.shape
    w = rng.normal(0.0, 0.1, (grid * grid, d))
    coords = np.array([(i // grid, i % grid) for i in range(grid * grid)], dtype=float)
    t, t_max = 0, epochs * n
    for _ in range(epochs):
        for i in rng.permutation(n):
            frac = t / t_max
            lr = 0.5 * (1.0 - frac) + 0.01 * frac
            sigma = max(grid / 2.0 * (1.0 - frac), 0.5)
            v = x[i]
            bmu = int(((w - v) ** 2).sum(1).argmin())
            h = np.exp(-((coords - coords[bmu]) ** 2).sum(1) / (2 * sigma ** 2))
            w += lr * h[:, None] * (v - w)
            t += 1
    return w


def asignar_som(x, w):
    """Celda BMU (0..grid^2-1) de cada fila, por lotes para no armar (n, celdas, d)."""
    import numpy as np
    out = []
    for s in range(0, x.shape[0], 2048):
        bloque = x[s:s + 2048]
        out.append(((bloque[:, None, :] - w[None]) ** 2).sum(-1).argmin(1))
    return np.concatenate(out)


def expandir_ckpts(spec, seed):
    """Resuelve la spec de checkpoints: patrones separados por coma, {seed} y glob.

    Ej: 'pesos/feat_ordinal_*_seed{seed}.pt,pesos/robu_init4*_seed{seed}.pt'
    Aborta si algun patron no matchea nada (el teacher TIENE que existir).
    """
    rutas = []
    for patron in spec.split(','):
        patron = patron.strip().format(seed=seed)
        if not patron:
            continue
        hallados = sorted((REPO_ROOT / p if not Path(p).is_absolute() else Path(p))
                          for p in glob.glob(str(REPO_ROOT / patron)))
        if not hallados:
            raise SystemExit(f'--init-from/--distill-from/--embed-from: ningun checkpoint '
                             f'matchea {patron!r} (¿falta correr al teacher primero?)')
        rutas.extend(hallados)
    vistos, unicos = set(), []
    for r in rutas:
        if r not in vistos:
            vistos.add(r)
            unicos.append(r)
    return unicos


def probs_teachers(rutas, x_cat, x_num, device, verbose=True):
    """Promedio de sigmoides de los teachers sobre las filas dadas (distillation).

    Clase 3: entrenar contra las PROBABILIDADES del modelo grande (soft labels)
    informa mas que la label dura; con n>1 teachers es el caso "integrando
    informacion de varios modelos" que menciono la profesora. Los teachers se
    cargan del mismo split (misma seed) — sin fuga: vieron exactamente el mismo
    train que va a ver el student.
    """
    from .model import load_checkpoint
    acum = None
    for ruta in rutas:
        teacher, _ = load_checkpoint(ruta, device=device)
        p = teacher.predict_proba(x_cat, x_num)
        acum = p if acum is None else acum + p
        del teacher
    probs = acum / len(rutas)
    if verbose:
        print(f"  teachers: {len(rutas)} ckpt(s), prob media {probs.mean():.4f}")
    return probs


def matriz_cruda(x_cat, x_num, cardinalities):
    """[one-hot de categoricas | numericas]: la entrada 'cruda' para PCA/AE (SIA)."""
    import torch.nn.functional as F
    partes = [F.one_hot(x_cat[:, i], card).float() for i, card in enumerate(cardinalities)]
    partes.append(x_num)
    return torch.cat(partes, dim=1)


def ajustar_pca(m_train, k):
    """PCA clasico (SVD sobre train centrado; la version cerrada de Oja/Sanger en SIA).

    Devuelve (media, componentes (D,k), std de las proyecciones de train) para
    proyectar cualquier split y blanquear con estadisticos de TRAIN.
    """
    media = m_train.mean(0, keepdim=True)
    _, _, vt = torch.linalg.svd(m_train - media, full_matrices=False)
    comp = vt[:k].T
    proy = (m_train - media) @ comp
    std = proy.std(0, keepdim=True).clamp_min(1e-6)
    return media, comp, std


def ajustar_ae(m_train, k, device, epochs=30, batch_size=256, lr=1e-3, seed=0, verbose=True):
    """Autoencoder denso (SIA, TP5) sobre la matriz cruda; devuelve el ENCODER.

    in -> 4k -> k -> 4k -> in con MSE. El espacio latente (k) reemplaza a las
    features del MLP (--ae-latent): feature extraction puro de la clase 3 —
    la representacion se aprende SIN mirar el target.
    """
    import torch.nn as nn
    torch.manual_seed(seed)
    d_in = m_train.shape[1]
    enc = nn.Sequential(nn.Linear(d_in, 4 * k), nn.ReLU(), nn.Linear(4 * k, k)).to(device)
    dec = nn.Sequential(nn.Linear(k, 4 * k), nn.ReLU(), nn.Linear(4 * k, d_in)).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    n = m_train.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=m_train.device)
        total = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            rec = dec(enc(m_train[idx]))
            loss = torch.nn.functional.mse_loss(rec, m_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * idx.shape[0]
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  ae-latente epoch {ep}: mse {total / n:.4f}")
    enc.eval()
    return enc


def resolve_device(arg):
    if arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return arg


def run_name(args, seed):
    base = args.formulation if args.arch == 'transformer' else args.arch
    if args.arch == 'listwise' and args.listwise_texto:
        base = 'listwisetexto'
    parts = [base, f"d{args.d_model}", f"h{args.n_head}", f"l{args.n_layer}", args.numeric_mode]
    if args.text_tokens != 'chars':
        parts.append('words' + ('w2v' if args.w2v_init else ''))
    if args.cat_encoding != 'embedding':
        parts.append(f"cat{args.cat_encoding}"
                     + (str(args.hash_buckets) if args.cat_encoding == 'hashing' else ''))
    if args.train_frac < 1.0:
        parts.append(f"frac{args.train_frac:g}")
    if args.init_seed is not None:
        parts.append(f"init{args.init_seed}")
    if args.pretrain_mlm:
        parts.append(f"mlm{args.pretrain_mlm}")
    if args.cv_k:
        parts.append(f"cv{args.cv_k}f{args.cv_fold}")
    if args.per_feature != 'none':
        parts.append(f"pf{args.per_feature}")
    if args.dropout != 0.1:
        parts.append(f"do{args.dropout:g}")
    if args.weight_decay != 1e-2:
        parts.append(f"wd{args.weight_decay:g}")
    if args.feature_dropout:
        parts.append(f"fdrop{args.feature_dropout:g}")
    if args.label_smoothing:
        parts.append(f"ls{args.label_smoothing:g}")
    if args.sin_residual:
        parts.append('sinres')
    if args.sin_layernorm:
        parts.append('sinln')
    if args.init_from:
        parts.append('ft')
    if args.freeze_backbone:
        parts.append('frz')
    if args.reinit_head:
        parts.append('rih')
    if args.l2sp:
        parts.append(f"l2sp{args.l2sp:g}")
    if args.distill_from:
        parts.append(f"dst{args.distill_alpha:g}")
    if args.embed_from:
        parts.append('embfrom')
    if args.som_feature:
        parts.append(f"som{args.som_feature}")
    if args.pretrain_ae:
        parts.append(f"ae{args.pretrain_ae}")
    if args.ae_latent:
        parts.append(f"ael{args.ae_latent}")
    if args.pca:
        parts.append(f"pca{args.pca}")
    if args.text_emb:
        parts.append('temb-' + Path(args.text_emb).stem.replace('_', ''))
    if args.text_emb_finetune:
        parts.append('tembft' + (f"-lr{args.text_emb_lr:g}"
                                 if args.text_emb_lr != 1e-5 else ''))
    if args.cat_feature_encoding:
        pares = sorted(p.strip().replace('_', '').replace('=', '-')
                       for p in args.cat_feature_encoding.split(',') if p.strip())
        parts.append('cfe-' + '+'.join(pares))
    if args.strip_status:
        parts.append('stripstatus')
    if args.drop_features:
        parts.append('sin-' + args.drop_features.replace(',', '-').replace('_', ''))
    if args.extra_features:
        parts.append('extra-' + args.extra_features.replace(',', '-').replace('_', ''))
    if args.pooling != 'cls':
        parts.append(args.pooling)
    if args.cls_position != 'first':
        parts.append('clslast')
    if args.cart_aux:
        parts.append(f"cartaux{args.cart_aux:g}")
    if args.ing_layer != 1:
        parts.append(f"il{args.ing_layer}")
    for flag in ('positional', 'causal', 'pos_weight'):
        if getattr(args, flag):
            parts.append(flag.replace('_', ''))
    parts.append(f"seed{seed}")
    name = '_'.join(parts)
    return f"{args.tag}_{name}" if args.tag else name


def unique_path(path):
    """Evita pisar corridas anteriores: agrega _2, _3, ... si el nombre existe."""
    if not path.exists():
        return path
    for i in range(2, 1000):
        candidate = path.with_stem(f"{path.stem}_{i}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f'demasiadas corridas con el nombre {path.stem}')


def drop_feature_columns(splits, drop, listwise=False, cat_features=None, num_features=None):
    """Saca features por nombre (p. ej. listing_status) recortando columnas de los tensores."""
    cat_features = cat_features or CAT_FEATURES
    num_features = num_features or NUM_FEATURES
    keep_cat = [i for i, f in enumerate(cat_features) if f not in drop]
    keep_num = [i for i, f in enumerate(num_features) if f not in drop]
    unknown = drop - set(cat_features) - set(num_features)
    if unknown:
        raise SystemExit(f'--drop-features desconocidos: {sorted(unknown)}')
    dim = 2 if listwise else 1  # en listwise los features son la 3ra dimension
    splits = {k: (v[0].index_select(dim, torch.tensor(keep_cat, dtype=torch.long)),
                  v[1].index_select(dim, torch.tensor(keep_num, dtype=torch.long)),
                  v[2], v[3]) for k, v in splits.items()}
    return splits, keep_cat, keep_num


def parse_cat_feature_encoding(args, cats):
    """'listing_status=ordinal,brand=hashing' -> dict validado {feature: modo}."""
    pares = {}
    for par in (p.strip() for p in args.cat_feature_encoding.split(',') if p.strip()):
        if '=' not in par:
            raise SystemExit(f'--cat-feature-encoding espera feature=modo, no: {par!r}')
        f, m = (s.strip() for s in par.split('=', 1))
        if f not in cats:
            raise SystemExit(f'--cat-feature-encoding: feature desconocida o excluida: {f!r} '
                             f'(disponibles: {cats})')
        if m not in ('embedding', 'target', 'freq', 'ordinal', 'hashing'):
            raise SystemExit(f'--cat-feature-encoding: modo invalido {m!r} '
                             '(embedding/target/freq/ordinal/hashing; onehot es solo global MLP)')
        pares[f] = m
    return pares


def build_cat_tables(modes, prep, train_df, keep_cat, hash_buckets):
    """Lookups por feature categorica para los modos que los requieren.

    Ajustados SOLO con train, alineados a los indices de prep.vocabs (0 = UNK):
      target:  nivel -> media suavizada de bought: (sum + m*global) / (n + m), m=50
               (el suavizado amortigua niveles chicos y la auto-inclusion del target)
      ordinal: nivel -> rango del nivel al ordenar por esa media suavizada,
               normalizado a [0,1] (UNK -> 0.5). Conserva el ORDEN aprendible de
               los datos y descarta las magnitudes; un orden semantico "a mano"
               seria indefendible (EDA: el wording no predice el tier).
      freq:    nivel -> frecuencia relativa del nivel en train
      hashing: nivel -> md5(feature|valor) % B  (el "modulo" clasico del hashing trick)

    Devuelve dict {posicion en la lista kept: tensor} o None si ninguna lo necesita.
    """
    import hashlib
    m, global_mean = 50.0, float(train_df[TARGET].mean())
    tablas = {}
    for pos, i in enumerate(keep_cat):
        modo = modes[pos]
        if modo == 'embedding':
            continue
        f = prep.cats[i]
        vocab = prep.vocabs[f]
        if modo == 'hashing':
            t = torch.zeros(len(vocab) + 1, dtype=torch.long)
            for valor, idx in vocab.items():
                h = int(hashlib.md5(f'{f}|{valor}'.encode()).hexdigest(), 16)
                t[idx] = h % hash_buckets
        elif modo == 'freq':
            t = torch.zeros(len(vocab) + 1)
            for valor, idx in vocab.items():
                t[idx] = float((train_df[f] == valor).sum()) / len(train_df)
        else:  # target u ordinal: ambos parten de la media suavizada por nivel
            grp = train_df.groupby(f)[TARGET].agg(['sum', 'count'])
            suav = {}
            for valor, idx in vocab.items():
                n = float(grp.loc[valor, 'count'])
                suav[idx] = (float(grp.loc[valor, 'sum']) + m * global_mean) / (n + m)
            t = torch.full((len(vocab) + 1,), global_mean if modo == 'target' else 0.5)
            if modo == 'target':
                for idx, v in suav.items():
                    t[idx] = v
            else:  # ordinal
                orden = sorted(suav, key=suav.get)  # indices ordenados por BTR suavizado
                k = max(len(orden) - 1, 1)
                for rango, idx in enumerate(orden):
                    t[idx] = rango / k
        tablas[pos] = t
    return tablas or None


def build_model(args, prep, cardinalities, n_numeric, bin_edges, pos_weight,
                max_products=None, cat_tables=None, cat_modes=None, text_emb_dim=0):
    """Configura y construye la arquitectura pedida; devuelve (modelo, config para el ckpt)."""
    common = dict(d_model=args.d_model, dropout=args.dropout,
                  numeric_mode=args.numeric_mode, pos_weight=pos_weight)
    encod = dict(cat_encoding=args.cat_encoding, hash_buckets=args.hash_buckets,
                 cat_modes=cat_modes)
    if args.arch == 'transformer':
        config = dict(formulation=args.formulation, cat_cardinalities=cardinalities,
                      n_numeric=n_numeric, char_vocab_size=prep.char_vocab_size,
                      max_text_len=prep.max_text_len, n_head=args.n_head,
                      n_layer=args.n_layer, causal=args.causal, pooling=args.pooling,
                      use_positional=args.positional, cls_position=args.cls_position,
                      cart_lambda=args.cart_aux, per_feature=args.per_feature,
                      sin_residual=args.sin_residual, sin_layernorm=args.sin_layernorm,
                      feature_dropout=args.feature_dropout, text_emb_dim=text_emb_dim,
                      hf_model=args.text_emb_finetune,
                      ing_vocab_size=(prep.ing_vocab_size
                                      if args.formulation in ING_FORMULATIONS else None),
                      max_ingredients=(MAX_INGREDIENTS
                                       if args.formulation in ING_FORMULATIONS else 0),
                      ing_layer=args.ing_layer,
                      **encod, **common)
        model = BTRTransformer(**config, bin_edges=bin_edges, cat_tables=cat_tables)
    elif args.arch == 'mlp':
        config = dict(cat_cardinalities=cardinalities, n_numeric=n_numeric,
                      cart_lambda=args.cart_aux, **encod, **common)
        model = MLPBaseline(**config, bin_edges=bin_edges, cat_tables=cat_tables)
    elif args.arch == 'tower':
        config = dict(cat_cardinalities=cardinalities, n_numeric=n_numeric,
                      char_vocab_size=prep.char_vocab_size, max_text_len=prep.max_text_len,
                      n_head=args.n_head, n_layer=args.n_layer, causal=args.causal,
                      cart_lambda=args.cart_aux, **encod, **common)
        model = TextTowerModel(**config, bin_edges=bin_edges, cat_tables=cat_tables)
    elif args.arch == 'listwise':
        config = dict(cat_cardinalities=cardinalities, n_numeric=n_numeric,
                      max_products=max_products, n_head=args.n_head,
                      n_layer=args.n_layer, use_text=args.listwise_texto, **common)
        if args.listwise_texto:
            config.update(char_vocab_size=prep.char_vocab_size, max_text_len=prep.max_text_len)
        model = ListwiseTransformer(**config, bin_edges=bin_edges)
    elif args.arch == 'ing_tower':
        config = dict(cat_cardinalities=cardinalities, n_numeric=n_numeric,
                      ing_vocab_size=prep.ing_vocab_size, max_ingredients=MAX_INGREDIENTS,
                      n_head=args.n_head, ing_layer=args.ing_layer,
                      cart_lambda=args.cart_aux, **encod, **common)
        model = IngredientTowerModel(**config, bin_edges=bin_edges, cat_tables=cat_tables)
    else:
        raise SystemExit(f'arquitectura desconocida: {args.arch}')
    return model, config


def run(csv_path, seed, args, device):
    """Una corrida completa: prepara datos, entrena, guarda resultados/pesos."""
    torch.manual_seed(seed if args.init_seed is None else args.init_seed)
    listwise = args.arch == 'listwise'
    usa_ing = args.formulation in ING_FORMULATIONS or args.arch == 'ing_tower'
    if args.pretrain_mlm and (args.arch != 'transformer' or args.formulation != 'features'
                              or args.cls_position != 'first'):
        raise SystemExit('--pretrain-mlm es solo para transformer features con CLS al inicio')
    if args.per_feature != 'none' and (args.arch != 'transformer'
                                       or args.formulation != 'features' or args.causal):
        raise SystemExit('--per-feature es solo para transformer formulation features '
                         '(la posicion tiene que SER el feature) y sin mascara causal')
    if (args.cv_k or args.train_frac < 1.0) and listwise:
        raise SystemExit('cv / train-frac no implementados para listwise')
    if args.cat_encoding == 'onehot' and args.arch != 'mlp':
        raise SystemExit('cat-encoding onehot es solo para --arch mlp: para el transformer, '
                         'one-hot + proyeccion lineal aprende la misma matriz que el '
                         'embedding (propuesta 6.1) — no hay experimento que correr')
    if args.w2v_init and args.text_tokens != 'words':
        raise SystemExit('--w2v-init requiere --text-tokens words')
    if args.text_tokens == 'words' and listwise:
        raise SystemExit('text-tokens words no implementado para listwise')
    if args.cat_encoding != 'embedding' and listwise:
        raise SystemExit('cat-encoding alternativo no implementado para listwise')
    if args.cat_feature_encoding and listwise:
        raise SystemExit('cat-feature-encoding no implementado para listwise')
    if args.cat_feature_encoding and args.cat_encoding == 'onehot':
        raise SystemExit('cat-feature-encoding no se combina con onehot (que es global del MLP)')
    # ---- guards de la 6ta tanda (regularizacion / transfer / SIA) ----
    if args.feature_dropout and (args.arch != 'transformer' or args.formulation != 'features'):
        raise SystemExit('--feature-dropout es solo para transformer formulation features '
                         '(anula tokens de features; en texto seria otro experimento)')
    if (args.sin_residual or args.sin_layernorm) and args.arch != 'transformer':
        raise SystemExit('--sin-residual/--sin-layernorm son ablaciones de los bloques '
                         'del transformer')
    if args.label_smoothing and (listwise or args.cart_aux):
        raise SystemExit('--label-smoothing no esta implementado para listwise ni multi-task')
    if args.distill_from and (listwise or args.cart_aux or args.label_smoothing):
        raise SystemExit('--distill-from no se combina con listwise, --cart-aux ni '
                         '--label-smoothing (un solo target de entrenamiento por corrida)')
    if (args.init_from or args.freeze_backbone) and args.arch != 'transformer':
        raise SystemExit('--init-from/--freeze-backbone estan implementados solo para '
                         'el transformer')
    if args.reinit_head and not args.init_from:
        raise SystemExit('--reinit-head solo tiene sentido con --init-from (sin carga previa '
                         'la cabeza ya arranca aleatoria)')
    if args.l2sp and not (args.init_from or args.pretrain_mlm or args.pretrain_ae):
        raise SystemExit('--l2sp ancla a pesos PRE-ENTRENADOS: requiere --init-from, '
                         '--pretrain-mlm o --pretrain-ae')
    if args.embed_from and args.arch != 'mlp':
        raise SystemExit('--embed-from es feature extraction PARA el MLP (--arch mlp)')
    if args.embed_from and args.som_feature:
        raise SystemExit('--embed-from no se combina con --som-feature (el extractor espera '
                         'las columnas originales)')
    if (args.ae_latent or args.pca) and args.arch != 'mlp':
        raise SystemExit('--ae-latent/--pca reemplazan la entrada del MLP (--arch mlp)')
    if args.ae_latent and args.pca:
        raise SystemExit('--ae-latent y --pca son excluyentes (una compresion por corrida)')
    if (args.ae_latent or args.pca) and (args.cat_encoding != 'embedding'
                                         or args.numeric_mode != 'linear'
                                         or args.cat_feature_encoding or args.som_feature
                                         or args.embed_from or args.cart_aux):
        raise SystemExit('--ae-latent/--pca: la entrada pasa a ser [one-hot|numericas] '
                         'comprimida; no se combina con otros encodings/extras')
    if args.som_feature and (listwise or args.arch == 'tower'
                             or (args.arch == 'transformer' and args.formulation in ('text', 'ing'))
                             or args.cat_encoding == 'onehot'):
        raise SystemExit('--som-feature agrega una categorica a la rama tabular '
                         '(features/hybrid/fusion o mlp sin onehot)')
    if args.pretrain_ae and (args.arch != 'transformer' or args.formulation != 'features'
                             or args.cls_position != 'first'):
        raise SystemExit('--pretrain-ae es solo para transformer features con CLS al inicio')
    if args.pretrain_ae and args.pretrain_mlm:
        raise SystemExit('--pretrain-ae y --pretrain-mlm: elegir UN pre-entrenamiento')
    # ---- guards de la 8va tanda (transfer desde un preentrenado externo) ----
    if args.text_emb and args.text_emb_finetune:
        raise SystemExit('--text-emb y --text-emb-finetune: elegir UN regimen '
                         '(feature extraction congelado o fine-tuning)')
    if args.text_emb or args.text_emb_finetune:
        if args.cv_k or args.train_frac < 1.0 or listwise:
            raise SystemExit('preentrenado externo: no implementado con cv/train-frac/'
                             'listwise (los embeddings se alinean al split holdout)')
        if args.pca or args.ae_latent or args.som_feature or args.embed_from:
            raise SystemExit('preentrenado externo: no se combina con pca/ae-latent/'
                             'som/embed-from')
        if args.pretrain_mlm or args.pretrain_ae or args.init_from or args.per_feature != 'none':
            raise SystemExit('preentrenado externo: no se combina con mlm/ae/init-from/'
                             'per-feature')
    if args.text_emb and not (args.arch == 'mlp' or (args.arch == 'transformer'
                                                     and args.formulation == 'features')):
        raise SystemExit('--text-emb: transformer formulation features (un token extra) '
                         'o mlp (numericas extra)')
    if args.text_emb_finetune and not (args.arch == 'transformer'
                                       and args.formulation == 'features'):
        raise SystemExit('--text-emb-finetune: solo transformer formulation features')
    # ---- guards de la 9na tanda (ingredientes como conjunto) ----
    if args.formulation in ING_FORMULATIONS and args.arch != 'transformer':
        raise SystemExit('las formulaciones ing_* son del transformer; para encoder de '
                         'ingredientes + MLP usar --arch ing_tower')
    if usa_ing and (args.text_emb or args.text_emb_finetune or args.w2v_init):
        raise SystemExit('ingredientes: la lista ocupa el slot del texto; no se combina '
                         'con --text-emb/--text-emb-finetune/--w2v-init')
    extras = tuple(f.strip() for f in args.extra_features.split(',') if f.strip())
    if extras == ('all',):
        extras = EXTRA_ALL  # congelado: ver data.EXTRA_ALL
    if listwise:
        if args.cart_aux:
            raise SystemExit('--cart-aux no implementado para listwise')
        if extras:
            raise SystemExit('--extra-features no implementado para listwise')
        prep, max_products, splits = prepare_listwise(
            csv_path, seed=seed, with_text=args.listwise_texto,
            max_text_len=args.max_text_len, strip_status=args.strip_status)
        train_df = None
    else:
        if args.listwise_texto:
            raise SystemExit('--listwise-texto requiere --arch listwise')
        prep, train_df, splits = prepare(csv_path, seed=seed, max_text_len=args.max_text_len,
                                         strip_status=args.strip_status,
                                         extra_features=extras, include_cart=args.cart_aux > 0,
                                         text_tokens=args.text_tokens,
                                         train_frac=args.train_frac,
                                         cv_k=args.cv_k, cv_fold=args.cv_fold,
                                         use_ingredients=usa_ing)
        max_products = None

    drop = {f.strip() for f in args.drop_features.split(',') if f.strip()}
    if drop == {'all'}:
        # sin features tabulares: solo tiene sentido si otra cosa alimenta al
        # modelo (p. ej. --text-emb: ¿cuanto ve el preentrenado por si solo?)
        if not (args.text_emb or args.text_emb_finetune):
            raise SystemExit('--drop-features all requiere --text-emb/--text-emb-finetune')
        drop = set(prep.cats) | set(prep.nums)
    splits, keep_cat, keep_num = drop_feature_columns(splits, drop, listwise,
                                                      prep.cats, prep.nums)
    splits = {k: tuple(t.to(device) for t in v) for k, v in splits.items()}

    bin_edges = None
    if args.numeric_mode == 'bins':
        if listwise:
            raise SystemExit('numeric-mode bins no implementado para listwise')
        bin_edges = prep.bin_edges(train_df, args.n_bins)[keep_num]
    pos_weight = None
    if args.pos_weight:
        _, _, m_or_t, y_train = splits['train']
        if listwise:
            mask = m_or_t if m_or_t.dtype == torch.bool else (m_or_t != 0).any(-1)
            y_flat = y_train[mask]
        else:
            y_flat = y_train[:, 0] if y_train.dim() == 2 else y_train
        pos_weight = ((1 - y_flat.mean()) / y_flat.mean()).item()  # negativos/positivos

    cardinalities = [c for i, c in enumerate(prep.cat_cardinalities) if i in keep_cat]
    if listwise:
        cat_tables, cat_modes = None, None
    else:
        por_feature = parse_cat_feature_encoding(args, [prep.cats[i] for i in keep_cat])
        cat_modes = [por_feature.get(prep.cats[i], args.cat_encoding) for i in keep_cat]
        if args.cat_encoding == 'onehot':
            cat_modes = None  # onehot es global del MLP, no entra al tokenizer
            cat_tables = None
        else:
            cat_tables = build_cat_tables(cat_modes, prep, train_df, keep_cat,
                                          args.hash_buckets)
            if all(m == args.cat_encoding for m in cat_modes):
                cat_modes = None  # sin overrides: config mas limpia

    n_num_model = len(keep_num)
    som_weights = None
    if args.som_feature:
        # Kohonen (SIA): SOM sobre las numericas de TRAIN; la celda BMU entra
        # como una categorica extra (indices 1..G^2; el 0 queda como UNK)
        g = args.som_feature
        som_weights = entrenar_som(splits['train'][1].cpu().numpy(), g, seed)
        nuevos = {}
        for kk, (a_, b_, c_, y_) in splits.items():
            celdas = torch.as_tensor(asignar_som(b_.cpu().numpy(), som_weights) + 1,
                                     dtype=torch.long, device=a_.device)
            nuevos[kk] = (torch.cat([a_, celdas.unsqueeze(1)], dim=1), b_, c_, y_)
        splits = nuevos
        cardinalities = cardinalities + [g * g + 1]
        modos = list(cat_modes) if cat_modes else [args.cat_encoding] * (len(cardinalities) - 1)
        modos.append('embedding')  # la celda no tiene orden: siempre embedding
        cat_modes = None if all(m == 'embedding' for m in modos) else modos
        if not args.quiet:
            print(f"  som {g}x{g}: celda BMU agregada como categorica ({g * g + 1} niveles)")

    if args.ae_latent or args.pca:
        # SIA (TP5 / PCA): comprimir [one-hot|numericas] SIN mirar el target y
        # entrenar el MLP sobre esa representacion (feature extraction, clase 3)
        k = args.ae_latent or args.pca
        m_tr = matriz_cruda(splits['train'][0], splits['train'][1], cardinalities)
        if args.pca:
            media, comp, std = ajustar_pca(m_tr, k)
            transf = lambda a_, b_: ((matriz_cruda(a_, b_, cardinalities) - media) @ comp) / std
        else:
            enc = ajustar_ae(m_tr, k, device, seed=seed, verbose=not args.quiet)
            with torch.no_grad():
                lat = enc(m_tr)
            media_l = lat.mean(0, keepdim=True)
            std_l = lat.std(0, keepdim=True).clamp_min(1e-6)

            def transf(a_, b_):
                with torch.no_grad():
                    return (enc(matriz_cruda(a_, b_, cardinalities)) - media_l) / std_l
        nuevos = {}
        for kk, (a_, b_, c_, y_) in splits.items():
            vacio = torch.zeros(a_.shape[0], 0, dtype=torch.long, device=a_.device)
            nuevos[kk] = (vacio, transf(a_, b_), c_, y_)
        splits = nuevos
        cardinalities, cat_tables, cat_modes = [], None, None
        n_num_model = k
        if not args.quiet:
            fuente = 'PCA' if args.pca else 'autoencoder'
            print(f"  {fuente}: entrada del MLP = {k} dimensiones (blanqueadas con train)")

    text_emb_dim = 0
    if args.text_emb or args.text_emb_finetune:
        # transfer learning desde un preentrenado EXTERNO (8va tanda): los dos
        # regimenes de la clase 3 sobre el MISMO texto (title+description).
        # Los tensores de transform() preservan el orden del df, asi que el
        # indice de fila del split recupera la fila del embedding precomputado.
        df_emb = load_dataset(csv_path)
        dfs = dict(zip(('train', 'val', 'test'), split_by_query(df_emb, seed=seed)))
        nuevos = {}
        if args.text_emb:
            E = np.load(args.text_emb)
            if E.shape[0] != len(df_emb):
                raise SystemExit(f'--text-emb: {E.shape[0]} filas vs {len(df_emb)} del CSV '
                                 '(regenerar con eda/embed_texto.py)')
            for kk, (a_, b_, c_, y_) in splits.items():
                e = torch.tensor(E[dfs[kk].index.to_numpy()], dtype=torch.float32,
                                 device=a_.device)
                assert e.shape[0] == a_.shape[0], 'desalineacion embeddings/split'
                if args.arch == 'mlp':
                    nuevos[kk] = (a_, torch.cat([b_, e], dim=1), c_, y_)
                else:
                    nuevos[kk] = (a_, b_, e, y_)  # el embedding ocupa el slot del texto
            if args.arch == 'mlp':
                n_num_model += E.shape[1]
            else:
                text_emb_dim = E.shape[1]
            if not args.quiet:
                print(f"  text-emb: {Path(args.text_emb).name} "
                      f"({E.shape[1]} dims congeladas)")
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.text_emb_finetune)
            if tok.pad_token_id != 0:
                raise SystemExit('--text-emb-finetune: el modelo debe usar pad_token_id=0 '
                                 '(la mascara del grafo asume pad=0)')
            for kk, (a_, b_, c_, y_) in splits.items():
                pares = zip(dfs[kk]['title'], dfs[kk]['description'])
                if args.strip_status:
                    pares = (strip_status_from_text(t, d) for t, d in pares)
                textos = [t + '\n' + d for t, d in pares]
                enc = tok(textos, padding=True, truncation=True, max_length=128,
                          return_tensors='pt')
                nuevos[kk] = (a_, b_, enc['input_ids'].to(a_.device), y_)
            if not args.quiet:
                print(f"  text-emb-finetune: {args.text_emb_finetune} "
                      f"(lr encoder {args.text_emb_lr:g})")
        splits = nuevos

    if args.embed_from:
        # feature extraction (clase 3): el embedding pooled del transformer
        # CONGELADO, concatenado a las numericas del MLP ("el embedding mas un
        # monton de cosas")
        rutas = expandir_ckpts(args.embed_from, seed)
        if len(rutas) != 1:
            raise SystemExit(f'--embed-from espera UN checkpoint, matchearon {len(rutas)}')
        from .model import load_checkpoint
        extractor, _ = load_checkpoint(rutas[0], device=device)
        nuevos = {}
        for kk, (a_, b_, c_, y_) in splits.items():
            emb = extractor.representar(a_, b_)
            nuevos[kk] = (a_, torch.cat([b_, emb], dim=1), c_, y_)
        splits = nuevos
        n_num_model += emb.shape[1]
        if not args.quiet:
            print(f"  embed-from: +{emb.shape[1]} numericas desde {rutas[0].name}")
        del extractor

    model, model_config = build_model(args, prep, cardinalities, n_num_model,
                                      bin_edges, pos_weight, max_products, cat_tables,
                                      cat_modes, text_emb_dim=text_emb_dim)
    model = model.to(device)
    if args.w2v_init:
        tabla = None
        if getattr(model, 'text_encoder', None) is not None:
            tabla = model.text_encoder.char_embedding_table
        elif getattr(model, 'char_embedding_table', None) is not None:
            tabla = model.char_embedding_table
        if tabla is None:
            raise SystemExit('--w2v-init requiere una arquitectura que vea el texto')
        pesos_w2v = pretrain_w2v(train_df, prep, args.d_model, device, verbose=not args.quiet)
        with torch.no_grad():
            tabla.weight.copy_(pesos_w2v)
    if args.init_from:
        # fine-tuning (clase 3): arrancar de pesos ya entrenados en vez de aleatorios
        rutas = expandir_ckpts(args.init_from, seed)
        if len(rutas) != 1:
            raise SystemExit(f'--init-from espera UN checkpoint, matchearon {len(rutas)}')
        ckpt_ini = torch.load(rutas[0], map_location=device, weights_only=False)
        model.load_state_dict(ckpt_ini['state_dict'])
        if args.reinit_head:
            # probe honesto: la cabeza vuelve a aleatorio; lo que se mide es la
            # REPRESENTACION congelada, no la cabeza ya entrenada
            model.cls_head.apply(model._init_weights)
        if not args.quiet:
            print(f"  init-from: {rutas[0].name}"
                  + (' (cabeza reinicializada)' if args.reinit_head else ''))

    n_params = sum(p.numel() for p in model.parameters())
    name = run_name(args, seed)
    print(f"\n=== {name} | {n_params:,} parametros | device {device} ===")

    if args.pretrain_mlm:
        pretrain_mlm(model, splits['train'], cardinalities, epochs=args.pretrain_mlm,
                     batch_size=args.batch_size, lr=args.lr, device=device,
                     verbose=not args.quiet)
    if args.pretrain_ae:
        pretrain_ae(model, splits['train'], cardinalities, epochs=args.pretrain_ae,
                    batch_size=args.batch_size, lr=args.lr, device=device,
                    verbose=not args.quiet)

    if args.freeze_backbone:
        # feature extraction (clase 3): congelar el tronco; entrena SOLO la cabeza
        congelados = 0
        for nom, p in model.named_parameters():
            if not nom.startswith(('cls_head', 'cart_head')):
                p.requires_grad = False
                congelados += p.numel()
        if not args.quiet:
            entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  backbone congelado ({congelados:,} params); "
                  f"entrena solo la cabeza ({entrenables:,})")

    l2sp_ref = None
    if args.l2sp:
        # snapshot POST pre-training: el "modelo del que no quiero alejarme"
        l2sp_ref = {n: p.detach().clone() for n, p in model.named_parameters()
                    if p.requires_grad}

    train_targets = None
    y_tr = splits['train'][3]
    if args.label_smoothing:
        s = args.label_smoothing
        train_targets = y_tr * (1 - s) + s / 2
    if args.distill_from:
        rutas = expandir_ckpts(args.distill_from, seed)
        p_teacher = probs_teachers(rutas, splits['train'][0], splits['train'][1],
                                   device, verbose=not args.quiet)
        al = args.distill_alpha
        train_targets = al * p_teacher + (1 - al) * y_tr

    history = train_model(model, splits, epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, patience=args.patience, verbose=not args.quiet,
                          weight_decay=args.weight_decay, train_targets=train_targets,
                          l2sp=args.l2sp, l2sp_ref=l2sp_ref, text_emb_lr=args.text_emb_lr)
    val_m = evaluate(model, splits['val'])
    test_m = evaluate(model, splits['test'])
    print(f"{name} -> VAL: ROC-AUC {val_m['roc_auc']:.4f} PR-AUC {val_m['pr_auc']:.4f} | "
          f"TEST: loss {test_m['loss']:.4f} ROC-AUC {test_m['roc_auc']:.4f} "
          f"PR-AUC {test_m['pr_auc']:.4f}")

    if not args.no_save:
        results_dir = REPO_ROOT / 'resultados'
        results_dir.mkdir(exist_ok=True)
        out = unique_path(results_dir / f'{name}.json')
        out.write_text(json.dumps({
            'nombre': out.stem,
            'fecha': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'seed': seed,
            'device': device,
            'n_parametros': n_params,
            'config': {k: v for k, v in vars(args).items() if k not in ('quiet', 'no_save', 'save_pesos')},
            'historial': history,
            'val': val_m,
            'test': test_m,
        }, indent=2))
        print(f"resultados -> {out.relative_to(REPO_ROOT)}")

        if args.save_pesos and args.text_emb_finetune:
            print('  (checkpoint NO guardado: el encoder HF fine-tuneado pesa ~90MB '
                  'y la corrida es exploratoria)')
        elif args.save_pesos:
            pesos_dir = REPO_ROOT / 'pesos'
            pesos_dir.mkdir(exist_ok=True)
            ckpt = pesos_dir / f'{out.stem}.pt'  # mismo nombre que el JSON de resultados
            torch.save({
                'arch': args.arch,
                'model_config': model_config,
                'bin_edges': bin_edges,
                'cat_tables': cat_tables,
                'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'preprocessor': prep,
                'cat_features': [prep.cats[i] for i in keep_cat],
                'num_features': [prep.nums[i] for i in keep_num],
                'som': som_weights,  # pesos del SOM (--som-feature); None si no aplica
            }, ckpt)
            print(f"pesos      -> {ckpt.relative_to(REPO_ROOT)}")

    return test_m['roc_auc'], test_m['pr_auc']


def build_parser():
    parser = argparse.ArgumentParser(description='Entrena los modelos de BTR')
    parser.add_argument('--csv', default=str(REPO_ROOT / 'supermarket_products.csv'))
    parser.add_argument('--seeds', type=int, default=1, help='cantidad de corridas a promediar')
    parser.add_argument('--seed-start', type=int, default=42, help='primera seed de la serie')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--tag', default='', help='prefijo para el nombre de la corrida')
    parser.add_argument('--arch', choices=['transformer', 'mlp', 'tower', 'listwise',
                                           'ing_tower'],
                        default='transformer')
    parser.add_argument('--formulation', choices=['features', 'text', 'hybrid', 'fusion',
                                                  'ing', 'ing_fusion', 'ing_hybrid'],
                        default='features',
                        help='que es un token (solo aplica a --arch transformer)')
    parser.add_argument('--max-text-len', type=int, default=MAX_TEXT_LEN)
    parser.add_argument('--strip-status', action='store_true',
                        help='texto sin sufijo/oracion de estado (variante "producto nuevo")')
    parser.add_argument('--drop-features', default='',
                        help='features a excluir, separados por coma (ej: listing_status)')
    parser.add_argument('--extra-features', default='',
                        help=f'features descartados a reintroducir, separados por coma '
                             f'(o "all"): {sorted(EXTRA_FEATURES)}')
    parser.add_argument('--cat-encoding', default='embedding',
                        choices=['embedding', 'onehot', 'target', 'freq', 'ordinal', 'hashing'],
                        help='encoding de las categoricas (onehot: solo --arch mlp)')
    parser.add_argument('--hash-buckets', type=int, default=8,
                        help='buckets del hashing trick (--cat-encoding hashing)')
    parser.add_argument('--cat-feature-encoding', default='', metavar='F=MODO,...',
                        help='override de encoding POR feature categorica, p. ej. '
                             'listing_status=ordinal (el resto usa --cat-encoding)')
    parser.add_argument('--cls-position', default='first', choices=['first', 'last'],
                        help='last: CLS al final (necesario para que --causal tenga sentido)')
    parser.add_argument('--cart-aux', type=float, default=0.0, metavar='LAMBDA',
                        help='multi-task: peso de la BCE auxiliar sobre cart (0 = apagado)')
    parser.add_argument('--listwise-texto', action='store_true',
                        help='listwise: enriquecer el token de producto con la torre de texto')
    parser.add_argument('--ing-layer', type=int, default=1,
                        help='bloques del encoder de conjunto de ingredientes '
                             '(ing_fusion / ing_tower; la lista tiene <= 5 items)')
    parser.add_argument('--text-tokens', default='chars', choices=['chars', 'words'],
                        help='tokenizacion del texto: caracteres (demo) o palabras (5b)')
    parser.add_argument('--w2v-init', action='store_true',
                        help='pre-entrenar los embeddings de palabras con skipgram sobre el '
                             'corpus de train (requiere --text-tokens words)')
    parser.add_argument('--train-frac', type=float, default=1.0, metavar='F',
                        help='curva de aprendizaje: fraccion de las QUERIES de train (val/test intactos)')
    parser.add_argument('--init-seed', type=int, default=None, metavar='N',
                        help='seed de inicializacion/entrenamiento independiente del split '
                             '(default: la misma seed; sirve para separar varianza y deep-ensembles)')
    parser.add_argument('--pretrain-mlm', type=int, default=0, metavar='EPOCHS',
                        help='pre-entrenar el tronco enmascarando una feature por fila '
                             '(solo transformer features, CLS al inicio)')
    parser.add_argument('--cv-k', type=int, default=0, help='GroupKFold por query: cantidad de folds')
    parser.add_argument('--per-feature', default='none',
                        choices=['none', 'qkv', 'ffn', 'both', 'gate'],
                        help='parametros PROPIOS por posicion/feature dentro del transformer: '
                             'W_q/W_k/W_v y/o la FFN (solo formulation features)')
    parser.add_argument('--cv-fold', type=int, default=0, help='que fold es test (0..k-1)')
    parser.add_argument('--d-model', type=int, default=32)
    parser.add_argument('--n-head', type=int, default=4)
    parser.add_argument('--n-layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    # ---- regularizacion (6ta tanda: nunca barrida hasta ahora) ----
    parser.add_argument('--weight-decay', type=float, default=1e-2,
                        help='weight decay de AdamW (1e-2 fue el default implicito '
                             'de todas las corridas previas)')
    parser.add_argument('--feature-dropout', type=float, default=0.0, metavar='P',
                        help='anular tokens de features al azar en TRAIN (nunca el CLS); '
                             'augmentation "faltan features" (solo transformer features)')
    parser.add_argument('--label-smoothing', type=float, default=0.0, metavar='S',
                        help="suavizar las labels de TRAIN: y' = y(1-S) + S/2 "
                             '(= distillation con teacher uniforme)')
    parser.add_argument('--sin-residual', action='store_true',
                        help='ablacion: bloques SIN conexiones residuales')
    parser.add_argument('--sin-layernorm', action='store_true',
                        help='ablacion: bloques SIN LayerNorm (ni ln_f)')
    # ---- transfer learning (clase 3: feature extraction / fine-tuning / distillation) ----
    parser.add_argument('--init-from', default='', metavar='CKPT',
                        help='fine-tuning: cargar pesos iniciales de un checkpoint '
                             '(admite {seed} y glob; debe matchear la arquitectura)')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='feature extraction: congelar todo salvo la cabeza '
                             '(linear probe sobre la representacion)')
    parser.add_argument('--reinit-head', action='store_true',
                        help='reinicializar la cabeza tras --init-from (probe honesto)')
    parser.add_argument('--l2sp', type=float, default=0.0, metavar='LAMBDA',
                        help='penalidad L2 hacia los pesos post pre-training (el analogo '
                             'de la KL penalty de la clase 3); requiere --init-from, '
                             '--pretrain-mlm o --pretrain-ae')
    parser.add_argument('--distill-from', default='', metavar='CKPTS',
                        help='knowledge distillation: entrenar contra las PROBABILIDADES '
                             'del promedio de estos checkpoints (patrones con coma, '
                             '{seed} y glob); teacher(s) del MISMO split')
    parser.add_argument('--distill-alpha', type=float, default=1.0, metavar='A',
                        help='target = A*prob_teacher + (1-A)*label dura (1 = puro soft)')
    parser.add_argument('--embed-from', default='', metavar='CKPT',
                        help='feature extraction para --arch mlp: concatenar el embedding '
                             'pooled (congelado) de este transformer a las numericas')
    # ---- herramientas de SIA (Kohonen / PCA / autoencoder) ----
    parser.add_argument('--som-feature', type=int, default=0, metavar='G',
                        help='Kohonen: entrenar un SOM GxG sobre las numericas de train '
                             'y agregar la celda BMU como feature categorica extra')
    parser.add_argument('--pretrain-ae', type=int, default=0, metavar='EPOCHS',
                        help='pre-entrenar el tronco como autoencoder: el CLS debe '
                             'reconstruir todas las features (cuello de botella d_model)')
    parser.add_argument('--ae-latent', type=int, default=0, metavar='K',
                        help='--arch mlp: reemplazar la entrada por el espacio latente K '
                             'de un autoencoder sobre [one-hot|numericas] (SIA TP5)')
    parser.add_argument('--text-emb', default='', metavar='NPY',
                        help='transfer desde un preentrenado EXTERNO, feature extraction: '
                             'matriz (N, E) precomputada por eda/embed_texto.py alineada al '
                             'CSV; entra como UN token extra (transformer features) o como '
                             'E numericas extra (mlp)')
    parser.add_argument('--text-emb-finetune', default='', metavar='HF_MODEL',
                        help='transfer desde un preentrenado EXTERNO, fine-tuning: el '
                             'encoder HF (p. ej. sentence-transformers/all-MiniLM-L6-v2) '
                             'entra al grafo y se actualiza con --text-emb-lr')
    parser.add_argument('--text-emb-lr', type=float, default=1e-5,
                        help='lr del encoder preentrenado con --text-emb-finetune')
    parser.add_argument('--pca', type=int, default=0, metavar='K',
                        help='--arch mlp: reemplazar la entrada por las K primeras '
                             'componentes principales de [one-hot|numericas]')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--numeric-mode', choices=['linear', 'bins'], default='linear')
    parser.add_argument('--n-bins', type=int, default=16)
    parser.add_argument('--pooling', choices=['cls', 'mean'], default='cls')
    parser.add_argument('--positional', action='store_true',
                        help='ablacion: positional encoding aprendido (en text/hybrid va siempre)')
    parser.add_argument('--causal', action='store_true', help='ablacion: mascara causal del decoder')
    parser.add_argument('--pos-weight', action='store_true', help='pesar la clase positiva en la BCE')
    parser.add_argument('--no-save', action='store_true', help='no escribir resultados/')
    parser.add_argument('--save-pesos', action='store_true', help='guardar checkpoint en pesos/')
    parser.add_argument('--quiet', action='store_true', help='no imprimir el log por epoca')
    return parser


def main():
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    results = [run(args.csv, seed, args, device)
               for seed in range(args.seed_start, args.seed_start + args.seeds)]
    rocs, prs = zip(*results)
    print(f"\n===== {args.seeds} corrida(s) | TEST ROC-AUC {np.mean(rocs):.4f} +- {np.std(rocs):.4f} "
          f"| TEST PR-AUC {np.mean(prs):.4f} +- {np.std(prs):.4f} =====")


if __name__ == '__main__':
    main()
