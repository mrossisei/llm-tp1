"""Entrenamiento y evaluacion de los modelos de BTR.

Protocolo de propuesta.md (seccion 7): split por query, AdamW, early stopping
por PR-AUC de validacion, promedio de corridas con seeds distintas. Cada epoca
y cada split final guardan TODAS las metricas (ver compute_metrics): las
decisiones se toman con PR-AUC (desbalance 13%) pero el resto queda registrado
para poder graficar cualquiera despues. Cada corrida escribe salidas/resultados/<nombre>.json
y, con --save-pesos, salidas/pesos/<nombre>.pt (recargable con btr.model.load_checkpoint).

Arquitecturas (--arch) y formulaciones (--formulation), ver propuesta.md 4:
  transformer + features   cada feature tabular es un token (FT-Transformer)
  transformer + text       cada caracter de title+description es un token (demo)
  transformer + hybrid     [CLS] + features + caracteres en una secuencia
  mlp                      baseline sin atencion (mismos embeddings, MLP denso)
  tower                    transformer SOLO como encoder de texto -> embedding
                           que se concatena con lo tabular y clasifica un MLP
  transformer + fusion     una torre de texto resume los caracteres a UN token
                           que entra a la secuencia tabular
  transformer + ing        SOLO los ingredientes como tokens (conjunto)
  transformer + ing_fusion encoder de conjunto de ingredientes -> su [ING] entra
                           como un token mas de la secuencia tabular
  transformer + ing_hybrid un token POR ingrediente en la secuencia tabular

Ejes transversales: --cat-encoding (embedding / ordinal / target / freq /
hashing; onehot solo para el MLP), --drop-features listing_status (modelo sin
el estado parseado), --strip-status (texto sin sufijo ni oracion de estado: la
variante "producto nuevo"), --numeric-mode bins, --positional, --causal,
--pooling, --pos-weight, --cls-position, --pretrain-mlm, --w2v-init,
--train-frac, --init-seed, --cv-k. La suite curada esta en experimentos.py.

Disciplina: las decisiones (hiperparametros, early stopping) se toman con
VALIDACION; test se mira solo para reportar las configuraciones finales.
"""

import argparse
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

from .data import (CAT_FEATURES, MAX_INGREDIENTS, MAX_TEXT_LEN, NUM_FEATURES, TARGET,
                   _palabras, prepare)
from .model import ING_FORMULATIONS, BTRTransformer, MLPBaseline, TextTowerModel

REPO_ROOT = Path(__file__).resolve().parent.parent
SALIDAS = REPO_ROOT / 'salidas'   # resultados/ (json por corrida) y pesos/ (checkpoints)
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

    Los splits son 4-tuplas posicionales (x_cat, x_num, x_text, y) que se pasan
    tal cual al modelo: model(x_cat, x_num, x_text, y).
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
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=model.pos_weight
        ).item()
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = y.cpu().numpy()
    return compute_metrics(y_true, probs, loss)


def train_model(model, splits, epochs=60, batch_size=256, lr=1e-3, patience=8, verbose=True,
                weight_decay=1e-2):
    """Entrena con early stopping por PR-AUC de validacion; restaura el mejor estado."""
    a, b, c, y = splits['train']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history, best_pr, best_state, since_best = [], -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(a.shape[0], device=a.device)
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            _, loss = model(a[idx], b[idx], c[idx], y[idx])
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


def resolve_device(arg):
    if arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return arg


def run_name(args, seed):
    base = args.formulation if args.arch == 'transformer' else args.arch
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
    if args.dropout != 0.1:
        parts.append(f"do{args.dropout:g}")
    if args.weight_decay != 1e-2:
        parts.append(f"wd{args.weight_decay:g}")
    if args.strip_status:
        parts.append('stripstatus')
    if args.drop_features:
        parts.append('sin-' + args.drop_features.replace(',', '-').replace('_', ''))
    if args.pooling != 'cls':
        parts.append(args.pooling)
    if args.cls_position != 'first':
        parts.append('clslast')
    if getattr(args, 'mlp_hidden', ''):
        parts.append('mh' + args.mlp_hidden.replace(',', 'x'))
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


def drop_feature_columns(splits, drop, cat_features=None, num_features=None):
    """Saca features por nombre (p. ej. listing_status) recortando columnas de los tensores."""
    cat_features = cat_features or CAT_FEATURES
    num_features = num_features or NUM_FEATURES
    keep_cat = [i for i, f in enumerate(cat_features) if f not in drop]
    keep_num = [i for i, f in enumerate(num_features) if f not in drop]
    unknown = drop - set(cat_features) - set(num_features)
    if unknown:
        raise SystemExit(f'--drop-features desconocidos: {sorted(unknown)}')
    splits = {k: (v[0].index_select(1, torch.tensor(keep_cat, dtype=torch.long)),
                  v[1].index_select(1, torch.tensor(keep_num, dtype=torch.long)),
                  v[2], v[3]) for k, v in splits.items()}
    return splits, keep_cat, keep_num


def build_cat_tables(modo, prep, train_df, keep_cat, hash_buckets):
    """Lookups por feature categorica para los encodings que los requieren.

    Ajustados SOLO con train, alineados a los indices de prep.vocabs (0 = UNK):
      target:  nivel -> media suavizada de bought: (sum + m*global) / (n + m), m=50
               (el suavizado amortigua niveles chicos y la auto-inclusion del target)
      ordinal: nivel -> rango del nivel al ordenar por esa media suavizada,
               normalizado a [0,1] (UNK -> 0.5). Conserva el ORDEN aprendible de
               los datos y descarta las magnitudes; un orden semantico "a mano"
               seria indefendible (EDA: el wording no predice el tier).
      freq:    nivel -> frecuencia relativa del nivel en train
      hashing: nivel -> md5(feature|valor) % B  (el "modulo" clasico del hashing trick)

    Devuelve dict {posicion en la lista kept: tensor} o None si el modo no lo necesita.
    """
    import hashlib
    if modo in ('embedding', 'onehot'):
        return None
    m, global_mean = 50.0, float(train_df[TARGET].mean())
    tablas = {}
    for pos, i in enumerate(keep_cat):
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


def build_model(args, prep, cardinalities, n_numeric, bin_edges, pos_weight, cat_tables=None):
    """Configura y construye la arquitectura pedida; devuelve (modelo, config para el ckpt)."""
    common = dict(d_model=args.d_model, dropout=args.dropout,
                  numeric_mode=args.numeric_mode, pos_weight=pos_weight)
    encod = dict(cat_encoding=args.cat_encoding, hash_buckets=args.hash_buckets)
    if args.arch == 'transformer':
        config = dict(formulation=args.formulation, cat_cardinalities=cardinalities,
                      n_numeric=n_numeric, char_vocab_size=prep.char_vocab_size,
                      max_text_len=prep.max_text_len, n_head=args.n_head,
                      n_layer=args.n_layer, causal=args.causal, pooling=args.pooling,
                      use_positional=args.positional, cls_position=args.cls_position,
                      ing_vocab_size=(prep.ing_vocab_size
                                      if args.formulation in ING_FORMULATIONS else None),
                      max_ingredients=(MAX_INGREDIENTS
                                       if args.formulation in ING_FORMULATIONS else 0),
                      **encod, **common)
        model = BTRTransformer(**config, bin_edges=bin_edges, cat_tables=cat_tables)
    elif args.arch == 'mlp':
        config = dict(cat_cardinalities=cardinalities, n_numeric=n_numeric,
                      mlp_hidden=(args.mlp_hidden or None), **encod, **common)
        model = MLPBaseline(**config, bin_edges=bin_edges, cat_tables=cat_tables)
    elif args.arch == 'tower':
        config = dict(cat_cardinalities=cardinalities, n_numeric=n_numeric,
                      char_vocab_size=prep.char_vocab_size, max_text_len=prep.max_text_len,
                      n_head=args.n_head, n_layer=args.n_layer, causal=args.causal,
                      **encod, **common)
        model = TextTowerModel(**config, bin_edges=bin_edges, cat_tables=cat_tables)
    else:
        raise SystemExit(f'arquitectura desconocida: {args.arch}')
    return model, config


def run(csv_path, seed, args, device):
    """Una corrida completa: prepara datos, entrena, guarda resultados/pesos."""
    torch.manual_seed(seed if args.init_seed is None else args.init_seed)
    usa_ing = args.formulation in ING_FORMULATIONS
    if args.pretrain_mlm and (args.arch != 'transformer' or args.formulation != 'features'
                              or args.cls_position != 'first'):
        raise SystemExit('--pretrain-mlm es solo para transformer features con CLS al inicio')
    if args.cat_encoding == 'onehot' and args.arch != 'mlp':
        raise SystemExit('cat-encoding onehot es solo para --arch mlp: para el transformer, '
                         'one-hot + proyeccion lineal aprende la misma matriz que el '
                         'embedding (propuesta 6.1) — no hay experimento que correr')
    if args.w2v_init and args.text_tokens != 'words':
        raise SystemExit('--w2v-init requiere --text-tokens words')
    if usa_ing and args.w2v_init:
        raise SystemExit('ingredientes: la lista ocupa el slot del texto; no se combina '
                         'con --w2v-init')
    prep, train_df, splits = prepare(csv_path, seed=seed, max_text_len=args.max_text_len,
                                     strip_status=args.strip_status,
                                     text_tokens=args.text_tokens, train_frac=args.train_frac,
                                     cv_k=args.cv_k, cv_fold=args.cv_fold,
                                     use_ingredients=usa_ing)

    drop = {f.strip() for f in args.drop_features.split(',') if f.strip()}
    splits, keep_cat, keep_num = drop_feature_columns(splits, drop, prep.cats, prep.nums)
    splits = {k: tuple(t.to(device) for t in v) for k, v in splits.items()}

    bin_edges = None
    if args.numeric_mode == 'bins':
        bin_edges = prep.bin_edges(train_df, args.n_bins)[keep_num]
    pos_weight = None
    if args.pos_weight:
        y_train = splits['train'][3]
        pos_weight = ((1 - y_train.mean()) / y_train.mean()).item()  # negativos/positivos

    cardinalities = [c for i, c in enumerate(prep.cat_cardinalities) if i in keep_cat]
    cat_tables = build_cat_tables(args.cat_encoding, prep, train_df, keep_cat,
                                  args.hash_buckets)

    model, model_config = build_model(args, prep, cardinalities, len(keep_num), bin_edges,
                                      pos_weight, cat_tables)
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

    n_params = sum(p.numel() for p in model.parameters())
    name = run_name(args, seed)
    print(f"\n=== {name} | {n_params:,} parametros | device {device} ===")

    if args.pretrain_mlm:
        pretrain_mlm(model, splits['train'], cardinalities, epochs=args.pretrain_mlm,
                     batch_size=args.batch_size, lr=args.lr, device=device,
                     verbose=not args.quiet)

    history = train_model(model, splits, epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, patience=args.patience, verbose=not args.quiet,
                          weight_decay=args.weight_decay)
    val_m = evaluate(model, splits['val'])
    test_m = evaluate(model, splits['test'])
    print(f"{name} -> VAL: ROC-AUC {val_m['roc_auc']:.4f} PR-AUC {val_m['pr_auc']:.4f} | "
          f"TEST: loss {test_m['loss']:.4f} ROC-AUC {test_m['roc_auc']:.4f} "
          f"PR-AUC {test_m['pr_auc']:.4f}")

    if not args.no_save:
        results_dir = SALIDAS / 'resultados'
        results_dir.mkdir(parents=True, exist_ok=True)
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

        if args.save_pesos:
            pesos_dir = SALIDAS / 'pesos'
            pesos_dir.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument('--arch', choices=['transformer', 'mlp', 'tower'], default='transformer')
    parser.add_argument('--formulation', choices=['features', 'text', 'hybrid', 'fusion',
                                                  'ing', 'ing_fusion', 'ing_hybrid'],
                        default='features',
                        help='que es un token (solo aplica a --arch transformer)')
    parser.add_argument('--max-text-len', type=int, default=MAX_TEXT_LEN)
    parser.add_argument('--strip-status', action='store_true',
                        help='texto sin sufijo/oracion de estado (variante "producto nuevo")')
    parser.add_argument('--drop-features', default='',
                        help='features a excluir, separados por coma (ej: listing_status)')
    parser.add_argument('--cat-encoding', default='embedding',
                        choices=['embedding', 'onehot', 'target', 'freq', 'ordinal', 'hashing'],
                        help='encoding de las categoricas (onehot: solo --arch mlp)')
    parser.add_argument('--hash-buckets', type=int, default=8,
                        help='buckets del hashing trick (--cat-encoding hashing)')
    parser.add_argument('--cls-position', default='first', choices=['first', 'last'],
                        help='last: CLS al final (necesario para que --causal tenga sentido)')
    parser.add_argument('--text-tokens', default='chars', choices=['chars', 'words'],
                        help='tokenizacion del texto: caracteres (demo) o palabras')
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
    parser.add_argument('--cv-fold', type=int, default=0, help='que fold es test (0..k-1)')
    parser.add_argument('--d-model', type=int, default=32)
    parser.add_argument('--n-head', type=int, default=4)
    parser.add_argument('--n-layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='weight decay de AdamW')
    parser.add_argument('--mlp-hidden', default='', metavar='N,N,...',
                        help='--arch mlp: capas ocultas de la cabeza, p. ej. 256,128 '
                             '(default: 8d,2d = 256,64)')
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
    parser.add_argument('--no-save', action='store_true', help='no escribir salidas/resultados/')
    parser.add_argument('--save-pesos', action='store_true', help='guardar checkpoint en salidas/pesos/')
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
