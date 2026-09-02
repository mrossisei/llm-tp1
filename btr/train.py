"""Entrenamiento y evaluacion de los modelos de BTR.

Protocolo de propuesta.md (seccion 7): split por query, AdamW, early stopping
por PR-AUC de validacion, promedio de corridas con seeds distintas. Cada epoca
y cada split final guardan TODAS las metricas (ver compute_metrics): las
decisiones se toman con PR-AUC (desbalance 13%) pero el resto queda registrado
para poder graficar cualquiera despues. Cada corrida escribe salidas/resultados/<nombre>.json
y, con --save-pesos, salidas/pesos/<nombre>.pt (recargable con btr.model.load_checkpoint).

Formulaciones (--formulation), ver propuesta.md 4 y btr/model.py:
  features     cada feature tabular es un token (FT-Transformer) — el modelo final
  ing_fusion   un encoder de conjunto resume los ingredientes a UN token mas
  ing_hybrid   un token POR ingrediente en la secuencia tabular
  ing          SOLO los ingredientes (control)
Transfer learning (clase 3), sobre formulation features:
  --text-emb NPY           el embedding del TITULO (sin badge) de un preentrenado,
                           precomputado por eda/embed_titulos.py, entra como un
                           token mas (feature extraction, congelado)
  --text-emb-finetune HF   el encoder de Hugging Face entra al grafo y se ajusta
                           con --text-emb-lr (fine-tuning)

Ejes transversales: --cat-encoding (embedding / ordinal / target / freq / hashing),
--drop-features listing_status (modelo sin el estado parseado), --numeric-mode bins,
--positional, --causal, --pooling, --pos-weight, --cls-position, --pretrain-mlm,
--train-frac, --init-seed, --cv-k, --dropout, --weight-decay, --lr, --batch-size.
La suite curada esta en experimentos.py.

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

from .data import (CAT_FEATURES, MAX_INGREDIENTS, NUM_FEATURES, TARGET, load_dataset,
                   prepare, titulo_sin_estado)
from .model import ING_FORMULATIONS, BTRTransformer

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
                weight_decay=1e-2, text_emb_lr=1e-5):
    """Entrena con early stopping por PR-AUC de validacion; restaura el mejor estado.

    text_emb_lr: con --text-emb-finetune, lr chico y separado para el encoder
    preentrenado (no destruir lo aprendido); lr normal para el resto.
    """
    a, b, c, y = splits['train']
    params = list(model.parameters())
    if getattr(model, 'hf_encoder', None) is not None:
        hf_ids = {id(q) for q in model.hf_encoder.parameters()}
        grupos = [{'params': [q for q in params if id(q) not in hf_ids]},
                  {'params': [q for q in params if id(q) in hf_ids], 'lr': text_emb_lr}]
        optimizer = torch.optim.AdamW(grupos, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
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
    parts = [args.formulation, f"d{args.d_model}", f"h{args.n_head}", f"l{args.n_layer}",
             args.numeric_mode]
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
    if args.lr != 1e-3:
        parts.append(f"lr{args.lr:g}")
    if args.batch_size != 256:
        parts.append(f"bs{args.batch_size}")
    if args.text_emb:
        parts.append('temb-' + Path(args.text_emb).stem.replace('_', ''))
    if args.text_emb_finetune:
        # 'titulo': el texto que ve el encoder es el titulo sin badge (distinto de las corridas
        # viejas bert_*, que veian titulo + descripcion con el badge)
        parts.append('tembft-titulo' + (f"-lr{args.text_emb_lr:g}" if args.text_emb_lr != 1e-5 else ''))
    if args.drop_features:
        parts.append('sin-' + args.drop_features.replace(',', '-').replace('_', ''))
    if args.pooling != 'cls':
        parts.append(args.pooling)
    if args.cls_position != 'first':
        parts.append('clslast')
    if args.ing_layer != 1:
        parts.append(f"il{args.ing_layer}")
    if args.ing_d_model:
        parts.append(f"ingd{args.ing_d_model}")
    if args.ing_head:
        parts.append(f"ingh{args.ing_head}")
    if args.tiempo != 'none':
        parts.append({'ciclico': 'tiempo-ciclico', 'categorico': 'tiempo-cat'}[args.tiempo])
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


def drop_feature_columns(splits, drop, cat_features=None, num_features=None, n_cyclic=0):
    """Saca features por nombre (p. ej. listing_status) recortando columnas de los tensores.

    Las 2*n_cyclic columnas del final de x_num (los pares sin/cos de --tiempo ciclico) se
    conservan siempre.
    """
    cat_features = cat_features or CAT_FEATURES
    num_features = num_features or NUM_FEATURES
    keep_cat = [i for i, f in enumerate(cat_features) if f not in drop]
    keep_num = [i for i, f in enumerate(num_features) if f not in drop]
    unknown = drop - set(cat_features) - set(num_features)
    if unknown:
        raise SystemExit(f'--drop-features desconocidos: {sorted(unknown)}')
    cols_num = keep_num + [len(num_features) + j for j in range(2 * n_cyclic)]
    splits = {k: (v[0].index_select(1, torch.tensor(keep_cat, dtype=torch.long)),
                  v[1].index_select(1, torch.tensor(cols_num, dtype=torch.long)),
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


def build_model(args, prep, cardinalities, n_numeric, bin_edges, pos_weight, cat_tables=None,
                text_emb_dim=0, n_cyclic=0):
    """Configura y construye el transformer pedido; devuelve (modelo, config para el ckpt)."""
    usa_ing = args.formulation in ING_FORMULATIONS
    config = dict(formulation=args.formulation, cat_cardinalities=cardinalities,
                  n_numeric=n_numeric, d_model=args.d_model, n_head=args.n_head,
                  n_layer=args.n_layer, dropout=args.dropout, causal=args.causal,
                  pooling=args.pooling, use_positional=args.positional,
                  numeric_mode=args.numeric_mode, pos_weight=pos_weight,
                  cls_position=args.cls_position, cat_encoding=args.cat_encoding,
                  hash_buckets=args.hash_buckets,
                  ing_vocab_size=prep.ing_vocab_size if usa_ing else None,
                  max_ingredients=MAX_INGREDIENTS if usa_ing else 0, ing_layer=args.ing_layer,
                  ing_d_model=args.ing_d_model, ing_head=args.ing_head, n_cyclic=n_cyclic,
                  text_emb_dim=text_emb_dim, hf_model=args.text_emb_finetune)
    model = BTRTransformer(**config, bin_edges=bin_edges, cat_tables=cat_tables)
    return model, config


def run(csv_path, seed, args, device):
    """Una corrida completa: prepara datos, entrena, guarda resultados/pesos."""
    torch.manual_seed(seed if args.init_seed is None else args.init_seed)
    usa_ing = args.formulation in ING_FORMULATIONS
    if args.pretrain_mlm and (args.formulation != 'features' or args.cls_position != 'first'):
        raise SystemExit('--pretrain-mlm es solo para formulation features con CLS al inicio')
    if args.text_emb and args.text_emb_finetune:
        raise SystemExit('--text-emb y --text-emb-finetune: elegir UN regimen '
                         '(feature extraction congelado o fine-tuning)')
    if (args.text_emb or args.text_emb_finetune) and args.formulation != 'features':
        raise SystemExit('el titulo preentrenado es solo para formulation features '
                         '(las ing_* ya ocupan el slot del tercer tensor)')
    if args.pretrain_mlm and args.tiempo == 'ciclico':
        raise SystemExit('--pretrain-mlm no esta implementado con tokens ciclicos (--tiempo ciclico)')
    if (args.ing_d_model or args.ing_head) and args.formulation != 'ing_fusion':
        raise SystemExit('--ing-d-model / --ing-head son del encoder de conjunto (formulation ing_fusion)')
    prep, train_df, splits, indices = prepare(csv_path, seed=seed, train_frac=args.train_frac,
                                              cv_k=args.cv_k, cv_fold=args.cv_fold,
                                              use_ingredients=usa_ing, with_index=True,
                                              tiempo=args.tiempo)
    n_cyclic = len(prep.cyc)

    drop = {f.strip() for f in args.drop_features.split(',') if f.strip()}
    if drop == {'all'}:
        # sin features tabulares: el control "¿cuanto ve el titulo por si solo?"
        if not (args.text_emb or args.text_emb_finetune):
            raise SystemExit('--drop-features all requiere --text-emb/--text-emb-finetune')
        drop = set(prep.cats) | set(prep.nums)
    splits, keep_cat, keep_num = drop_feature_columns(splits, drop, prep.cats, prep.nums, n_cyclic)
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

    text_emb_dim = 0
    if args.text_emb or args.text_emb_finetune:
        # transfer learning desde un preentrenado EXTERNO: el TITULO del producto
        # (sin el badge de estado) como un token mas. Los tensores de transform()
        # preservan el orden del df, asi que los indices de fila de cada split
        # alinean la tabla externa con los tensores.
        nuevos = {}
        if args.text_emb:
            ruta = Path(args.text_emb)
            E = np.load(ruta if ruta.is_absolute() else REPO_ROOT / ruta).astype(np.float32)
            if E.shape[0] != len(load_dataset(csv_path)):
                raise SystemExit(f'--text-emb: {E.shape[0]} filas vs las del CSV '
                                 '(regenerar con eda/embed_titulos.py)')
            for kk, (a_, b_, c_, y_) in splits.items():
                e = torch.tensor(E[indices[kk]], device=a_.device)
                nuevos[kk] = (a_, b_, e, y_)  # el embedding ocupa el slot del tercer tensor
            text_emb_dim = E.shape[1]
            if not args.quiet:
                print(f"  text-emb: {ruta.name} ({E.shape[1]} dims congeladas)")
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.text_emb_finetune)
            if tok.pad_token_id != 0:
                raise SystemExit('--text-emb-finetune: el modelo debe usar pad_token_id=0 '
                                 '(la mascara del grafo asume pad=0)')
            df_all = load_dataset(csv_path)
            titulos = [titulo_sin_estado(t) for t in df_all['title']]
            for kk, (a_, b_, c_, y_) in splits.items():
                enc = tok([titulos[i] for i in indices[kk]], padding=True, truncation=True,
                          max_length=48, return_tensors='pt')
                nuevos[kk] = (a_, b_, enc['input_ids'].to(a_.device), y_)
            if not args.quiet:
                print(f"  text-emb-finetune: {args.text_emb_finetune} "
                      f"(lr encoder {args.text_emb_lr:g})")
        splits = nuevos

    model, model_config = build_model(args, prep, cardinalities, len(keep_num), bin_edges,
                                      pos_weight, cat_tables, text_emb_dim=text_emb_dim,
                                      n_cyclic=n_cyclic)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    name = run_name(args, seed)
    print(f"\n=== {name} | {n_params:,} parametros | device {device} ===")

    if args.pretrain_mlm:
        pretrain_mlm(model, splits['train'], cardinalities, epochs=args.pretrain_mlm,
                     batch_size=args.batch_size, lr=args.lr, device=device,
                     verbose=not args.quiet)

    history = train_model(model, splits, epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, patience=args.patience, verbose=not args.quiet,
                          weight_decay=args.weight_decay, text_emb_lr=args.text_emb_lr)
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

        if args.save_pesos and args.text_emb_finetune:
            print('  (checkpoint NO guardado: incluye el encoder preentrenado, que no se entrega)')
        elif args.save_pesos:
            pesos_dir = SALIDAS / 'pesos'
            pesos_dir.mkdir(parents=True, exist_ok=True)
            ckpt = pesos_dir / f'{out.stem}.pt'  # mismo nombre que el JSON de resultados
            torch.save({
                'arch': 'transformer',
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
    parser.add_argument('--formulation', choices=['features', 'ing', 'ing_fusion', 'ing_hybrid'],
                        default='features', help='que es un token (ver btr/model.py)')
    parser.add_argument('--ing-layer', type=int, default=1,
                        help='bloques del encoder de conjunto de ingredientes (ing_fusion)')
    parser.add_argument('--ing-d-model', type=int, default=None,
                        help='d_model propio del encoder de ingredientes (default: el del transformer); '
                             'su [ING] se proyecta a d_model')
    parser.add_argument('--ing-head', type=int, default=None,
                        help='cabezas del encoder de ingredientes (default: las del transformer)')
    parser.add_argument('--tiempo', default='none', choices=['none', 'ciclico', 'categorico'],
                        help='hora del dia y dia de la semana del timestamp: ciclico = un token (sin, cos) '
                             'por variable; categorico = dos categoricas mas (24 y 7 niveles)')
    parser.add_argument('--text-emb', default='', metavar='NPY',
                        help='transfer learning, feature extraction: matriz (N, E) del titulo '
                             'precomputada por eda/embed_titulos.py; entra como UN token extra')
    parser.add_argument('--text-emb-finetune', default='', metavar='HF_MODEL',
                        help='transfer learning, fine-tuning: el encoder de Hugging Face '
                             '(p. ej. sentence-transformers/all-MiniLM-L6-v2) entra al grafo')
    parser.add_argument('--text-emb-lr', type=float, default=1e-5,
                        help='lr del encoder preentrenado con --text-emb-finetune')
    parser.add_argument('--drop-features', default='',
                        help='features a excluir, separados por coma (ej: listing_status; '
                             '"all" con --text-emb: solo el titulo)')
    parser.add_argument('--cat-encoding', default='embedding',
                        choices=['embedding', 'target', 'freq', 'ordinal', 'hashing'],
                        help='encoding de las categoricas')
    parser.add_argument('--hash-buckets', type=int, default=8,
                        help='buckets del hashing trick (--cat-encoding hashing)')
    parser.add_argument('--cls-position', default='first', choices=['first', 'last'],
                        help='last: CLS al final (necesario para que --causal tenga sentido)')
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
