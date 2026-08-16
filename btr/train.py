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

Ejes transversales: --drop-features listing_status (modelo sin el estado
parseado), --strip-status (texto sin sufijo ni oracion de estado: la variante
"producto nuevo"), --numeric-mode bins, --positional, --causal, --pooling,
--pos-weight. La suite curada de experimentos esta en experimentos.py.

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

from .data import (CAT_FEATURES, EXTRA_FEATURES, MAX_TEXT_LEN, NUM_FEATURES, TARGET,
                   prepare, prepare_listwise)
from .model import BTRTransformer, ListwiseTransformer, MLPBaseline, TextTowerModel

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


def train_model(model, splits, epochs=60, batch_size=256, lr=1e-3, patience=8, verbose=True):
    """Entrena con early stopping por PR-AUC de validacion; restaura el mejor estado."""
    a, b, c, y = splits['train']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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


def resolve_device(arg):
    if arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return arg


def run_name(args, seed):
    base = args.formulation if args.arch == 'transformer' else args.arch
    if args.arch == 'listwise' and args.listwise_texto:
        base = 'listwisetexto'
    parts = [base, f"d{args.d_model}", f"h{args.n_head}", f"l{args.n_layer}", args.numeric_mode]
    if args.cat_encoding != 'embedding':
        parts.append(f"cat{args.cat_encoding}"
                     + (str(args.hash_buckets) if args.cat_encoding == 'hashing' else ''))
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
    splits = {k: (v[0].index_select(dim, torch.tensor(keep_cat)),
                  v[1].index_select(dim, torch.tensor(keep_num)),
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
                max_products=None, cat_tables=None, cat_modes=None):
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
                      cart_lambda=args.cart_aux, **encod, **common)
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
    else:
        raise SystemExit(f'arquitectura desconocida: {args.arch}')
    return model, config


def run(csv_path, seed, args, device):
    """Una corrida completa: prepara datos, entrena, guarda resultados/pesos."""
    torch.manual_seed(seed)
    listwise = args.arch == 'listwise'
    if args.cat_encoding == 'onehot' and args.arch != 'mlp':
        raise SystemExit('cat-encoding onehot es solo para --arch mlp: para el transformer, '
                         'one-hot + proyeccion lineal aprende la misma matriz que el '
                         'embedding (propuesta 6.1) — no hay experimento que correr')
    if args.cat_encoding != 'embedding' and listwise:
        raise SystemExit('cat-encoding alternativo no implementado para listwise')
    if args.cat_feature_encoding and listwise:
        raise SystemExit('cat-feature-encoding no implementado para listwise')
    if args.cat_feature_encoding and args.cat_encoding == 'onehot':
        raise SystemExit('cat-feature-encoding no se combina con onehot (que es global del MLP)')
    extras = tuple(f.strip() for f in args.extra_features.split(',') if f.strip())
    if extras == ('all',):
        extras = tuple(sorted(EXTRA_FEATURES))
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
                                         extra_features=extras, include_cart=args.cart_aux > 0)
        max_products = None

    drop = {f.strip() for f in args.drop_features.split(',') if f.strip()}
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
    model, model_config = build_model(args, prep, cardinalities, len(keep_num),
                                      bin_edges, pos_weight, max_products, cat_tables,
                                      cat_modes)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    name = run_name(args, seed)
    print(f"\n=== {name} | {n_params:,} parametros | device {device} ===")

    history = train_model(model, splits, epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, patience=args.patience, verbose=not args.quiet)
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

        if args.save_pesos:
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
    parser.add_argument('--arch', choices=['transformer', 'mlp', 'tower', 'listwise'],
                        default='transformer')
    parser.add_argument('--formulation', choices=['features', 'text', 'hybrid'], default='features',
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
    parser.add_argument('--d-model', type=int, default=32)
    parser.add_argument('--n-head', type=int, default=4)
    parser.add_argument('--n-layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
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
