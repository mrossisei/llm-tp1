"""Entrenamiento y evaluacion de los modelos de BTR.

Protocolo de propuesta.md (seccion 7): split por query, AdamW, early stopping
por PR-AUC de validacion, metricas ROC-AUC / PR-AUC (desbalance 13%), promedio
de corridas con seeds distintas. Cada corrida escribe resultados/<nombre>.json
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import CAT_FEATURES, MAX_TEXT_LEN, NUM_FEATURES, prepare, prepare_listwise
from .model import BTRTransformer, ListwiseTransformer, MLPBaseline, TextTowerModel

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_BATCH = 1024        # con secuencias largas no entra todo el split en un forward
TRAIN_EVAL_ROWS = 4000   # submuestra fija de train para las metricas por epoca


def evaluate(model, split, max_rows=None):
    """(loss, roc_auc, pr_auc) del split, en eval mode y por lotes.

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
        if a.dim() == 3:  # listwise: c es la mascara de productos reales
            logits, y = logits[c], y[c]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=model.pos_weight
        ).item()
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = y.cpu().numpy()
    return loss, roc_auc_score(y_true, probs), average_precision_score(y_true, probs)


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

        train_loss, train_roc, train_pr = evaluate(model, splits['train'], max_rows=TRAIN_EVAL_ROWS)
        val_loss, val_roc, val_pr = evaluate(model, splits['val'])
        history.append({'epoch': epoch,
                        'train_loss': train_loss, 'train_roc_auc': train_roc, 'train_pr_auc': train_pr,
                        'val_loss': val_loss, 'val_roc_auc': val_roc, 'val_pr_auc': val_pr})
        if verbose:
            print(f"epoch {epoch:3d} | loss train {train_loss:.4f} val {val_loss:.4f} | "
                  f"PR-AUC train {train_pr:.4f} val {val_pr:.4f} | ROC-AUC val {val_roc:.4f}")

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
    parts = [base, f"d{args.d_model}", f"h{args.n_head}", f"l{args.n_layer}", args.numeric_mode]
    if args.strip_status:
        parts.append('stripstatus')
    if args.drop_features:
        parts.append('sin-' + args.drop_features.replace(',', '-').replace('_', ''))
    if args.pooling != 'cls':
        parts.append(args.pooling)
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


def drop_feature_columns(splits, drop, listwise=False):
    """Saca features por nombre (p. ej. listing_status) recortando columnas de los tensores."""
    keep_cat = [i for i, f in enumerate(CAT_FEATURES) if f not in drop]
    keep_num = [i for i, f in enumerate(NUM_FEATURES) if f not in drop]
    unknown = drop - set(CAT_FEATURES) - set(NUM_FEATURES)
    if unknown:
        raise SystemExit(f'--drop-features desconocidos: {sorted(unknown)}')
    dim = 2 if listwise else 1  # en listwise los features son la 3ra dimension
    splits = {k: (v[0].index_select(dim, torch.tensor(keep_cat)),
                  v[1].index_select(dim, torch.tensor(keep_num)),
                  v[2], v[3]) for k, v in splits.items()}
    return splits, keep_cat, keep_num


def build_model(args, prep, cardinalities, n_numeric, bin_edges, pos_weight, max_products=None):
    """Configura y construye la arquitectura pedida; devuelve (modelo, config para el ckpt)."""
    common = dict(d_model=args.d_model, dropout=args.dropout,
                  numeric_mode=args.numeric_mode, pos_weight=pos_weight)
    if args.arch == 'transformer':
        config = dict(formulation=args.formulation, cat_cardinalities=cardinalities,
                      n_numeric=n_numeric, char_vocab_size=prep.char_vocab_size,
                      max_text_len=prep.max_text_len, n_head=args.n_head,
                      n_layer=args.n_layer, causal=args.causal, pooling=args.pooling,
                      use_positional=args.positional, **common)
        model = BTRTransformer(**config, bin_edges=bin_edges)
    elif args.arch == 'mlp':
        config = dict(cat_cardinalities=cardinalities, n_numeric=n_numeric, **common)
        model = MLPBaseline(**config, bin_edges=bin_edges)
    elif args.arch == 'tower':
        config = dict(cat_cardinalities=cardinalities, n_numeric=n_numeric,
                      char_vocab_size=prep.char_vocab_size, max_text_len=prep.max_text_len,
                      n_head=args.n_head, n_layer=args.n_layer, causal=args.causal, **common)
        model = TextTowerModel(**config, bin_edges=bin_edges)
    elif args.arch == 'listwise':
        config = dict(cat_cardinalities=cardinalities, n_numeric=n_numeric,
                      max_products=max_products, n_head=args.n_head,
                      n_layer=args.n_layer, **common)
        model = ListwiseTransformer(**config, bin_edges=bin_edges)
    else:
        raise SystemExit(f'arquitectura desconocida: {args.arch}')
    return model, config


def run(csv_path, seed, args, device):
    """Una corrida completa: prepara datos, entrena, guarda resultados/pesos."""
    torch.manual_seed(seed)
    listwise = args.arch == 'listwise'
    if listwise:
        prep, max_products, splits = prepare_listwise(csv_path, seed=seed)
    else:
        prep, train_df, splits = prepare(csv_path, seed=seed, max_text_len=args.max_text_len,
                                         strip_status=args.strip_status)
        max_products = None

    drop = {f.strip() for f in args.drop_features.split(',') if f.strip()}
    splits, keep_cat, keep_num = drop_feature_columns(splits, drop, listwise)
    splits = {k: tuple(t.to(device) for t in v) for k, v in splits.items()}

    bin_edges = None
    if args.numeric_mode == 'bins':
        if listwise:
            raise SystemExit('numeric-mode bins no implementado para listwise')
        bin_edges = prep.bin_edges(train_df, args.n_bins)[keep_num]
    pos_weight = None
    if args.pos_weight:
        _, _, m_or_t, y_train = splits['train']
        y_flat = y_train[m_or_t] if listwise else y_train
        pos_weight = ((1 - y_flat.mean()) / y_flat.mean()).item()  # negativos/positivos

    cardinalities = [c for i, c in enumerate(prep.cat_cardinalities) if i in keep_cat]
    model, model_config = build_model(args, prep, cardinalities, len(keep_num),
                                      bin_edges, pos_weight, max_products)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    name = run_name(args, seed)
    print(f"\n=== {name} | {n_params:,} parametros | device {device} ===")

    history = train_model(model, splits, epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, patience=args.patience, verbose=not args.quiet)
    val_loss, val_roc, val_pr = evaluate(model, splits['val'])
    test_loss, test_roc, test_pr = evaluate(model, splits['test'])
    print(f"{name} -> VAL: ROC-AUC {val_roc:.4f} PR-AUC {val_pr:.4f} | "
          f"TEST: loss {test_loss:.4f} ROC-AUC {test_roc:.4f} PR-AUC {test_pr:.4f}")

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
            'val': {'loss': val_loss, 'roc_auc': val_roc, 'pr_auc': val_pr},
            'test': {'loss': test_loss, 'roc_auc': test_roc, 'pr_auc': test_pr},
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
                'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'preprocessor': prep,
                'cat_features': [CAT_FEATURES[i] for i in keep_cat],
                'num_features': [NUM_FEATURES[i] for i in keep_num],
            }, ckpt)
            print(f"pesos      -> {ckpt.relative_to(REPO_ROOT)}")

    return test_roc, test_pr


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
