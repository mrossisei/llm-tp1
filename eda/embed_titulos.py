"""Precomputa embeddings del TITULO de cada producto con modelos PREENTRENADOS de texto
(transfer learning, clase 3: feature extraction).

    .venv/bin/python eda/embed_titulos.py            ->  salidas/embeddings/titulo_<modelo>.npy
    .venv/bin/python eda/embed_titulos.py minilm     ->  solo ese modelo

El titulo entra SIN el sufijo de estado "( ... )" (Best Seller, Top Rated, ...): lo que se
mide es si un modelo de lenguaje preentrenado ve en el nombre del producto algo que las 13
columnas no tengan, no si vuelve a leer el badge que ya extrae el regex.

Modelos (sentence-transformers; cada uno con SU pooling, como lo publica el autor):
  minilm  sentence-transformers/all-MiniLM-L6-v2   22M params, 384 dims  (el chico clasico)
  mpnet   sentence-transformers/all-mpnet-base-v2  110M params, 768 dims (el SBERT de referencia)
  bge     BAAI/bge-large-en-v1.5                   335M params, 1024 dims (el grande, MTEB)

Las matrices (float16, normalizadas L2) quedan alineadas 1:1 con las filas del CSV (orden de
pandas), que es lo que asume --text-emb en btr/train.py. Los modelos NO se entregan: se bajan
de Hugging Face al correr esto (CPU: minutos; GPU: segundos).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from btr.data import STATUS_SUFFIX_RE, load_dataset  # noqa: E402

MODELOS = {
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
    'bge': 'BAAI/bge-large-en-v1.5',
}
OUT = REPO / 'salidas' / 'embeddings'


def titulos_sin_estado(df):
    return [re.sub(STATUS_SUFFIX_RE, '', t).strip() for t in df['title']]


def main():
    from sentence_transformers import SentenceTransformer
    pedidos = sys.argv[1:] or list(MODELOS)
    df = load_dataset(REPO / 'supermarket_products.csv')
    textos = titulos_sin_estado(df)
    OUT.mkdir(parents=True, exist_ok=True)
    for nombre in pedidos:
        modelo = SentenceTransformer(MODELOS[nombre])
        e = modelo.encode(textos, batch_size=128, normalize_embeddings=True,
                          show_progress_bar=False, convert_to_numpy=True)
        np.save(OUT / f'titulo_{nombre}.npy', e.astype(np.float16))
        (OUT / f'titulo_{nombre}.json').write_text(json.dumps({
            'modelo': MODELOS[nombre], 'filas': int(e.shape[0]), 'dims': int(e.shape[1]),
            'parametros': int(sum(p.numel() for p in modelo.parameters())),
            'texto': 'title sin el sufijo de estado', 'ejemplo': textos[0]}, indent=1))
        print(f'escrito salidas/embeddings/titulo_{nombre}.npy {e.shape} '
              f'({MODELOS[nombre]}, {sum(p.numel() for p in modelo.parameters()) / 1e6:.0f}M params)')


if __name__ == '__main__':
    main()
