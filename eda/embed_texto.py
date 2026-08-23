"""Precomputa embeddings de texto con un modelo PREENTRENADO (transfer learning, clase 3).

    .venv/bin/python eda/embed_texto.py   ->  embeddings/minilm.npy  (10000, 384)
                                              embeddings/minilm_intr.npy

Modelo: sentence-transformers/all-MiniLM-L6-v2 (22M params, ingles) via
transformers. Receta identica a sentence-transformers: mean pooling sobre
last_hidden_state con la attention mask + normalizacion L2. La variante _intr
embebe el texto SIN el sufijo de estado del titulo ni la ultima oracion de la
descripcion (strip_status_from_text): mide si el preentrenado ve algo en el
texto MAS ALLA del status.

Las matrices quedan alineadas 1:1 con las filas del CSV (el orden de pandas),
que es lo que asume --text-emb en btr/train.py. Corre en CPU en unos minutos;
en la 3070, segundos.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from btr.data import load_dataset, strip_status_from_text  # noqa: E402

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH, MAX_LEN = 128, 128


def embed(textos, tokenizer, model, device):
    filas = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(textos), BATCH):
            enc = tokenizer(textos[i:i + BATCH], padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors='pt').to(device)
            h = model(**enc).last_hidden_state              # (B, L, H)
            m = enc['attention_mask'].unsqueeze(-1).float()  # (B, L, 1)
            pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)
            filas.append(torch.nn.functional.normalize(pooled, dim=1).cpu())
            print(f'\r  {min(i + BATCH, len(textos))}/{len(textos)}', end='', flush=True)
    print()
    return torch.cat(filas).numpy().astype(np.float32)


def main():
    from transformers import AutoModel, AutoTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device)
    df = load_dataset(REPO / 'supermarket_products.csv')

    out = REPO / 'embeddings'
    out.mkdir(exist_ok=True)
    completo = [t + '\n' + d for t, d in zip(df['title'], df['description'])]
    intr = ['\n'.join(strip_status_from_text(t, d))
            for t, d in zip(df['title'], df['description'])]
    for nombre, textos in [('minilm', completo), ('minilm_intr', intr)]:
        e = embed(textos, tokenizer, model, device)
        np.save(out / f'{nombre}.npy', e)
        print(f'escrito embeddings/{nombre}.npy {e.shape} (device {device})')


if __name__ == '__main__':
    main()
