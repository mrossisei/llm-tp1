"""Carga y preprocesamiento — SOLO lo que usa el modelo final.

El modelo final trabaja con 13 features tabulares por impresion (fila del CSV):

  7 categoricas  -> encoding ORDINAL: cada nivel se reemplaza por su rango al
                    ordenar los niveles por la media suavizada de `bought` en
                    TRAIN, normalizado a [0, 1]. Es un prior de orden derivado
                    de los datos (no un orden semantico a mano: el EDA mostro
                    que el wording no predice la propension — "Highly Rated"
                    suena igual que "Top Rated" y compra 50 veces menos).
  6 numericas    -> z-score con estadisticos de TRAIN (log1p previo en las dos
                    sesgadas a derecha: price y net_weight_oz).

Features derivados en la carga (sin mirar el target):
  listing_status <- sufijo "( ... )" al final del titulo (la senal dominante
                    del dataset segun el EDA; 21 niveles incluyendo 'None').
  price_rel      <- posicion del precio dentro del rango filtrado por el
                    usuario: (price - filter_min) / (filter_max - filter_min).
                    Es la senal RELACIONAL producto-busqueda (efecto en U
                    invertida, condicionado al tier de estado).

Exclusiones (justificadas en el EDA y verificadas por ablacion):
  cart (leakage estricto: bought => cart en el 100% de las filas), query_id
  (solo particiona), timestamp (roto: spans de 2 anos dentro de una misma
  busqueda), package_size/dimensions_in/ingredients (redundantes; reintroducirlos
  midio delta ~0), filter_category/filter_storage_type (los productos impresos
  siempre los cumplen; el rango de precio si entra, via price_rel).

Particion por query_id (group split) 70/15/15: las filas de una misma busqueda
nunca se reparten entre train/val/test (un split aleatorio por fila filtraria
informacion de la pagina). Vocabularios, estadisticos y tablas ordinales se
ajustan SOLO con train; el indice 0 de cada vocabulario es UNK.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

CAT_FEATURES = [
    'listing_status', 'category', 'brand', 'storage_type',
    'unit_of_measure', 'country_of_origin', 'allergens',
]
NUM_FEATURES = [
    'price', 'price_rel', 'filter_price_min', 'filter_price_max',
    'net_weight_oz', 'nutrition_score',
]
LOG_FEATURES = ['price', 'net_weight_oz']  # sesgadas a derecha -> log1p antes del z-score
TARGET = 'bought'
SUAVIZADO_M = 50.0  # m-estimate del encoding ordinal (amortigua niveles chicos)


def load_dataset(csv_path):
    """Lee el CSV y deriva listing_status y price_rel (sin mirar el target)."""
    df = pd.read_csv(csv_path)
    df['listing_status'] = df['title'].str.extract(r'\(([^)]+)\)$')[0].fillna('None')
    df['allergens'] = df['allergens'].fillna('None')  # "sin alergenos declarados"
    span = df['filter_price_max'] - df['filter_price_min']
    df['price_rel'] = (df['price'] - df['filter_price_min']) / span
    return df


def split_by_query(df, seed=42, val_frac=0.15, test_frac=0.15):
    """Particiona por query_id para que una misma busqueda no cruce splits."""
    rng = np.random.default_rng(seed)
    queries = df['query_id'].unique().to_numpy()
    rng.shuffle(queries)
    n_test = int(len(queries) * test_frac)
    n_val = int(len(queries) * val_frac)
    test_q = set(queries[:n_test])
    val_q = set(queries[n_test:n_test + n_val])
    is_test = df['query_id'].isin(test_q)
    is_val = df['query_id'].isin(val_q)
    return df[~is_test & ~is_val], df[is_val], df[is_test]


@dataclass
class Preprocessor:
    """Vocabularios + estadisticos + tablas ordinales, ajustados SOLO con train."""
    vocabs: dict                      # feature -> {valor: indice}, 0 reservado a UNK
    num_mean: np.ndarray              # (6,)
    num_std: np.ndarray               # (6,)
    cat_tables: list = field(default_factory=list)  # por feature: tensor (card+1,) en [0,1]

    @classmethod
    def fit(cls, train_df):
        vocabs = {
            f: {v: i + 1 for i, v in enumerate(sorted(train_df[f].unique()))}
            for f in CAT_FEATURES
        }
        num = cls._numeric_raw(train_df)
        prep = cls(vocabs=vocabs, num_mean=num.mean(axis=0), num_std=num.std(axis=0) + 1e-8)
        prep.cat_tables = [cls._tabla_ordinal(train_df, f, vocabs[f]) for f in CAT_FEATURES]
        return prep

    @staticmethod
    def _tabla_ordinal(train_df, feature, vocab):
        """nivel -> rango en [0,1] al ordenar por la media suavizada de bought (train)."""
        global_mean = float(train_df[TARGET].mean())
        grp = train_df.groupby(feature)[TARGET].agg(['sum', 'count'])
        suavizada = {}
        for valor, idx in vocab.items():
            n = float(grp.loc[valor, 'count'])
            suavizada[idx] = (float(grp.loc[valor, 'sum']) + SUAVIZADO_M * global_mean) \
                / (n + SUAVIZADO_M)
        tabla = torch.full((len(vocab) + 1,), 0.5)  # UNK -> rango medio
        orden = sorted(suavizada, key=suavizada.get)
        k = max(len(orden) - 1, 1)
        for rango, idx in enumerate(orden):
            tabla[idx] = rango / k
        return tabla

    @staticmethod
    def _numeric_raw(df):
        num = df[NUM_FEATURES].to_numpy(dtype=np.float64).copy()
        for f in LOG_FEATURES:
            j = NUM_FEATURES.index(f)
            num[:, j] = np.log1p(num[:, j])
        return num

    def transform(self, df):
        """df -> (x_cat long, x_num float32, y float32) listos para el modelo."""
        x_cat = np.stack(
            [df[f].map(lambda v, f=f: self.vocabs[f].get(v, 0)).to_numpy()
             for f in CAT_FEATURES], axis=1)
        x_num = (self._numeric_raw(df) - self.num_mean) / self.num_std
        y = df[TARGET].to_numpy()
        return (torch.tensor(x_cat, dtype=torch.long),
                torch.tensor(x_num, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))


def prepare(csv_path, seed=42):
    """Pipeline completo: carga -> split por query -> fit en train -> tensores."""
    df = load_dataset(csv_path)
    train_df, val_df, test_df = split_by_query(df, seed=seed)
    prep = Preprocessor.fit(train_df)
    return prep, {
        'train': prep.transform(train_df),
        'val': prep.transform(val_df),
        'test': prep.transform(test_df),
    }
