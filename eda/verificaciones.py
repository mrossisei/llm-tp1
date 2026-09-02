"""Verificaciones del EDA: reproduce todos los numeros citados en propuesta.md.

Correr con el venv del repo:
    .venv/bin/python eda/verificaciones.py

Cada seccion referencia la seccion de propuesta.md donde se usa el numero.
(El notebook prolijo del Ejercicio 1 va a salir de aca; este script es la fuente
reproducible de cada afirmacion del documento.)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from btr.data import load_dataset, split_by_query  # noqa: E402

TOP = ['#1 Pick', 'Top Rated', 'Best Seller', 'Customer Favorite']
MID = ['Popular Choice', 'Highly Rated', 'Shopper Favorite', 'Well Reviewed']


def seccion(titulo):
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")


def main():
    df_raw = pd.read_csv(REPO_ROOT / 'supermarket_products.csv')
    df = load_dataset(REPO_ROOT / 'supermarket_products.csv')  # + listing_status, price_rel

    seccion("2.1 Estructura de eventos")
    print(f"filas: {len(df)} | queries: {df['query_id'].nunique()} | titulos unicos: {df['title'].nunique()}")
    sizes = df.groupby('query_id').size()
    print(f"productos por query: min {sizes.min()} mediana {sizes.median():.0f} max {sizes.max()}")
    print(f"bought: {df['bought'].mean():.4f} | cart: {df['cart'].mean():.4f}")
    print("compras por query:", df.groupby('query_id')['bought'].sum().value_counts().sort_index().to_dict())
    fcols = ['filter_category', 'filter_price_min', 'filter_price_max', 'filter_storage_type']
    varia = {c: int((df.groupby('query_id')[c].nunique() > 1).sum()) for c in fcols}
    print("queries donde el filtro varia adentro (debe ser 0):", varia)
    print("producto cumple su filtro:",
          f"categoria {(df['category'] == df['filter_category']).mean():.2f}",
          f"storage {(df['storage_type'] == df['filter_storage_type']).mean():.2f}",
          f"precio en rango {((df['price'] >= df['filter_price_min']) & (df['price'] <= df['filter_price_max'])).mean():.2f}")

    seccion("2.2 cart es parte del target (leakage)")
    print("crosstab cart x bought (bought=True con cart=False debe ser 0):")
    print(pd.crosstab(df['cart'], df['bought']))

    seccion("2.3 La senal dominante esta en el texto")
    print("cobertura del sufijo:", f"{(df['listing_status'] != 'None').mean():.4f}")
    tab = df.groupby('listing_status')['bought'].agg(['mean', 'count']).round(3).sort_values('mean')
    print(tab)
    last = df['description'].str.split('. ').str[-1].str.rstrip('.')
    print("\nBTR por frase final de la descripcion (redundante con el sufijo):")
    print(df.assign(l=last).groupby('l')['bought'].agg(['mean', 'count']).round(3).sort_values('mean'))

    seccion("2.4 Efectos secundarios condicionados al tier alto")
    top = df[df['listing_status'].isin(TOP)]
    print("U invertida - p(bought|top) por posicion del precio en el rango del filtro:")
    print(top.groupby(pd.cut(top['price_rel'], [0, .2, .4, .6, .8, 1.0]), observed=True)['bought']
          .agg(['mean', 'count']).round(3))
    print("\np(bought|top) por alergeno:")
    print(top.groupby(top['allergens'])['bought'].agg(['mean', 'count']).round(3).sort_values('mean'))
    print("\np(bought|top) por categoria (extremos):")
    porcat = top.groupby('category')['bought'].mean().sort_values()
    print(porcat.head(2).round(3).to_dict(), "...", porcat.tail(2).round(3).to_dict())
    print("nutrition_score (cuartiles, todas las filas):",
          df.groupby(pd.qcut(df['nutrition_score'], 4), observed=True)['bought'].mean().round(3).tolist())

    seccion("2.5 Interaccion entre productos de la misma pagina (debil)")
    df['n_top'] = df.groupby('query_id')['listing_status'].transform(lambda s: s.isin(TOP).sum())
    print("p(bought | soy top, k tops en mi query):")
    print(df[df['listing_status'].isin(TOP)].groupby('n_top')['bought'].agg(['mean', 'count']).round(3))

    seccion("2.6 Calidad de datos y redundancias")
    ts = pd.to_datetime(df['timestamp'])
    span_dias = (df.assign(ts=ts).groupby('query_id')['ts'].agg(lambda s: s.max() - s.min())
                 .dt.total_seconds() / 86400)
    print(f"span de timestamps dentro de una query (dias): mediana {span_dias.median():.0f}, max {span_dias.max():.0f}")
    print("BTR por anio (deberia ser ~constante):",
          df.groupby(ts.dt.year)['bought'].mean().round(3).to_dict())
    oz = df[df['unit_of_measure'] == 'oz']
    pkg = oz['package_size'].str.extract(r'([\d.]+)').astype(float)[0]
    print(f"corr net_weight_oz vs numero de package_size: {np.corrcoef(pkg, oz['net_weight_oz'])[0, 1]:.3f}")
    print(f"nulos en allergens: {df_raw['allergens'].isna().mean():.4f} (unico campo con nulos)")

    seccion("2.6b ¿query_id es un template re-ejecutado? (chequeo de productos repetidos)")
    # Si query_id fuera un template (combinacion exacta de filtros) ejecutado muchas veces a lo
    # largo del tiempo, el MISMO producto deberia reaparecer entre ejecuciones. No hay product_id:
    # identidad = las 14 columnas catalograficas (todo menos filtros, eventos y timestamp).
    catalogo = [c for c in df_raw.columns if c not in
                ('query_id', 'timestamp', 'cart', 'bought', 'filter_category',
                 'filter_price_min', 'filter_price_max', 'filter_storage_type')]
    d = df_raw.assign(pid=df_raw[catalogo].fillna('NULO').astype(str).agg('|'.join, axis=1))
    print(f"identidad de producto = {len(catalogo)} columnas catalograficas (no hay product_id)")
    print(f"productos unicos: {d['pid'].nunique()} en {len(d)} filas -> "
          f"repetidos en TODO el dataset: {len(d) - d['pid'].nunique()}")
    rep_p = d.groupby('query_id')['pid'].apply(lambda s: s.duplicated().sum())
    rep_t = d.groupby('query_id')['title'].apply(lambda s: s.duplicated().sum())
    print(f"queries con producto repetido adentro: {int((rep_p > 0).sum())} de {rep_p.size}")
    print(f"queries con TITULO repetido adentro: {int((rep_t > 0).sum())} "
          "(colision del generador de nombres: precio/peso/dimensiones distintos)")
    fcols = ['filter_category', 'filter_price_min', 'filter_price_max', 'filter_storage_type']
    uq = d.drop_duplicates('query_id')
    combo = uq[fcols].astype(str).agg('|'.join, axis=1)
    print(f"combos EXACTOS de filtros distintos: {combo.nunique()} en {len(uq)} queries "
          f"(el filtro de precio esta sobre una grilla: {uq['filter_price_min'].nunique()} minimos distintos)")
    por_combo = uq.groupby(combo.values)['query_id'].apply(list)
    repetidos = por_combo[por_combo.apply(len) > 1]
    comp = 0
    for qs in repetidos:
        sets = [set(d.loc[d['query_id'] == q, 'pid']) for q in qs]
        comp += sum(len(sets[i] & sets[j])
                    for i in range(len(qs)) for j in range(i + 1, len(qs)))
    print(f"queries que comparten combo exacto con otra: {int(repetidos.apply(len).sum())} "
          f"({len(repetidos)} combos) -> productos compartidos entre ellas: {comp}")
    # ultimo resquicio: titulos BASE repetidos (sin sufijo) ¿son el mismo producto fisico
    # re-observado con otro badge/precio? Peso y dimensiones no cambiarian -> comparar.
    b = d.assign(base=d['title'].str.replace(r'\s*\([^)]+\)$', '', regex=True))
    rep_b = b.groupby('base').filter(lambda x: len(x) > 1).groupby('base')
    fisico = rep_b.apply(lambda x: x['net_weight_oz'].nunique() == 1
                         and x['dimensions_in'].nunique() == 1, include_groups=False)
    print(f"titulos base repetidos: {len(fisico)} -> con peso Y dimensiones identicos "
          f"entre apariciones: {int(fisico.sum())} (todos colisiones de nombre, no re-observaciones)")
    print("veredicto: ningun producto reaparece jamas — ni dentro de una query ni entre queries")
    print("con identico filtro (el mejor escenario del template). No existe catalogo persistente:")
    print("query_id NO es un template re-ejecutado -> los spans intra-query de ~2 anios no tienen")
    print("lectura fisica posible -> timestamp = ruido del generador (cierra la dicotomia de 2.6)")

    seccion("7.1 Overlap de productos entre splits (split por query, seed 42)")
    tr, va, te = split_by_query(df, seed=42)
    base = lambda s: s.str.replace(r'\s*\([^)]+\)$', '', regex=True)
    print(f"test con titulo visto en train: {te['title'].isin(set(tr['title'])).mean():.4f}")
    print(f"test con producto base visto en train: {base(te['title']).isin(set(base(tr['title']))).mean():.4f}")

    seccion("7.3 / 2.3.1 Baselines (split por query, seed 42)")
    baselines(df)


def baselines(df):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat = ['listing_status', 'category', 'brand', 'storage_type',
           'unit_of_measure', 'country_of_origin', 'allergens']
    num = ['price', 'price_rel', 'filter_price_min', 'filter_price_max',
           'net_weight_oz', 'nutrition_score']
    tr, _, te = split_by_query(df, seed=42)
    y_tr, y_te = tr['bought'].astype(int), te['bought'].astype(int)
    print(f"prevalencia test (PR-AUC de azar): {y_te.mean():.4f}")

    def reporte(nombre, modelo, cols_cat, X_tr, X_te):
        p = modelo.predict_proba(X_te)[:, 1]
        print(f"{nombre:45s} ROC-AUC={roc_auc_score(y_te, p):.4f}  PR-AUC={average_precision_score(y_te, p):.4f}")

    for nombre, cc in [('logistica SOLO listing_status', ['listing_status']),
                       ('logistica todo (one-hot + z-score)', cat),
                       ('logistica SIN listing_status', [c for c in cat if c != 'listing_status'])]:
        nn = num if len(cc) > 1 else []
        pre = ColumnTransformer(
            [('c', OneHotEncoder(handle_unknown='ignore'), cc)]
            + ([('n', StandardScaler(), nn)] if nn else []))
        m = Pipeline([('p', pre), ('lr', LogisticRegression(max_iter=2000))]).fit(tr[cc + nn], y_tr)
        reporte(nombre, m, cc, tr[cc + nn], te[cc + nn])

    low = ~df['listing_status'].isin(TOP + MID)
    print(f"filas en tiers de BTR=0 (sin senal de popularidad): {low.mean():.2%}, BTR real: {df.loc[low, 'bought'].mean():.4f}")


if __name__ == '__main__':
    main()
