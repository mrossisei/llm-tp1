"""Figura de la 8va tanda: transfer desde un preentrenado externo (MiniLM).

    .venv/bin/python eda/grafico_bert.py   ->  graficos/bert_transfer.png

Dot plot (media +- desvio, 6 seeds) de las 7 configs bert_* contra las
referencias del proyecto. La historia: la particion de tiers es anti-semantica
-> el embedding CONGELADO resta en todos lados; el FINE-TUNING la repara
(6/6 en ambos pares) y sin el regex lee el status desde el texto mejor que
nuestros encoders entrenados de cero — pero nada alcanza al campeon.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
TEAL, VIOLETA = '#0E9B7E', '#7052C9'   # fine-tuning / congelado (par validado CVD)

FILAS = [  # (etiqueta, tag, color)  — de abajo hacia arriba
    ('solo el embedding, texto sin status\n(bert_solo_intr)',
     'bert_solo_intr_mlp_d32_h4_l2_linear_temb-minilmintr_sin-all', VIOLETA),
    ('solo el embedding congelado\n(bert_solo)',
     'bert_solo_mlp_d32_h4_l2_linear_temb-minilm_sin-all', VIOLETA),
    ('campeón sin regex + token congelado\n(bert_token_sin)',
     'bert_token_sin_features_d32_h4_l2_linear_catordinal_temb-minilm_sin-listingstatus',
     VIOLETA),
    ('campeón + token congelado\n(bert_token)',
     'bert_token_features_d32_h4_l2_linear_catordinal_temb-minilm', VIOLETA),
    ('MLP + 384 numéricas congeladas\n(bert_mlp)',
     'bert_mlp_mlp_d32_h4_l2_linear_catordinal_temb-minilm', VIOLETA),
    ('campeón sin regex + encoder FINE-TUNED\n(bert_ft_sin)',
     'bert_ft_sin_features_d32_h4_l2_linear_catordinal_tembft_sin-listingstatus', TEAL),
    ('campeón + encoder FINE-TUNED\n(bert_ft)',
     'bert_ft_features_d32_h4_l2_linear_catordinal_tembft', TEAL),
]
REFS = [  # (valor, etiqueta) — el contexto medido del proyecto
    (0.131, 'azar 0.131'),
    (0.162, 'techo sin estado 0.162'),
    (0.660, 'logística 0.660'),
    (0.775, 'tower (encoder propio) 0.775'),
    (0.824, 'campeón 0.824'),
]


def grupo(tag):
    prs = [json.loads((REPO / 'resultados' / f'{tag}_seed{s}.json').read_text())
           ['test']['pr_auc'] for s in range(42, 48)]
    m = sum(prs) / len(prs)
    return m, (sum((x - m) ** 2 for x in prs) / len(prs)) ** 0.5


def main():
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for v, et in REFS:
        ax.axvline(v, color='#BBBBBB', lw=1, ls='--', zorder=1)
        ax.text(v, len(FILAS) - 0.25, et, rotation=90, fontsize=7.5,
                color='#777777', ha='right', va='top')
    for i, (et, tag, color) in enumerate(FILAS):
        m, sd = grupo(tag)
        ax.errorbar(m, i, xerr=sd, color=color, marker='o', ms=8, capsize=3,
                    lw=2, zorder=3)
        ax.annotate(f'{m:.3f}', (m, i), textcoords='offset points', xytext=(0, 9),
                    fontsize=9, ha='center', color='#444444')
    ax.set_yticks(range(len(FILAS)), [f[0] for f in FILAS], fontsize=8.5)
    ax.set_xlabel('test PR-AUC (media ± desvío, 6 seeds)')
    ax.set_xlim(0.05, 0.92)
    ax.set_ylim(-0.6, len(FILAS) - 0.4)
    ax.set_title('Preentrenado externo (MiniLM): congelado RESTA, fine-tuneado repara\n'
                 '(la partición de tiers es anti-semántica: hay que ajustar el encoder)',
                 fontsize=11)
    ax.grid(True, axis='x', alpha=0.25, lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=VIOLETA, marker='o', lw=2, label='embedding congelado (feature extraction)'),
        Line2D([], [], color=TEAL, marker='o', lw=2, label='encoder fine-tuneado (lr 1e-5)'),
    ], loc='lower right', fontsize=8.5, frameon=False)
    fig.tight_layout()
    out = REPO / 'graficos' / 'bert_transfer.png'
    fig.savefig(out, dpi=140)
    print(f'escrito {out.relative_to(REPO)}')


if __name__ == '__main__':
    main()
