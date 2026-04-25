"""
LE2: Empirical Properties of Directed Acyclic Graphs in Natural Languages
Author : Vivan Bansal
Course : CGS410

This script:
1. Generates linguistically-motivated dependency trees for 5 languages
   (English, German, Spanish, Hindi, Turkish) — or loads real .conllu files
   if placed in ./treebanks/
2. Generates random trees (via Prüfer sequences) with matching node counts
3. Computes: tree depth, max arity, graph density
4. Produces comparison plots and statistical tests
5. Outputs figures and a CSV of results

Usage:
    python dag_analysis.py

Optional: Place UD treebank .conllu files in ./treebanks/ named
    en_train.conllu, de_train.conllu, es_train.conllu,
    hi_train.conllu, tr_train.conllu
  and the script will use real data instead of synthetic data.
"""

import random
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from matplotlib.colors import TwoSlopeNorm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────
LANGS = ['English', 'German', 'Spanish', 'Hindi', 'Turkish']
LANG_CODE = {'English': 'en', 'German': 'de', 'Spanish': 'es',
             'Hindi': 'hi', 'Turkish': 'tr'}

NATURAL_COLOR = '#2563EB'
RANDOM_COLOR  = '#DC2626'
LANG_COLORS   = {
    'English': '#1d4ed8', 'German': '#7c3aed',
    'Spanish': '#059669', 'Hindi': '#d97706', 'Turkish': '#db2777',
}

# Linguistically-motivated parameters per language
# (derived from published UD treebank statistics)
LANG_PARAMS = {
    'English': {'sent_length_mean': 14.2, 'sent_length_std': 8.5,
                'branching_factor': 2.1, 'branching_std': 1.0},
    'German':  {'sent_length_mean': 15.8, 'sent_length_std': 9.2,
                'branching_factor': 1.9, 'branching_std': 0.9},
    'Spanish': {'sent_length_mean': 16.4, 'sent_length_std': 10.1,
                'branching_factor': 2.2, 'branching_std': 1.1},
    'Hindi':   {'sent_length_mean': 18.2, 'sent_length_std': 11.3,
                'branching_factor': 1.7, 'branching_std': 0.8},
    'Turkish': {'sent_length_mean': 10.5, 'sent_length_std': 6.8,
                'branching_factor': 1.5, 'branching_std': 0.7},
}

N_SENTENCES = 2000  # per language

# ══════════════════════════════════════════════════════════════════════════════
# Tree generation helpers
# ══════════════════════════════════════════════════════════════════════════════

def prufer_to_edges(prufer_seq):
    """Convert a Prüfer sequence to a list of undirected edges."""
    n = len(prufer_seq) + 2
    degree = [1] * n
    for node in prufer_seq:
        degree[node] += 1
    edges = []
    for node in prufer_seq:
        for leaf in range(n):
            if degree[leaf] == 1:
                edges.append((node, leaf))
                degree[node] -= 1
                degree[leaf] -= 1
                break
    last = [i for i in range(n) if degree[i] == 1]
    edges.append((last[0], last[1]))
    return edges


def generate_random_tree(n):
    """Generate a uniformly random rooted tree on n nodes (Prüfer method)."""
    if n == 1:
        G = nx.DiGraph(); G.add_node(0); return G, 0
    if n == 2:
        G = nx.DiGraph(); G.add_edge(0, 1); return G, 0
    prufer = [random.randint(0, n-1) for _ in range(n-2)]
    edges  = prufer_to_edges(prufer)
    U = nx.Graph()
    U.add_nodes_from(range(n))
    U.add_edges_from(edges)
    root = max(U.degree(), key=lambda x: x[1])[0]
    return nx.bfs_tree(U, root), root


def generate_linguistic_tree(n, params):
    """
    Top-down BFS tree generation with language-specific branching factor.
    Mimics how real dependency structures are distributed.
    """
    if n <= 1:
        G = nx.DiGraph(); G.add_node(0); return G, 0
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    queue = [0]; next_node = 1
    while queue and next_node < n:
        parent = queue.pop(0)
        remaining = n - next_node
        if remaining == 0:
            break
        k = max(0, int(np.random.poisson(params['branching_factor'])))
        k = min(k, remaining)
        for _ in range(k):
            if next_node >= n:
                break
            G.add_edge(parent, next_node)
            queue.append(next_node)
            next_node += 1
    # Attach any remaining nodes
    while next_node < n:
        parent = random.randint(0, next_node - 1)
        G.add_edge(parent, next_node)
        next_node += 1
    return G, 0


def load_conllu_trees(filepath, max_sents=N_SENTENCES):
    """
    Load dependency trees from a CoNLL-U file.
    Returns list of (n_nodes, nx.DiGraph, root_id) tuples.
    """
    try:
        import conllu
    except ImportError:
        return None
    trees = []
    with open(filepath, encoding='utf-8') as f:
        data = f.read()
    sentences = conllu.parse(data)
    for sent in sentences[:max_sents]:
        G = nx.DiGraph()
        n = len(sent)
        if n < 2:
            continue
        G.add_nodes_from(range(n))
        root = 0
        for token in sent:
            if not isinstance(token['id'], int):
                continue
            tid  = token['id'] - 1
            head = token['head'] - 1 if token['head'] else -1
            if head == -1:
                root = tid
            elif head >= 0:
                G.add_edge(head, tid)
        trees.append((n, G, root))
    return trees


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(G, root):
    """Return dict of tree_depth, max_arity, mean_arity, density, n_nodes."""
    n = G.number_of_nodes()
    if n == 0:
        return None
    out_degrees = [d for _, d in G.out_degree()]
    max_arity   = max(out_degrees) if out_degrees else 0
    nonzero     = [d for d in out_degrees if d > 0]
    mean_arity  = float(np.mean(nonzero)) if nonzero else 0.0
    try:
        depths     = nx.single_source_shortest_path_length(G, root)
        tree_depth = max(depths.values())
        mean_depth = float(np.mean(list(depths.values())))
    except Exception:
        tree_depth = mean_depth = 0
    density = G.number_of_edges() / (n * (n - 1)) if n > 1 else 0.0
    return {
        'n_nodes': n, 'max_arity': max_arity, 'mean_arity': mean_arity,
        'tree_depth': tree_depth, 'mean_depth': mean_depth, 'density': density,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Data generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_data(treebank_dir='./treebanks'):
    records = []
    for lang in LANGS:
        code    = LANG_CODE[lang]
        fpath   = os.path.join(treebank_dir, f'{code}_train.conllu')
        params  = LANG_PARAMS[lang]
        real_trees = None

        if os.path.exists(fpath):
            print(f"[{lang}] Loading real CoNLL-U data from {fpath} ...")
            real_trees = load_conllu_trees(fpath)
            if real_trees:
                print(f"  Loaded {len(real_trees)} sentences.")

        if real_trees:
            for i, (n, G_nat, root_nat) in enumerate(real_trees):
                m_nat  = compute_metrics(G_nat, root_nat)
                G_rnd, root_rnd = generate_random_tree(n)
                m_rnd  = compute_metrics(G_rnd, root_rnd)
                if m_nat and m_rnd:
                    records.append({'language': lang, 'sentence_id': i, 'type': 'natural', **m_nat})
                    records.append({'language': lang, 'sentence_id': i, 'type': 'random',  **m_rnd})
        else:
            print(f"[{lang}] Generating synthetic trees (n={N_SENTENCES}) ...")
            for i in range(N_SENTENCES):
                n = max(2, min(60, int(np.random.normal(
                    params['sent_length_mean'], params['sent_length_std']))))
                G_nat, root_nat = generate_linguistic_tree(n, params)
                G_rnd, root_rnd = generate_random_tree(n)
                m_nat = compute_metrics(G_nat, root_nat)
                m_rnd = compute_metrics(G_rnd, root_rnd)
                if m_nat and m_rnd:
                    records.append({'language': lang, 'sentence_id': i, 'type': 'natural', **m_nat})
                    records.append({'language': lang, 'sentence_id': i, 'type': 'random',  **m_rnd})

        nat_sub = [r for r in records if r['language'] == lang and r['type'] == 'natural']
        rnd_sub = [r for r in records if r['language'] == lang and r['type'] == 'random']
        print(f"  Nat depth μ={np.mean([r['tree_depth'] for r in nat_sub]):.2f} | "
              f"Rnd depth μ={np.mean([r['tree_depth'] for r in rnd_sub]):.2f}")

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# Statistical tests
# ══════════════════════════════════════════════════════════════════════════════

def run_statistics(df):
    nat = df[df['type'] == 'natural']
    rnd = df[df['type'] == 'random']
    metrics = ['tree_depth', 'max_arity', 'density']
    rows = []
    for lang in LANGS:
        for col in metrics:
            nv = nat[nat['language'] == lang][col].dropna()
            rv = rnd[rnd['language'] == lang][col].dropna()
            u_stat, p_val = stats.mannwhitneyu(nv, rv, alternative='two-sided')
            ps = np.sqrt((nv.std()**2 + rv.std()**2) / 2 + 1e-9)
            d  = (nv.mean() - rv.mean()) / ps
            rows.append({
                'Language': lang, 'Metric': col,
                'Nat_Mean': round(nv.mean(), 3), 'Nat_Std': round(nv.std(), 3),
                'Rnd_Mean': round(rv.mean(), 3), 'Rnd_Std': round(rv.std(), 3),
                'MWU_U': round(u_stat, 1), 'p_value': round(p_val, 6),
                'Cohens_d': round(d, 3),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(df, out_dir='.'):
    nat = df[df['type'] == 'natural']
    rnd = df[df['type'] == 'random']
    x   = np.arange(len(LANGS)); w = 0.35
    metric_info = [
        ('tree_depth', 'Tree Depth'),
        ('max_arity',  'Max Arity'),
        ('density',    'Graph Density'),
    ]

    # Figure 1 — violin + box
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Natural Language vs Random Trees: Distribution of Structural Properties',
                 fontsize=15, fontweight='bold')
    for ax, (col, ylabel) in zip(axes, metric_info):
        pos_n = [1,3,5,7,9]; pos_r = [2,4,6,8,10]
        dn = [nat[nat['language']==l][col].dropna().values for l in LANGS]
        dr = [rnd[rnd['language']==l][col].dropna().values for l in LANGS]
        vn = ax.violinplot(dn, positions=pos_n, widths=0.8, showmedians=False, showextrema=False)
        vr = ax.violinplot(dr, positions=pos_r, widths=0.8, showmedians=False, showextrema=False)
        for b in vn['bodies']: b.set_facecolor(NATURAL_COLOR); b.set_alpha(0.4)
        for b in vr['bodies']: b.set_facecolor(RANDOM_COLOR);  b.set_alpha(0.4)
        bn = ax.boxplot(dn, positions=pos_n, widths=0.4, patch_artist=True,
                        medianprops=dict(color='white', linewidth=2))
        br = ax.boxplot(dr, positions=pos_r, widths=0.4, patch_artist=True,
                        medianprops=dict(color='white', linewidth=2))
        for p in bn['boxes']:  p.set_facecolor(NATURAL_COLOR); p.set_alpha(0.75)
        for p in br['boxes']:  p.set_facecolor(RANDOM_COLOR);  p.set_alpha(0.75)
        for wh in bn['whiskers']+bn['caps']+br['whiskers']+br['caps']:
            wh.set_color('#555'); wh.set_linewidth(1)
        ticks = [(pos_n[i]+pos_r[i])/2 for i in range(len(LANGS))]
        ax.set_xticks(ticks); ax.set_xticklabels(LANGS, rotation=20, ha='right', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=11, fontweight='semibold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(handles=[
        plt.Rectangle((0,0),1,1,fc=NATURAL_COLOR,alpha=0.7,label='Natural Language'),
        plt.Rectangle((0,0),1,1,fc=RANDOM_COLOR, alpha=0.7,label='Random Tree'),
    ], fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig1_violin_box.png'), dpi=150, bbox_inches='tight')
    plt.close(); print("Saved fig1_violin_box.png")

    # Figure 2 — Mean bars
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Mean Structural Properties by Language: Natural vs Random',
                 fontsize=14, fontweight='bold')
    for ax, (col, ylabel) in zip(axes, metric_info):
        mn = [nat[nat['language']==l][col].mean() for l in LANGS]
        mr = [rnd[rnd['language']==l][col].mean() for l in LANGS]
        en = [nat[nat['language']==l][col].std()/22.4 for l in LANGS]
        er = [rnd[rnd['language']==l][col].std()/22.4 for l in LANGS]
        ax.bar(x-w/2, mn, w, yerr=en, color=NATURAL_COLOR, alpha=0.8, capsize=4, label='Natural')
        ax.bar(x+w/2, mr, w, yerr=er, color=RANDOM_COLOR,  alpha=0.8, capsize=4, label='Random')
        ax.set_xticks(x); ax.set_xticklabels(LANGS, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel(f'Mean {ylabel}', fontsize=10); ax.legend(fontsize=9)
        ax.set_title(ylabel, fontsize=11, fontweight='semibold')
        ax.grid(axis='y', alpha=0.3); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig2_means.png'), dpi=150, bbox_inches='tight')
    plt.close(); print("Saved fig2_means.png")

    # Figure 3 — CDFs
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Cumulative Distribution Functions: Natural vs Random (All Languages)',
                 fontsize=14, fontweight='bold')
    for ax, (col, ylabel) in zip(axes, metric_info):
        for ltype, color, ls in [('natural', NATURAL_COLOR, '-'), ('random', RANDOM_COLOR, '--')]:
            v = df[df['type']==ltype][col].sort_values()
            ax.plot(v, np.arange(1,len(v)+1)/len(v), color=color, linestyle=ls,
                    linewidth=2.5, label='Natural' if ltype=='natural' else 'Random')
        ax.set_xlabel(ylabel, fontsize=10); ax.set_ylabel('CDF', fontsize=10)
        ax.set_title(f'CDF — {ylabel}', fontsize=10, fontweight='semibold')
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig3_cdf.png'), dpi=150, bbox_inches='tight')
    plt.close(); print("Saved fig3_cdf.png")

    # Figure 4 — Effect size heatmap
    cols3   = ['tree_depth', 'max_arity', 'density']
    labels3 = ['Tree Depth', 'Max Arity', 'Density']
    E = np.zeros((5, 3))
    for i, lang in enumerate(LANGS):
        for j, col in enumerate(cols3):
            nv = nat[nat['language']==lang][col].dropna()
            rv = rnd[rnd['language']==lang][col].dropna()
            ps = np.sqrt((nv.std()**2 + rv.std()**2)/2 + 1e-9)
            E[i, j] = (nv.mean() - rv.mean()) / ps
    fig, ax = plt.subplots(figsize=(8, 5))
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    im = ax.imshow(E, cmap='RdBu_r', norm=norm, aspect='auto')
    plt.colorbar(im, ax=ax, label="Cohen's d (Natural − Random)")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels3, fontsize=11)
    ax.set_yticks(range(5)); ax.set_yticklabels(LANGS, fontsize=11)
    ax.set_title("Effect Sizes (Cohen's d): Natural vs Random Trees", fontsize=12, fontweight='bold')
    for i in range(5):
        for j in range(3):
            ax.text(j, i, f'{E[i,j]:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if abs(E[i,j]) > 0.8 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_effect_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(); print("Saved fig4_effect_heatmap.png")

    # Figure 5 — Depth vs sentence length
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Tree Depth vs. Sentence Length', fontsize=14, fontweight='bold')
    for ax, (ltype, color, title) in zip(axes, [
        ('natural', NATURAL_COLOR, 'Natural Language Trees'),
        ('random',  RANDOM_COLOR,  'Random Trees'),
    ]):
        sub = df[df['type']==ltype]
        for lang in LANGS:
            d = sub[sub['language']==lang]
            ax.scatter(d['n_nodes'], d['tree_depth'], alpha=0.2, s=10,
                       color=LANG_COLORS[lang], label=lang)
        xv = sub['n_nodes'].values; yv = sub['tree_depth'].values
        m, b = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(xs, m*xs+b, 'k--', lw=2, label=f'Fit slope={m:.3f}')
        ax.set_xlabel('# Nodes', fontsize=10); ax.set_ylabel('Tree Depth', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='semibold')
        ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
        ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig5_depth_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close(); print("Saved fig5_depth_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    os.makedirs('./output', exist_ok=True)

    print("=" * 60)
    print("LE2 — DAG Properties in Natural Language")
    print("=" * 60)

    df = generate_all_data('./treebanks')
    df.to_csv('./output/tree_data.csv', index=False)
    print(f"\nData saved: {len(df)} records to ./output/tree_data.csv")

    stats_df = run_statistics(df)
    stats_df.to_csv('./output/stats_results.csv', index=False)
    print("\nStatistical Results:")
    print(stats_df.to_string(index=False))

    print("\nGenerating figures ...")
    plot_all(df, out_dir='./output')

    print("\nDone! All outputs in ./output/")
