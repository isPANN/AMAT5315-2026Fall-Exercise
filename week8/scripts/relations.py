import argparse
import csv
import json
from pathlib import Path

from scripts.embeddings import ROOT, REPO, REVISION, SEED, load_embeddings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

RELATIONS = {
    'Country to capital': [('iran', 'tehran'), ('turkey', 'ankara'), ('russia', 'moscow'), ('china', 'beijing'), ('france', 'paris'), ('sweden', 'stockholm'), ('japan', 'tokyo'), ('peru', 'lima'), ('kenya', 'nairobi')],
    'Male to female': [('king', 'queen'), ('man', 'woman'), ('he', 'she'), ('brother', 'sister'), ('father', 'mother'), ('son', 'daughter'), ('uncle', 'aunt'), ('boy', 'girl'), ('husband', 'wife')],
    'Adjective to comparative': [('big', 'bigger'), ('small', 'smaller'), ('fast', 'faster'), ('dark', 'darker'), ('cold', 'colder'), ('old', 'older'), ('strong', 'stronger'), ('long', 'longer'), ('tall', 'taller')],
}


def offset_cosines(table, pairs):
    table = jnp.asarray(table)
    table = table / jnp.linalg.norm(table, axis=1, keepdims=True)
    ids = jnp.asarray(pairs)
    offsets = table[ids[:, 1]] - table[ids[:, 0]]
    others = (jnp.sum(offsets, axis=0) - offsets) / (len(pairs) - 1)
    return jnp.sum(offsets * others, axis=1) / (jnp.linalg.norm(offsets, axis=1) * jnp.linalg.norm(others, axis=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/relations')
    parser.add_argument('--figure', type=Path, default=ROOT / 'evidence/relations.png')
    args = parser.parse_args()
    tokenizer, table, control = load_embeddings()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    rows = []
    means = {}
    colors = plt.get_cmap('tab10').colors
    for ax, (relation, words) in zip(axes, RELATIONS.items()):
        ids = [(tokenizer.token_to_id(a), tokenizer.token_to_id(b)) for a, b in words]
        if any(a is None or b is None for a, b in ids):
            raise ValueError(f'Missing exact token in {relation}')
        original = np.asarray(offset_cosines(table, ids))
        shuffled = np.asarray(offset_cosines(control, ids))
        means[relation] = {'table': float(original.mean()), 'control': float(shuffled.mean())}
        for i, ((a, b), x, y) in enumerate(zip(words, original, shuffled)):
            rows.append([relation, a, b, float(x), float(y)])
            for cosine, sign in ((x, 1), (y, -1)):
                angle = np.arccos(np.clip(cosine, -1, 1))
                ax.arrow(0, 0, np.cos(angle), sign * np.sin(angle), color=colors[i],
                         width=0.004, head_width=0.04, length_includes_head=True, alpha=0.85)
            ax.plot([], [], color=colors[i], label=f'{a} → {b}')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect='equal', title=relation, xlabel='cosine')
        ax.set_ylabel('Table above / control below')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.51), ncol=2, fontsize=8, frameon=False)
        print(f"{relation}: table {means[relation]['table']:.6f}, control {means[relation]['control']:.6f}")
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    with (args.output / 'cosines.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['relation', 'source', 'target', 'table', 'control'])
        writer.writerows(rows)
    (args.output / 'settings.json').write_text(json.dumps({
        'model': REPO, 'revision': REVISION, 'seed': SEED, 'jax_version': jax.__version__,
        'device': str(table.device), 'normalization': 'unit row before offsets',
        'cosine': 'offset versus mean of the other eight offsets', 'means': means,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
