import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
REPO = 'minishlab/potion-base-8M'
REVISION = 'bf8b056'
SEED = 1730


def load_embeddings():
    paths = [ROOT / 'inputs' / name for name in ('model.safetensors', 'tokenizer.json')]
    for path in paths:
        if not path.exists():
            hf_hub_download(REPO, path.name, revision=REVISION, local_dir=path.parent)
    tokenizer = Tokenizer.from_file(str(paths[1]))
    table = jnp.asarray(load_file(str(paths[0]))['embeddings'])
    control = jax.random.permutation(jax.random.key(SEED), table, axis=0)
    return tokenizer, table, control


def word_vector(tokenizer, table, word):
    token_id = tokenizer.token_to_id(word)
    if token_id is None:
        raise KeyError(f'No exact token: {word}')
    return table[token_id]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/embeddings')
    args = parser.parse_args()
    _, table, control = load_embeddings()
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'control.npy', np.asarray(control))
    (args.output / 'settings.json').write_text(json.dumps({
        'model': REPO, 'revision': REVISION, 'seed': SEED,
        'shuffle': 'jax.random.permutation of rows', 'shape': list(table.shape),
        'device': str(table.device), 'jax_version': jax.__version__,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
