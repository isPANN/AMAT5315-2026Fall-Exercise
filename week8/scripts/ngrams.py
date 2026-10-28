import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import sentencepiece as spm
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
KS = (0.0001, 0.001, 0.01, 0.1, 1.0)


def split_stories(text):
    stories = [story.strip() for story in text.split('<|endoftext|>') if story.strip()]
    cut = len(stories) * 9 // 10
    return stories[:cut], stories[cut:]


def encode_stories(stories, tokenizer):
    encoded = tokenizer.encode(stories)
    result = []
    for i, tokens in enumerate(encoded):
        if i:
            result.append(tokenizer.eos_id())
        result.extend(tokens)
    return np.asarray(result, dtype=np.int32)


def gram_keys(tokens, n, eos, prefix=()):
    history = jnp.concatenate((jnp.full(n - 1, eos, dtype=jnp.uint64), jnp.asarray(prefix, dtype=jnp.uint64)))
    history = history[-(n-1):] if n > 1 else history[:0]
    data = jnp.concatenate((history, jnp.asarray(tokens, dtype=jnp.uint64)))
    keys = jnp.zeros(len(tokens), dtype=jnp.uint64)
    for i in range(n):
        keys = (keys << 9) | data[i:i+len(tokens)]
    return keys


def fit_counts(tokens, n, eos):
    keys = gram_keys(tokens, n, eos)
    grams, counts = jnp.unique(keys, return_counts=True)
    contexts, context_counts = jnp.unique(keys >> 9, return_counts=True)
    return grams, counts, contexts, context_counts


def lookup(keys, values, queries):
    indices = jnp.searchsorted(keys, queries)
    indices = jnp.minimum(indices, len(keys) - 1)
    return jnp.where(keys[indices] == queries, values[indices], 0)


def observed_counts(model, tokens, n, eos, prefix=()):
    grams, counts, contexts, context_counts = model
    keys = gram_keys(tokens, n, eos, prefix)
    return lookup(grams, counts, keys), lookup(contexts, context_counts, keys >> 9)


def loss(model, tokens, n, eos, vocabulary, k, prefix=()):
    counts, contexts = observed_counts(model, tokens, n, eos, prefix)
    return -jnp.mean(jnp.log((counts + k) / (contexts + vocabulary * k)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data/TinyStoriesV2-GPT4-valid.txt')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/ngrams')
    args = parser.parse_args()
    model_path = ROOT / 'inputs/stories260K/tok512.model'
    if not model_path.exists():
        hf_hub_download('karpathy/tinyllamas', 'stories260K/tok512.model', revision='0bd21da', local_dir=ROOT / 'inputs')
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
    train_stories, valid_stories = split_stories(args.data.read_text(encoding='utf-8'))
    train = encode_stories(train_stories, tokenizer)
    valid = encode_stories(valid_stories, tokenizer)
    cut = len(train) * 9 // 10
    chars = sum(len(story) for story in valid_stories)
    vocab, eos = tokenizer.vocab_size(), tokenizer.eos_id()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'tokens.npz', train=train, validation=valid)
    rows = []
    tuning = {}

    def report(name, k, value):
        row = {'model': name, 'k': k, 'cross_entropy_nats': value,
               'perplexity': math.exp(value), 'bits_per_character': value * len(valid) / (math.log(2) * chars)}
        rows.append(row)
        print(f"{name:7s} k={str(k):6s} nats/token={value:.6f} perplexity={row['perplexity']:.6f} bits/character={row['bits_per_character']:.6f}", flush=True)

    report('uniform', '', math.log(vocab))
    for n in range(1, 7):
        print(f'Fitting {n}-gram for tuning...', flush=True)
        model = fit_counts(train[:cut], n, eos)
        counts, contexts = observed_counts(model, train[cut:], n, eos, train[max(0, cut-n+1):cut])
        losses = [float(-jnp.mean(jnp.log((counts + k) / (contexts + vocab*k)))) for k in KS]
        tuning[str(n)] = dict(zip(map(str, KS), losses))
        k = KS[int(np.argmin(losses))]
        del model, counts, contexts
        print(f'Refitting {n}-gram...', flush=True)
        model = fit_counts(train, n, eos)
        np.savez(args.output / f'{n}-gram.npz',
                 **dict(zip(('grams', 'counts', 'contexts', 'context_counts'), map(np.asarray, model))))
        value = float(loss(model, valid, n, eos, vocab, k))
        report(f'{n}-gram', k, value)
        del model
    with (args.output / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    settings = {'source': str(args.data), 'tokenizer': 'karpathy/tinyllamas/stories260K/tok512.model',
                'revision': '0bd21da', 'vocabulary_size': vocab, 'eos_id': eos,
                'train_stories': len(train_stories), 'validation_stories': len(valid_stories),
                'train_tokens': len(train), 'tuning_start': cut, 'validation_tokens': len(valid),
                'validation_characters': chars, 'smoothing_candidates': KS, 'tuning_loss_nats': tuning,
                'initial_context': 'EOS repeated n-1 times', 'jax_version': jax.__version__, 'sentencepiece_version': spm.__version__, 'device': str(jax.devices()[0])}
    (args.output / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')


if __name__ == '__main__':
    main()
