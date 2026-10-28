import math
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm

from scripts.ngrams import encode_stories, fit_counts, loss, split_stories

ROOT = Path(__file__).resolve().parents[1]


def test_stories_and_actual_tokenizer():
    train, valid = split_stories('  alpha  <|endoftext|>  <|endoftext|> beta\n<|endoftext|> gamma ')
    assert train == ['alpha', 'beta']
    assert valid == ['gamma']
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / 'inputs/stories260K/tok512.model'))
    tokens = encode_stories(train, sp)
    assert tokens.tolist() == sp.encode('alpha') + [sp.eos_id()] + sp.encode('beta')
    assert encode_stories(valid, sp).tolist() == sp.encode('gamma')


def test_counts_and_leave_history_probabilities():
    train = np.array([1, 2, 1, 2, 3, 1], dtype=np.int32)
    target = np.array([2, 3, 1, 1], dtype=np.int32)
    vocabulary, eos, k = 4, 0, 0.1
    for n in range(1, 7):
        padded = [eos] * (n - 1) + train.tolist()
        counts = Counter(tuple(padded[i:i+n]) for i in range(len(train)))
        contexts = Counter({ctx: sum(v for key, v in counts.items() if key[:-1] == ctx)
                            for ctx in {key[:-1] for key in counts}})
        history = [eos] * (n - 1) + target.tolist()
        expected = []
        for i in range(len(target)):
            gram = tuple(history[i:i+n])
            p = (counts[gram] + k) / (contexts[gram[:-1]] + k * vocabulary)
            expected.append(-math.log(p))
        model = fit_counts(train, n, eos)
        actual = float(loss(model, target, n, eos, vocabulary, k))
        np.testing.assert_allclose(actual, np.mean(expected), rtol=1e-6)


def test_saved_workflow(tmp_path):
    import csv
    import json
    import subprocess
    import sys

    stories = [f'Once a cat saw dog number {i}.' for i in range(12)]
    source = tmp_path / 'stories.txt'
    source.write_text(' <|endoftext|> '.join(stories))
    output = tmp_path / 'run'
    subprocess.run([sys.executable, '-m', 'scripts.ngrams', '--data', str(source), '--output', str(output)], cwd=ROOT, check=True)
    settings = json.loads((output / 'settings.json').read_text())
    assert settings['train_stories'] == 10
    assert settings['validation_stories'] == 2
    assert settings['validation_characters'] == sum(map(len, stories[10:]))
    rows = list(csv.DictReader((output / 'metrics.csv').open()))
    assert [r['model'] for r in rows] == ['uniform', '1-gram', '2-gram', '3-gram', '4-gram', '5-gram', '6-gram']
    assert math.isclose(float(rows[0]['cross_entropy_nats']), math.log(512), rel_tol=1e-6)
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / 'inputs/stories260K/tok512.model'))
    train = encode_stories(stories[:10], sp).tolist()
    valid = encode_stories(stories[10:], sp).tolist()
    cut = len(train) * 9 // 10
    ks = [0.0001, 0.001, 0.01, 0.1, 1.0]
    for n, row in enumerate(rows[1:], 1):
        def ref_loss(training, evaluation, k, prefix):
            data = [sp.eos_id()] * (n - 1) + training
            counts = Counter(tuple(data[i:i+n]) for i in range(len(training)))
            contexts = Counter()
            for gram, count in counts.items():
                contexts[gram[:-1]] += count
            data = ([sp.eos_id()] * (n - 1) + prefix)[-(n - 1):] + evaluation if n > 1 else evaluation
            return sum(-math.log((counts[tuple(data[i:i+n])] + k) / (contexts[tuple(data[i:i+n-1])] + 512*k)) for i in range(len(evaluation))) / len(evaluation)
        best = min(ks, key=lambda k: ref_loss(train[:cut], train[cut:], k, train[:cut]))
        assert float(row['k']) == best
        with np.load(output / f'{n}-gram.npz') as saved:
            model = tuple(saved[name] for name in ('grams', 'counts', 'contexts', 'context_counts'))
            actual = float(loss(model, valid, n, sp.eos_id(), 512, best))
        expected = ref_loss(train, valid, best, [])
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        np.testing.assert_allclose(float(row['cross_entropy_nats']), expected, rtol=1e-6)
        np.testing.assert_allclose(float(row['perplexity']), math.exp(expected), rtol=1e-6)
        np.testing.assert_allclose(float(row['bits_per_character']), expected * len(valid) / (math.log(2) * sum(map(len, stories[10:]))), rtol=1e-6)
