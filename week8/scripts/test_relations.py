import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer

from scripts.relations import RELATIONS, offset_cosines

ROOT = Path(__file__).resolve().parents[1]


def test_leave_one_out_offsets():
    vectors = np.array([[1, 0], [0, 1], [1, 1], [0, -1], [-1, 0], [-1, -1]], dtype=float)
    pairs = [(0, 1), (2, 3), (4, 5)]
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    offsets = np.array([normalized[b] - normalized[a] for a, b in pairs])
    expected = []
    for i in range(len(pairs)):
        other = np.mean(np.delete(offsets, i, axis=0), axis=0)
        expected.append(np.dot(offsets[i], other) / (np.linalg.norm(offsets[i]) * np.linalg.norm(other)))
    np.testing.assert_allclose(offset_cosines(vectors, pairs), expected, atol=1e-6)


def test_real_analysis_outputs(tmp_path):
    output = tmp_path / 'run'
    figure = tmp_path / 'relations.png'
    subprocess.run([sys.executable, '-m', 'scripts.relations', '--output', str(output), '--figure', str(figure)], cwd=ROOT, check=True)
    tokenizer = Tokenizer.from_file(str(ROOT / 'inputs/tokenizer.json'))
    table = load_file(str(ROOT / 'inputs/model.safetensors'))['embeddings'].astype(np.float64)
    control = np.load(ROOT / 'artifacts/embeddings/control.npy').astype(np.float64)
    rows = list(csv.DictReader((output / 'cosines.csv').open()))
    assert len(rows) == 27
    for relation, words in RELATIONS.items():
        group = [row for row in rows if row['relation'] == relation]
        assert [(row['source'], row['target']) for row in group] == words
        for name, data in [('table', table), ('control', control)]:
            vectors = np.array([[data[tokenizer.token_to_id(a)], data[tokenizer.token_to_id(b)]] for a, b in words])
            vectors /= np.linalg.norm(vectors, axis=2, keepdims=True)
            offsets = vectors[:, 1] - vectors[:, 0]
            expected = []
            for i in range(9):
                other = np.mean(np.delete(offsets, i, axis=0), axis=0)
                expected.append(np.dot(offsets[i], other) / (np.linalg.norm(offsets[i]) * np.linalg.norm(other)))
            np.testing.assert_allclose([float(row[name]) for row in group], expected, atol=1e-6)
    assert json.loads((output / 'settings.json').read_text())['seed'] == 1730
    assert figure.stat().st_size > 1000
