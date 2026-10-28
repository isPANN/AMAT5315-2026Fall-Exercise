import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file
from tokenizers import Tokenizer

from scripts.embeddings import load_embeddings, word_vector

ROOT = Path(__file__).resolve().parents[1]


def test_table_and_word_lookup():
    tokenizer, table, control = load_embeddings()
    original = load_file(str(ROOT / 'inputs/model.safetensors'))['embeddings']
    reference_tokenizer = Tokenizer.from_file(str(ROOT / 'inputs/tokenizer.json'))
    assert table.shape == control.shape == (29528, 256)
    np.testing.assert_array_equal(table, original)
    for word in ('cat', 'dog', 'hello'):
        token_id = reference_tokenizer.token_to_id(word)
        np.testing.assert_array_equal(word_vector(tokenizer, table, word), original[token_id])
        np.testing.assert_array_equal(word_vector(tokenizer, control, word), control[token_id])
    with pytest.raises(KeyError):
        word_vector(tokenizer, table, 'not-a-token-5315')
    _, _, repeated = load_embeddings()
    np.testing.assert_array_equal(control, repeated)
    original_order = np.lexsort((original[:, 1], original[:, 0]))
    control_order = np.lexsort((np.asarray(control)[:, 1], np.asarray(control)[:, 0]))
    np.testing.assert_array_equal(np.asarray(control)[control_order], original[original_order])
    assert not np.array_equal(control, original)


def test_saved_control(tmp_path):
    subprocess.run([sys.executable, '-m', 'scripts.embeddings', '--output', str(tmp_path)], cwd=ROOT, check=True)
    settings = json.loads((tmp_path / 'settings.json').read_text())
    assert settings['revision'] == 'bf8b056'
    assert settings['seed'] == 1730
    _, _, control = load_embeddings()
    np.testing.assert_array_equal(np.load(tmp_path / 'control.npy'), control)
