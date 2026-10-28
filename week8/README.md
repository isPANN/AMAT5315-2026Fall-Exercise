# Week 8: static embeddings

Run from `week8/` using uv (CPU by default):

```sh
uv run --no-project --with jax --with numpy --with huggingface-hub --with safetensors --with tokenizers python -m scripts.embeddings
uv run --no-project --with jax --with numpy --with huggingface-hub --with safetensors --with tokenizers --with pytest python -m pytest scripts/test_embeddings.py -q
```

The command downloads `model.safetensors` and `tokenizer.json` from
`minishlab/potion-base-8M` at revision `bf8b056` into `inputs/`. It loads the
table without PyTorch, permutes its rows using JAX seed 1730, and saves the
control and settings under `artifacts/embeddings/`. Both directories are ignored
by git. Set `JAX_PLATFORMS` explicitly to override the CPU default.

`load_embeddings()` returns the tokenizer, original table and shuffled table.
`word_vector(tokenizer, table, word)` looks up an exact vocabulary token and
returns its row; an absent token raises `KeyError` rather than using an unknown
or subword token.

## Relation offsets

```sh
uv run --no-project --with jax --with numpy --with huggingface-hub --with safetensors --with tokenizers --with matplotlib python -m scripts.relations
uv run --no-project --with jax --with numpy --with huggingface-hub --with safetensors --with tokenizers --with matplotlib --with pytest python -m pytest scripts -q
```

For each table, every row is normalized to unit length. Each pair's offset is
its target row minus its source row; the reported cosine compares that offset
to the average of the other eight offsets in the same relation. The horizontal
coordinate of each unit arrow is this cosine. Original-table arrows appear above
the axis and shuffled-control arrows below it. `evidence/relations.png` contains
the figure. `artifacts/relations/cosines.csv` records all 54 cosines, and
`settings.json` records the seed, revision, device, JAX version and means.

| Relation | Table mean | Control mean |
| --- | ---: | ---: |
| Country to capital | 0.465874 | 0.055170 |
| Male to female | 0.548518 | 0.003449 |
| Adjective to comparative | 0.433004 | -0.047844 |

These are results for one seeded permutation, not a statistical significance test.
