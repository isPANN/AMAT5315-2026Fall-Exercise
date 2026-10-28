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
