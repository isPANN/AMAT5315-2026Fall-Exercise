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

## TinyStories n-grams

Place `TinyStoriesV2-GPT4-valid.txt` in `data/`, then run:

```sh
uv run --no-project --with jax --with numpy --with sentencepiece --with huggingface-hub python -m scripts.ngrams
uv run --no-project --with jax --with numpy --with sentencepiece --with huggingface-hub --with pytest python -m pytest scripts/test_ngrams.py -q
```

The input file is the supplied TinyStories validation file, split anew here:
separate stories on `<|endoftext|>`, strip them, discard empty stories and take
the first floor(90%) as training. The remaining stories are held out. The
SentencePiece BPE model is downloaded from
`karpathy/tinyllamas/stories260K/tok512.model` at revision `0bd21da` into
`inputs/`. Each split is encoded separately, with one EOS token between
stories and no added trailing EOS. For the start of each split, the context is
padded with `n-1` EOS tokens; contexts elsewhere can cross story boundaries
through the EOS separator. Every held-out token is scored.

For each n from 1 to 6, counts fitted to the first floor(90%) of training
**tokens** are used to select k from `0.0001, 0.001, 0.01, 0.1, 1` by the
lowest cross-entropy on the remaining training tokens. Tuning uses the actual
preceding training tokens as context, and the selected model is then refitted
on all training tokens. For vocabulary size 512, the probability is
`(count(context, token) + k) / (count(context) + 512*k)`; unseen contexts are
therefore uniform. The uniform baseline assigns `1/512` to each token.
Cross-entropy is the mean negative natural log probability. Perplexity is its
exponential. Bits per character is total held-out negative log probability in
bits divided by the number of Unicode characters in the stripped held-out
stories; EOS separators contribute to the numerator but not the denominator.
SentencePiece normalization applies, so this is token loss per source character,
not necessarily a lossless compression rate.

This run has 24,867 training stories (9,667,034 tokens) and 2,763 held-out
stories (1,078,995 tokens; 2,213,915 characters):

| Model | k | Nats/token | Perplexity | Bits/character |
| --- | ---: | ---: | ---: | ---: |
| Uniform | — | 6.238325 | 512.000000 | 4.386327 |
| 1-gram | 0.0001 | 4.839399 | 126.393370 | 3.402706 |
| 2-gram | 0.01 | 3.038192 | 20.867476 | 2.136231 |
| 3-gram | 0.01 | 2.094340 | 8.120083 | 1.472585 |
| 4-gram | 0.01 | 1.862754 | 6.441454 | 1.309751 |
| 5-gram | 0.01 | 2.082535 | 8.024786 | 1.464284 |
| 6-gram | 0.001 | 2.550398 | 12.812203 | 1.793251 |

`artifacts/ngrams/` contains the token arrays, fitted n-gram and context counts,
tuning losses, selected settings and metrics. Counts are stored as sorted integer
keys, using nine bits per token. Inputs, data and artifacts are ignored by git.
