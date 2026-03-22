# NumPy word2vec

This repository implements the core training loop of **word2vec** in pure **NumPy**, using the **skip-gram with negative sampling** objective.

## What is included

- forward pass for positive and negative pairs
- loss computation with numerically stable log-sigmoid terms
- analytical gradients for center and context embeddings
- SGD parameter updates with `np.add.at` so repeated word ids are handled correctly
- a small bundled corpus based on the public-domain opening of *Alice's Adventures in Wonderland*

## Model choice

The implementation uses skip-gram with negative sampling:

- input embedding matrix `V` stores center-word vectors
- output embedding matrix `U` stores context-word vectors
- for each observed pair `(center, context)`, the model maximizes

$$
\log \sigma(u_o^\top v_c) + \sum_{k=1}^{K} \log \sigma(-u_{n_k}^\top v_c)
$$

where `o` is the positive context word and `n_k` are negative samples.

## Gradient summary

For one training example:

- positive logit: $x = u_o^\top v_c$
- negative logits: $z_k = u_{n_k}^\top v_c$

The loss is

$$
L = -\log \sigma(x) - \sum_{k=1}^{K} \log \sigma(-z_k)
$$

The resulting gradients are:

$$
\frac{\partial L}{\partial v_c} = (\sigma(x)-1)u_o + \sum_{k=1}^{K} \sigma(z_k)u_{n_k}
$$

$$
\frac{\partial L}{\partial u_o} = (\sigma(x)-1)v_c
$$

$$
\frac{\partial L}{\partial u_{n_k}} = \sigma(z_k)v_c
$$

The code applies the same formulas in mini-batches and averages gradients over the batch before each SGD update.

## Files

- `word2vec_numpy.py` – full data pipeline, objective, gradients, optimizer, and embedding export
- `data/alice_excerpt.txt` – small offline corpus
- `requirements.txt` – minimal dependency list

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python word2vec_numpy.py
```

Optional arguments:

```bash
python word2vec_numpy.py --embedding-dim 64 --window-size 3 --negative-samples 8 --epochs 80
```

The script prints epoch losses, saves embeddings to `artifacts/embeddings.npz`, and shows a few nearest-neighbor examples.

## Tests

The repository includes a unit test suite in `tests/test_word2vec_numpy.py` that checks:

- tokenization behavior
- vocabulary filtering and deterministic ordering
- context-pair generation for a fixed window
- negative sampling constraints (forbidden ids are not sampled)
- gradient correctness with finite-difference checks
- that one SGD step reduces loss on the same mini-batch

Run tests with:

```bash
python -m unittest discover -s tests -v
```

Expected result: all tests pass.


