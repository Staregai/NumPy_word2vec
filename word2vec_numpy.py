from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-z']+") #regex: keep lowercase and apostrophes, ignore punctuation and numbers


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class Vocabulary:
    stoi: dict[str, int]
    itos: list[str]
    counts: np.ndarray

    @classmethod
    def from_tokens(cls, tokens: list[str], min_count: int = 1) -> "Vocabulary":
        counts = Counter(tokens)
        items = [(word, count) for word, count in counts.items() if count >= min_count] #filter low frequency words
        items.sort(key=lambda item: (-item[1], item[0])) #high frequency first, alphebetical if tied
        words = [word for word, _ in items]
        stoi = {word: index for index, word in enumerate(words)}
        freqs = np.array([count for _, count in items], dtype=np.int64)
        return cls(stoi=stoi, itos=words, counts=freqs)

    def encode(self, tokens: list[str]) -> np.ndarray:
        return np.array([self.stoi[token] for token in tokens if token in self.stoi], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.itos)


class NegativeSampler:
    def __init__(self, counts: np.ndarray, power: float = 0.75, seed: int = 0) -> None:
        adjusted = counts.astype(np.float64) ** power
        self.probabilities = adjusted / adjusted.sum()
        self.vocab_size = counts.shape[0]
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        batch_size: int,
        negative_samples: int,
        forbidden: np.ndarray | None = None,
    ) -> np.ndarray:
        negatives = self.rng.choice(
            self.vocab_size,
            size=(batch_size, negative_samples),
            p=self.probabilities,
        )
        if forbidden is None:
            return negatives
        if forbidden.ndim == 1:
            forbidden = forbidden[:, None]
        mask = np.any(negatives[:, :, None] == forbidden[:, None, :], axis=2)
        while np.any(mask):
            negatives[mask] = self.rng.choice(self.vocab_size, size=int(mask.sum()), p=self.probabilities)
            mask = np.any(negatives[:, :, None] == forbidden[:, None, :], axis=2)
        return negatives


class SkipGramNegativeSampling:
    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 0) -> None:
        self.embedding_dim = embedding_dim
        self.rng = np.random.default_rng(seed)
        limit = 0.5 / max(1, embedding_dim)
        self.input_embeddings = self.rng.uniform(
            -limit,
            limit,
            size=(vocab_size, embedding_dim),
        ).astype(np.float64)
        self.output_embeddings = self.rng.uniform(
            -limit,
            limit,
            size=(vocab_size, embedding_dim),
        ).astype(np.float64)

    @staticmethod
    def sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -15.0, 15.0) #clipping to prevent overflow
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def log_sigmoid(values: np.ndarray) -> np.ndarray: #usable for stable loss computation
        return -np.logaddexp(0.0, -values)

    def batch_loss_and_gradients(
        self,
        center_ids: np.ndarray,
        positive_ids: np.ndarray,
        negative_ids: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        center_vectors = self.input_embeddings[center_ids]
        positive_vectors = self.output_embeddings[positive_ids]
        negative_vectors = self.output_embeddings[negative_ids]

        positive_logits = np.sum(center_vectors * positive_vectors, axis=1)
        negative_logits = np.einsum("bd,bkd->bk", center_vectors, negative_vectors) #using einsum to compute dot product for whole batch at once

        positive_sigmoid = self.sigmoid(positive_logits)
        negative_sigmoid = self.sigmoid(negative_logits)

        loss = -np.mean(
            self.log_sigmoid(positive_logits)
            + np.sum(self.log_sigmoid(-negative_logits), axis=1)
        )

        scale = 1.0 / center_ids.shape[0]
        positive_residual = (positive_sigmoid - 1.0)[:, None] * scale
        negative_residual = negative_sigmoid[:, :, None] * scale

        grad_center = positive_residual * positive_vectors + np.sum(
            negative_residual * negative_vectors,
            axis=1,
        )
        grad_positive = positive_residual * center_vectors
        grad_negative = negative_residual * center_vectors[:, None, :]
        return loss, grad_center, grad_positive, grad_negative

    def apply_gradients(
        self,
        center_ids: np.ndarray,
        positive_ids: np.ndarray,
        negative_ids: np.ndarray,
        grad_center: np.ndarray,
        grad_positive: np.ndarray,
        grad_negative: np.ndarray,
        learning_rate: float,
    ) -> None:
        np.add.at(self.input_embeddings, center_ids, -learning_rate * grad_center)
        np.add.at(self.output_embeddings, positive_ids, -learning_rate * grad_positive)
        np.add.at(
            self.output_embeddings,
            negative_ids.reshape(-1),
            -learning_rate * grad_negative.reshape(-1, self.embedding_dim),
        )

    def fit(
        self,
        center_ids: np.ndarray,
        context_ids: np.ndarray,
        sampler: NegativeSampler,
        epochs: int,
        batch_size: int,
        negative_samples: int,
        learning_rate: float,
    ) -> list[float]:
        example_count = center_ids.shape[0]
        losses: list[float] = []
        for epoch in range(1, epochs + 1):
            permutation = self.rng.permutation(example_count)
            shuffled_centers = center_ids[permutation]
            shuffled_contexts = context_ids[permutation]
            epoch_losses: list[float] = []

            for start in range(0, example_count, batch_size):
                stop = min(start + batch_size, example_count)
                batch_centers = shuffled_centers[start:stop]
                batch_contexts = shuffled_contexts[start:stop]
                forbidden = np.stack([batch_centers, batch_contexts], axis=1)
                batch_negatives = sampler.sample(
                    batch_size=batch_centers.shape[0],
                    negative_samples=negative_samples,
                    forbidden=forbidden,
                )
                loss, grad_center, grad_positive, grad_negative = self.batch_loss_and_gradients(
                    batch_centers,
                    batch_contexts,
                    batch_negatives,
                )
                self.apply_gradients(
                    batch_centers,
                    batch_contexts,
                    batch_negatives,
                    grad_center,
                    grad_positive,
                    grad_negative,
                    learning_rate,
                )
                epoch_losses.append(loss)

            epoch_loss = float(np.mean(epoch_losses))
            losses.append(epoch_loss)
            print(f"epoch {epoch:03d} | loss {epoch_loss:.4f}")
        return losses

    def embeddings(self) -> np.ndarray:
        return self.input_embeddings + self.output_embeddings

    def most_similar(self, word_id: int, top_k: int = 5) -> list[tuple[int, float]]:
        vectors = self.embeddings()
        normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
        scores = normalized @ normalized[word_id]
        scores[word_id] = -np.inf
        best_ids = np.argpartition(scores, -top_k)[-top_k:]
        best_ids = best_ids[np.argsort(scores[best_ids])[::-1]]
        return [(int(index), float(scores[index])) for index in best_ids]


def build_training_pairs(token_ids: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    centers: list[int] = []
    contexts: list[int] = []
    for position, center_id in enumerate(token_ids):
        left = max(0, position - window_size)
        right = min(token_ids.shape[0], position + window_size + 1)
        for context_position in range(left, right):
            if context_position == position:
                continue
            centers.append(int(center_id))
            contexts.append(int(token_ids[context_position]))
    return np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)


def load_corpus(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def summarize_neighbors(model: SkipGramNegativeSampling, vocabulary: Vocabulary, query_words: list[str]) -> None:
    available = [word for word in query_words if word in vocabulary.stoi]
    if not available:
        return
    print("\nnearest neighbors")
    for word in available:
        neighbors = model.most_similar(vocabulary.stoi[word])
        formatted = ", ".join(f"{vocabulary.itos[index]} ({score:.3f})" for index, score in neighbors)
        print(f"{word}: {formatted}")


def save_embeddings(path: Path, vocabulary: Vocabulary, vectors: np.ndarray) -> None:
    np.savez_compressed(path, words=np.array(vocabulary.itos), embeddings=vectors)
    print(f"\nsaved embeddings to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train skip-gram word2vec with negative sampling in pure NumPy.")
    parser.add_argument("--corpus", type=Path, default=Path("data/alice_excerpt.txt"))
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--negative-samples", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts/embeddings.npz"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = load_corpus(args.corpus)
    tokens = tokenize(text)
    vocabulary = Vocabulary.from_tokens(tokens, min_count=args.min_count)
    token_ids = vocabulary.encode(tokens)
    center_ids, context_ids = build_training_pairs(token_ids, window_size=args.window_size)

    print(f"tokens: {len(tokens)}")
    print(f"vocab size: {len(vocabulary)}")
    print(f"training pairs: {center_ids.shape[0]}")

    sampler = NegativeSampler(vocabulary.counts, seed=args.seed)
    model = SkipGramNegativeSampling(
        vocab_size=len(vocabulary),
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )
    model.fit(
        center_ids=center_ids,
        context_ids=context_ids,
        sampler=sampler,
        epochs=args.epochs,
        batch_size=args.batch_size,
        negative_samples=args.negative_samples,
        learning_rate=args.learning_rate,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_embeddings(args.output, vocabulary, model.embeddings())
    summarize_neighbors(model, vocabulary, ["alice", "rabbit", "sister", "book", "pictures"])


if __name__ == "__main__":
    main()
