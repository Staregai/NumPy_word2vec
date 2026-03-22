import unittest

import numpy as np

import word2vec_numpy as w2v


class TestWord2VecNumpy(unittest.TestCase):
    def test_tokenize_keeps_words_and_apostrophes(self) -> None:
        text = "Hello, WORLD! It's 2026."
        tokens = w2v.tokenize(text)
        self.assertEqual(tokens, ["hello", "world", "it's"])

    def test_vocabulary_sorting_and_min_count(self) -> None:
        tokens = ["b", "a", "b", "c", "a", "d"]
        vocab = w2v.Vocabulary.from_tokens(tokens, min_count=2)
        # counts: a=2, b=2 (alphabetical tie-break), c=1, d=1 removed
        self.assertEqual(vocab.itos, ["a", "b"])
        self.assertEqual(vocab.stoi, {"a": 0, "b": 1})
        self.assertTrue(np.array_equal(vocab.counts, np.array([2, 2], dtype=np.int64)))

    def test_build_training_pairs_window_one(self) -> None:
        token_ids = np.array([0, 1, 2], dtype=np.int64)
        centers, contexts = w2v.build_training_pairs(token_ids, window_size=1)
        expected_centers = np.array([0, 1, 1, 2], dtype=np.int64)
        expected_contexts = np.array([1, 0, 2, 1], dtype=np.int64)
        self.assertTrue(np.array_equal(centers, expected_centers))
        self.assertTrue(np.array_equal(contexts, expected_contexts))

    def test_negative_sampler_respects_forbidden(self) -> None:
        counts = np.array([10, 8, 6, 4, 2, 1], dtype=np.int64)
        sampler = w2v.NegativeSampler(counts=counts, seed=123)
        forbidden = np.array([[0, 1], [1, 2], [2, 3], [4, 5]], dtype=np.int64)
        negatives = sampler.sample(batch_size=4, negative_samples=20, forbidden=forbidden)

        for row in range(4):
            forbidden_set = set(forbidden[row].tolist())
            self.assertTrue(all(int(x) not in forbidden_set for x in negatives[row]))

    def test_batch_gradients_match_finite_difference(self) -> None:
        model = w2v.SkipGramNegativeSampling(vocab_size=8, embedding_dim=4, seed=7)

        center_ids = np.array([1, 2], dtype=np.int64)
        positive_ids = np.array([3, 4], dtype=np.int64)
        negative_ids = np.array([[5, 6], [0, 7]], dtype=np.int64)

        loss, grad_center, grad_positive, grad_negative = model.batch_loss_and_gradients(
            center_ids=center_ids,
            positive_ids=positive_ids,
            negative_ids=negative_ids,
        )
        self.assertTrue(np.isfinite(loss))

        eps = 1e-6

        def numerical_grad_input(word_id: int, dim: int) -> float:
            original = model.input_embeddings[word_id, dim]
            model.input_embeddings[word_id, dim] = original + eps
            loss_plus, *_ = model.batch_loss_and_gradients(center_ids, positive_ids, negative_ids)
            model.input_embeddings[word_id, dim] = original - eps
            loss_minus, *_ = model.batch_loss_and_gradients(center_ids, positive_ids, negative_ids)
            model.input_embeddings[word_id, dim] = original
            return float((loss_plus - loss_minus) / (2 * eps))

        def numerical_grad_output(word_id: int, dim: int) -> float:
            original = model.output_embeddings[word_id, dim]
            model.output_embeddings[word_id, dim] = original + eps
            loss_plus, *_ = model.batch_loss_and_gradients(center_ids, positive_ids, negative_ids)
            model.output_embeddings[word_id, dim] = original - eps
            loss_minus, *_ = model.batch_loss_and_gradients(center_ids, positive_ids, negative_ids)
            model.output_embeddings[word_id, dim] = original
            return float((loss_plus - loss_minus) / (2 * eps))

        # Choose ids that appear exactly once in this batch for direct comparison.
        self.assertAlmostEqual(numerical_grad_input(1, 0), float(grad_center[0, 0]), places=5)
        self.assertAlmostEqual(numerical_grad_output(3, 1), float(grad_positive[0, 1]), places=5)
        self.assertAlmostEqual(numerical_grad_output(5, 2), float(grad_negative[0, 0, 2]), places=5)

    def test_single_update_reduces_same_batch_loss(self) -> None:
        model = w2v.SkipGramNegativeSampling(vocab_size=10, embedding_dim=8, seed=21)
        center_ids = np.array([1, 2, 3, 4], dtype=np.int64)
        positive_ids = np.array([2, 3, 4, 5], dtype=np.int64)
        negative_ids = np.array(
            [
                [6, 7, 8],
                [7, 8, 9],
                [0, 6, 9],
                [0, 1, 8],
            ],
            dtype=np.int64,
        )

        loss_before, grad_center, grad_positive, grad_negative = model.batch_loss_and_gradients(
            center_ids, positive_ids, negative_ids
        )
        model.apply_gradients(
            center_ids,
            positive_ids,
            negative_ids,
            grad_center,
            grad_positive,
            grad_negative,
            learning_rate=0.5,
        )
        loss_after, *_ = model.batch_loss_and_gradients(center_ids, positive_ids, negative_ids)
        self.assertLess(loss_after, loss_before)


if __name__ == "__main__":
    unittest.main()
