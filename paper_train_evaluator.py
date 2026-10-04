"""Training rollout adapter using the canonical per-car inference path.

The old batched VAE/RNN path gave different closed-loop scores from the
single-car evaluator, even with identical images and latent RNG seeds. CMA-ES
must rank candidates with the same inference path used for final testing.
"""

import numpy as np

from paper_controller_evaluation import PaperCanonicalEvaluator


class Evaluator:
    def __init__(self, count, vision, memory, device):
        self.count = count
        self.canonical = PaperCanonicalEvaluator(count, vision, memory, device)

    def close(self):
        self.canonical.close()

    def evaluate(self, solutions, seeds, max_steps=1000, latent_seeds=None):
        if len(solutions) != self.count or len(seeds) != self.count:
            raise ValueError("One solution and track seed required per lane")
        if latent_seeds is None:
            latent_seeds = seeds
        if len(latent_seeds) != self.count:
            raise ValueError("One latent seed required per lane")
        rows = self.canonical.evaluate(solutions, seeds, latent_seeds,
                                       max_steps=max_steps)
        returns = np.array([row["episode_return"] for row in rows], dtype=np.float64)
        steps = np.array([row["steps"] for row in rows], dtype=np.int64)
        return returns, steps
