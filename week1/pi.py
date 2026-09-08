import random


def estimate_pi(n, seed):
    rng = random.Random(seed)
    inside = sum(rng.random() ** 2 + rng.random() ** 2 <= 1 for _ in range(n))
    return 4 * inside / n
