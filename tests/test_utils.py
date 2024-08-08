import numpy as np


def cosine_similarity(x:np.array, y:np.array) -> float:
    """Calculate the cosine similarity between two vectors.
    Parameters
    ----------
    x: np.array
    y: np.array
    Returns
    -------
    float
    """
    x = x.flatten()
    y = y.flatten()
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))