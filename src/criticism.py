from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from torch import Tensor

def c2st(
    X: Tensor,
    Y: Tensor,
    seed: int = 1,
    n_folds: int = 5,
    metric: str = "accuracy",
    classifier_kwargs: Optional[Dict[str, Any]] = None,
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    verbosity: int = 0,
) -> Tensor:

    clf_class = RandomForestClassifier
    clf_kwargs = classifier_kwargs or {}  # use sklearn defaults

    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        # Set std to 1 if it is close to zero.
        X_std[X_std < 1e-14] = 1
        assert not torch.any(torch.isnan(X_mean)), "X_mean contains NaNs"
        assert not torch.any(torch.isnan(X_std)), "X_std contains NaNs"
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * torch.randn(X.shape)
        Y += noise_scale * torch.randn(Y.shape)

    clf = clf_class(random_state=seed, **clf_kwargs)

    # prepare data, convert to numpy
    data = np.concatenate((X.cpu().numpy(), Y.cpu().numpy()))
    # labels
    target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],))))

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(
        clf, data, target, cv=shuffle, scoring=metric, verbose=verbosity
    )

    return torch.from_numpy(scores).mean()


def z_test(
    log_probs: Tensor, log_prob_xo: float, alpha: float = 0.05
):
    """Perform a hypothesis test to check if log_prob_xo is unusually low.

    The lo_prob_xo is compared to the given log probabilities from the distribution.

    Args:
        log_probs: array-like, log probabilities of known samples
        log_prob_xo: float, log probability of the test sample
        alpha: significance level (default 0.05)

    Returns:
        - p_value: float, proportion of log_probs below log_prob_xo
        - reject_H0: bool, whether to reject H0 at the given alpha level
    """
    # Compute empirical CDF value (proportion of samples with lower log prob)
    p_value = (log_probs <= log_prob_xo).float().mean()

    # Reject H0 if p_value is below the significance level
    reject_H0 = p_value < alpha

    return p_value, reject_H0