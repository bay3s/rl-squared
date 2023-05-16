from dataclasses import dataclass


@dataclass
class PPOUpdate:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    pass
