import torch

class ZerothOrderOptimizerScalar:
    def __init__(self, lr, eps, beta, min_beta):
        self.lr = lr
        self.eps = eps  # Perturbation size
        self.beta = beta  # Parameter to optimize
        self.min_beta = min_beta  # Minimum parameter value

    def step(self, score_fn, args):
        # Ensure backward perturbation is not less than minimum (numerical stability)
        backward_min = max(self.beta - self.eps, 1e-8)
        # Forward perturbation evaluation (Eq. 10)
        forward_score = score_fn(self.beta + self.eps, **args)
        # Backward perturbation evaluation (Eq. 11)
        backward_score = score_fn(backward_min, **args)
        # Gradient estimate (Eq. 12)
        grad_estimate = (forward_score - backward_score) / (2 * self.eps)
        # Parameter update (Eq. 13)
        self.beta = self.beta - self.lr * grad_estimate
        self.beta = max(self.beta, self.min_beta)
        return {
            "beta": self.beta,
            "f_score": forward_score,
            "b_score": backward_score,
            "grad_est": grad_estimate,
        }

class ZerothOrderOptimizerVector:
    def __init__(self, lr, eps, beta, min_beta):
        self.lr = lr
        self.eps = eps
        self.beta = beta
        self.min_beta = min_beta

    def step(self, score_fn, args):
        # Generate random unit direction vector
        u = torch.randn_like(self.beta)
        u = u / u.norm()
        # Perturbation evaluation in random direction
        forward_score = score_fn(self.beta + self.eps * u, **args)
        backward_score = score_fn(self.beta - self.eps * u, **args)
        # Directional derivative estimate (variant of Eq. 14)
        grad_estimate = (forward_score - backward_score) / (2 * self.eps) * u
        # Parameter update
        self.beta = self.beta - self.lr * grad_estimate
        self.beta = torch.max(self.beta, torch.tensor(self.min_beta))  # Per-dimension constraint
        return self.beta, forward_score, backward_score, grad_estimate