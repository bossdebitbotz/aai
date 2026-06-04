"""Confidence calibration via temperature scaling (per-horizon), plus ECE."""
import torch
import torch.nn as nn


def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Return temperature-scaled softmax probabilities. Argmax is invariant to T>0."""
    return torch.softmax(logits / T, dim=-1)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Fit a single scalar temperature on (logits, labels) by minimizing NLL. Returns T>0."""
    logits = logits.detach()
    log_T = torch.zeros(1, requires_grad=True)  # optimize log T to keep T>0
    opt = torch.optim.LBFGS([log_T], lr=0.05, max_iter=max_iter)
    ce = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(logits / log_T.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp().item())


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Standard ECE over confidence bins."""
    conf, preds = probs.max(dim=-1)
    acc = (preds == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            ece += m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()
    return float(ece.item())


def reliability_bins(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
    """Return (bin_centers, bin_acc, bin_conf, bin_count) for a reliability diagram."""
    conf, preds = probs.max(dim=-1)
    acc = (preds == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    centers, b_acc, b_conf, counts = [], [], [], []
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        centers.append(((edges[i] + edges[i + 1]) / 2).item())
        counts.append(int(m.sum().item()))
        b_acc.append(float(acc[m].mean().item()) if m.any() else float("nan"))
        b_conf.append(float(conf[m].mean().item()) if m.any() else float("nan"))
    return centers, b_acc, b_conf, counts
