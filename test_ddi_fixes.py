#!/usr/bin/env python3
"""
Unit test for DDI loss fixes — runs without GPU, data, or LLaMA model.
Tests:
  1. DDI loss normalization (magnitude check)
  2. Gradient clipping prevents NaN
  3. metric_report NaN guard
"""

import torch
import torch.nn as nn
import numpy as np
import sys

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} — {detail}")


def test_ddi_loss_normalization():
    """DDI loss should be ~O(1), not ~O(num_pairs)"""
    print("\n[1] DDI Loss Normalization")

    V = 130  # med vocab size
    bs = 4

    # Create fake DDI adjacency matrix with ~2000 pairs
    ddi_adj = torch.zeros(V, V)
    num_pairs = 2000
    indices = torch.randint(0, V, (num_pairs, 2))
    for i, j in indices:
        ddi_adj[i, j] = 1
        ddi_adj[j, i] = 1
    actual_pairs = ddi_adj.sum().item()

    # Simulate random init: sigmoid(0) ≈ 0.5
    output = torch.zeros(bs, V)
    probs = torch.sigmoid(output)  # all ~0.5

    # OLD: unnormalized
    old_loss = torch.bmm(
        torch.bmm(
            probs.unsqueeze(1),
            ddi_adj.unsqueeze(0).expand(bs, -1, -1)
        ),
        probs.unsqueeze(2)
    ).squeeze(-1).squeeze(-1)

    # NEW: normalized
    num_ddi_pairs = ddi_adj.sum().clamp(min=1.0)
    new_loss = old_loss / num_ddi_pairs

    old_mean = old_loss.mean().item()
    new_mean = new_loss.mean().item()

    test("Old DDI loss is large (>10)", old_mean > 10,
         f"old_loss={old_mean:.1f}")
    test("New DDI loss is small (<1)", new_mean < 1.0,
         f"new_loss={new_mean:.4f}")
    test("New DDI loss ~ 0.25 (expected for sigmoid(0))",
         0.1 < new_mean < 0.5,
         f"new_loss={new_mean:.4f}, expected ~0.25")

    print(f"  Info: DDI pairs={actual_pairs:.0f}, old_loss={old_mean:.1f}, new_loss={new_mean:.4f}")


def test_gradient_clipping_prevents_nan():
    """Large loss should not cause NaN weights after clipping"""
    print("\n[2] Gradient Clipping Prevents NaN")

    # Simple model
    model = nn.Linear(64, 130)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    x = torch.randn(4, 64)

    # Simulate huge DDI loss (no normalization)
    for step in range(100):
        optimizer.zero_grad()
        output = model(x)

        # Fake huge loss
        loss = (output ** 2).sum() * 1000
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    # Check weights not NaN
    has_nan = any(torch.isnan(p).any().item() for p in model.parameters())
    test("Weights not NaN after 100 steps with huge loss + clipping", not has_nan)

    # Without clipping — should explode
    model2 = nn.Linear(64, 130)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=2e-4)

    for step in range(100):
        optimizer2.zero_grad()
        output = model2(x)
        loss = (output ** 2).sum() * 1000
        loss.backward()
        # NO clipping
        optimizer2.step()

    has_nan2 = any(torch.isnan(p).any().item() for p in model2.parameters())
    test("Without clipping, weights become NaN (confirms bug)", has_nan2,
         "weights stayed finite — test may need more steps")


def test_metric_report_nan_guard():
    """metric_report should not crash on NaN predictions"""
    print("\n[3] NaN Guard in metric_report")

    # Import the function
    sys.path.insert(0, '.')
    from utils.utils import metric_report
    import logging
    logger = logging.getLogger("test")
    logger.addHandler(logging.StreamHandler())

    V = 130
    n_samples = 10

    # Normal case
    y_pred_clean = np.random.rand(n_samples, V)
    y_true = np.zeros((n_samples, V))
    for i in range(n_samples):
        indices = np.random.choice(V, 5, replace=False)
        y_true[i, indices] = 1

    try:
        result = metric_report(logger, y_pred_clean.copy(), y_true.copy())
        test("Normal predictions — no crash", True)
    except Exception as e:
        test("Normal predictions — no crash", False, str(e))

    # NaN case
    y_pred_nan = np.random.rand(n_samples, V)
    y_pred_nan[0, :] = np.nan  # first sample all NaN
    y_pred_nan[5, 10] = np.nan  # partial NaN

    try:
        result = metric_report(logger, y_pred_nan.copy(), y_true.copy())
        test("NaN predictions — no crash (guard works)", True)
    except Exception as e:
        test("NaN predictions — no crash (guard works)", False, str(e))

    # All NaN case
    y_pred_all_nan = np.full((n_samples, V), np.nan)

    try:
        result = metric_report(logger, y_pred_all_nan.copy(), y_true.copy())
        test("All-NaN predictions — no crash", True)
    except Exception as e:
        test("All-NaN predictions — no crash", False, str(e))


def test_squeeze_batch_size_1():
    """DDI loss squeeze should work with batch_size=1"""
    print("\n[4] DDI Loss Squeeze with batch_size=1")

    V = 130
    ddi_adj = torch.randint(0, 2, (V, V)).float()

    for bs in [1, 4, 8]:
        output = torch.randn(bs, V)
        probs = torch.sigmoid(output)

        ddi_loss = torch.bmm(
            torch.bmm(
                probs.unsqueeze(1),
                ddi_adj.unsqueeze(0).expand(bs, -1, -1)
            ),
            probs.unsqueeze(2)
        ).squeeze(-1).squeeze(-1)  # safe squeeze

        num_ddi_pairs = ddi_adj.sum().clamp(min=1.0)
        ddi_loss = ddi_loss / num_ddi_pairs

        test(f"batch_size={bs}: shape={ddi_loss.shape}, no NaN",
             ddi_loss.shape == (bs,) and not torch.isnan(ddi_loss).any().item(),
             f"shape={ddi_loss.shape}")


def _count_ddi_pairs_binary(pred_bin, ddi_adj):
    drugs = np.where(pred_bin == 1)[0].tolist()
    cnt = 0
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            a, b = drugs[i], drugs[j]
            if ddi_adj[a, b] == 1 or ddi_adj[b, a] == 1:
                cnt += 1
    return cnt


def test_posthoc_ddi_budget():
    """Budgeted post-hoc should reduce DDI without forcing it to zero."""
    print("\n[5] Post-hoc DDI Budget (Non-zero allowed)")

    sys.path.insert(0, ".")
    from utils.utils import apply_ddi_constraints_budget

    V = 6
    ddi_adj = np.zeros((V, V), dtype=np.int32)
    # Triangle DDI among 0,1,2
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        ddi_adj[a, b] = 1
        ddi_adj[b, a] = 1

    # One sample: top probs for 0,1,2,3 above threshold=0.5
    y_prob = np.array([[0.9, 0.85, 0.8, 0.7, 0.2, 0.1]])

    # Case A: budget is loose => keep initial set, DDI remains (not forced to 0)
    pred_loose = apply_ddi_constraints_budget(
        y_prob, ddi_adj, threshold=0.5, target_ddi_rate=0.5, min_keep=1, refill=False
    )
    ddi_pairs_loose = _count_ddi_pairs_binary(pred_loose[0], ddi_adj)
    test(
        "Loose budget keeps some DDI (not forced to 0)",
        ddi_pairs_loose > 0,
        f"ddi_pairs={ddi_pairs_loose}",
    )

    # Case B: tighter budget => reduce DDI pairs <= budget
    pred_tight = apply_ddi_constraints_budget(
        y_prob, ddi_adj, threshold=0.5, target_ddi_rate=0.1, min_keep=2, refill=False
    )
    # initial size n0=4 => total_pairs=6 => budget_pairs=floor(0.1*6)=0
    ddi_pairs_tight = _count_ddi_pairs_binary(pred_tight[0], ddi_adj)
    test(
        "Tight budget reduces DDI pairs to <= budget",
        ddi_pairs_tight == 0,
        f"ddi_pairs={ddi_pairs_tight}",
    )
    test("Respects min_keep=2", pred_tight[0].sum() >= 2, f"kept={pred_tight[0].sum()}")


if __name__ == "__main__":
    print("=" * 50)
    print("DDI Fix Verification Tests")
    print("=" * 50)

    test_ddi_loss_normalization()
    test_gradient_clipping_prevents_nan()
    test_metric_report_nan_guard()
    test_squeeze_batch_size_1()
    test_posthoc_ddi_budget()

    print("\n" + "=" * 50)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 50)

    sys.exit(0 if FAIL == 0 else 1)
