"""Model evaluation: Brier Score, BSS, reliability diagrams, per-bucket metrics."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.common.logging import get_logger

logger = get_logger(__name__)


# Moneyness bucket boundaries (log_moneyness ranges)
MONEYNESS_BUCKETS = {
    "deep_itm": (0.03, float("inf")),
    "near_itm": (0.01, 0.03),
    "atm": (-0.01, 0.01),
    "near_otm": (-0.03, -0.01),
    "deep_otm": (float("-inf"), -0.03),
}


@dataclass
class BucketMetrics:
    """Per-moneyness-bucket evaluation metrics."""

    bucket: str
    n_samples: int
    brier_score: float
    bss: float  # vs climatology
    log_loss: float
    base_rate: float  # fraction of positives (climatology)


@dataclass
class EvalResult:
    """Full evaluation result for a fold."""

    bss_overall: float
    brier_overall: float
    log_loss_overall: float
    bucket_metrics: list[BucketMetrics] = field(default_factory=list)
    reliability_bins: list[dict[str, float]] = field(default_factory=list)  # 10-bin reliability
    sharpness: list[float] = field(default_factory=list)  # prediction distribution
    n_samples: int = 0


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean Brier Score."""
    return float(np.mean((y_pred - y_true) ** 2))


def brier_skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    BSS = 1 - (Brier_model / Brier_climatology).

    Climatology baseline = always predict the base rate.
    BSS > 0 means model beats baseline.
    """
    bs_model = brier_score(y_true, y_pred)
    base_rate = float(y_true.mean()) if len(y_true) > 0 else 0.5
    bs_clim = brier_score(y_true, np.full_like(y_pred, base_rate))
    if bs_clim < 1e-12:
        return 0.0
    return 1.0 - bs_model / bs_clim


def log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    """Compute mean log loss."""
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)))


def reliability_diagram_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """
    Compute reliability diagram data (predicted prob vs observed freq).

    Args:
        y_true: Binary labels
        y_pred: Calibrated probabilities
        n_bins: Number of bins

    Returns:
        List of dicts with keys: bin_center, mean_pred, frac_pos, count
    """
    bins = np.linspace(0, 1, n_bins + 1)
    result = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_pred >= lo) & (y_pred < hi)
        if i == n_bins - 1:
            mask = (y_pred >= lo) & (y_pred <= hi)

        count = int(mask.sum())
        if count == 0:
            continue

        result.append({
            "bin_center": float((lo + hi) / 2),
            "mean_pred": float(y_pred[mask].mean()),
            "frac_pos": float(y_true[mask].mean()),
            "count": count,
        })

    return result


def evaluate(
    y_true: np.ndarray,
    y_pred_calibrated: np.ndarray,
    moneyness: np.ndarray,
) -> EvalResult:
    """
    Full evaluation of model predictions.

    Args:
        y_true: Binary labels
        y_pred_calibrated: Calibrated model probabilities
        moneyness: log(spot/strike) per sample (used for bucket stratification)

    Returns:
        EvalResult with overall and per-bucket metrics
    """
    if len(y_true) == 0:
        return EvalResult(bss_overall=0.0, brier_overall=1.0, log_loss_overall=1.0)

    bss_overall = brier_skill_score(y_true, y_pred_calibrated)
    brier_overall = brier_score(y_true, y_pred_calibrated)
    ll_overall = log_loss(y_true, y_pred_calibrated)

    # Per-bucket metrics
    bucket_metrics = []
    for bucket_name, (lo, hi) in MONEYNESS_BUCKETS.items():
        mask = (moneyness >= lo) & (moneyness < hi)
        if mask.sum() < 5:
            continue

        bucket_y = y_true[mask]
        bucket_pred = y_pred_calibrated[mask]

        bucket_metrics.append(BucketMetrics(
            bucket=bucket_name,
            n_samples=int(mask.sum()),
            brier_score=brier_score(bucket_y, bucket_pred),
            bss=brier_skill_score(bucket_y, bucket_pred),
            log_loss=log_loss(bucket_y, bucket_pred),
            base_rate=float(bucket_y.mean()),
        ))

    reliability = reliability_diagram_data(y_true, y_pred_calibrated)
    sharpness = y_pred_calibrated.tolist()

    return EvalResult(
        bss_overall=bss_overall,
        brier_overall=brier_overall,
        log_loss_overall=ll_overall,
        bucket_metrics=bucket_metrics,
        reliability_bins=reliability,
        sharpness=sharpness,
        n_samples=len(y_true),
    )


def print_eval_summary(result: EvalResult) -> None:
    """Print a rich summary table to stdout."""
    try:
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(title=f"Model Evaluation (n={result.n_samples})")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        bss_color = "green" if result.bss_overall > 0 else "red"
        table.add_row("BSS (overall)", f"[{bss_color}]{result.bss_overall:.4f}[/{bss_color}]")
        table.add_row("Brier Score", f"{result.brier_overall:.4f}")
        table.add_row("Log Loss", f"{result.log_loss_overall:.4f}")

        console.print(table)

        if result.bucket_metrics:
            bucket_table = Table(title="Per-Bucket Metrics")
            bucket_table.add_column("Bucket", style="cyan")
            bucket_table.add_column("N", justify="right")
            bucket_table.add_column("BSS", justify="right")
            bucket_table.add_column("Brier", justify="right")
            bucket_table.add_column("Base Rate", justify="right")

            for bm in result.bucket_metrics:
                bss_c = "green" if bm.bss > 0 else "red"
                bucket_table.add_row(
                    bm.bucket,
                    str(bm.n_samples),
                    f"[{bss_c}]{bm.bss:.4f}[/{bss_c}]",
                    f"{bm.brier_score:.4f}",
                    f"{bm.base_rate:.3f}",
                )

            console.print(bucket_table)

    except ImportError:
        # Fallback if rich not available
        print(f"BSS={result.bss_overall:.4f} | Brier={result.brier_overall:.4f} | LogLoss={result.log_loss_overall:.4f} | N={result.n_samples}")
        for bm in result.bucket_metrics:
            print(f"  {bm.bucket}: BSS={bm.bss:.4f} N={bm.n_samples}")
