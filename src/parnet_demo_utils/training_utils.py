"""Training utilities for PARNET eCLIP demo notebooks."""

from __future__ import annotations

import lightning.pytorch as pl


class MetricHistory(pl.Callback):
    """Accumulate per-epoch trainer metrics into a plain list of dicts.

    Attach to a ``Trainer`` to collect metrics after each training epoch::

        from parnet_demo_utils import MetricHistory

        history = MetricHistory()
        trainer = pl.Trainer(..., callbacks=[history])
        trainer.fit(model, train_loader, val_loader)

        import pandas as pd
        df = pd.DataFrame(history.history)

    Attributes:
        history: List of ``{metric_name: float}`` dicts, one per completed
            training epoch.
    """

    def __init__(self) -> None:
        self.history: list[dict[str, float]] = []

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self.history.append(
            {k: float(v) for k, v in trainer.callback_metrics.items()}
        )
