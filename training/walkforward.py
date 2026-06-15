#!/usr/bin/env python3
"""Nested, model-retrained, rolling-window walk-forward harness for the
inventory + trailing-stop strategy.

Heavy steps (retrain, signal precompute) run on GPU/Colab; the pure pieces
(fold ranges, strategy sim, inner sweep, aggregation) are TDD'd locally and
operate on cached per-decision signals.

See docs/superpowers/specs/2026-06-15-trailing-stop-walkforward-design.md.
"""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    train_start: dt.datetime
    tune_start: dt.datetime
    train_end: dt.datetime      # == test_start (rolling boundary)
    test_start: dt.datetime
    test_end: dt.datetime

    @property
    def tune_end(self) -> dt.datetime:
        return self.train_end


def make_folds(data_start: dt.datetime, data_end: dt.datetime,
               train_days: int = 45, test_days: int = 8, tune_days: int = 7) -> list[Fold]:
    """Rolling fixed-width folds: train[t..t+train] (last tune_days held out for
    inner param selection) then test[t+train .. t+train+test]. Step = test_days.
    Returns only fully-contained folds (no overlap of test with train; never
    runs past data_end)."""
    folds: list[Fold] = []
    train_w = dt.timedelta(days=train_days)
    test_w = dt.timedelta(days=test_days)
    tune_w = dt.timedelta(days=tune_days)
    t = data_start
    while t + train_w + test_w <= data_end:
        train_start = t
        train_end = t + train_w
        f = Fold(train_start=train_start, tune_start=train_end - tune_w,
                 train_end=train_end, test_start=train_end, test_end=train_end + test_w)
        # leakage guards
        assert f.train_start < f.tune_start < f.train_end, "tune slice must sit inside train"
        assert f.test_start >= f.train_end, "test must not overlap train"
        folds.append(f)
        t = t + test_w
    return folds
