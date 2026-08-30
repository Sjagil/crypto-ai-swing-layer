from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int


def expanding_folds(
    n: int,
    min_train: int,
    validation: int,
    test: int,
    purge: int,
    step: int | None = None,
) -> list[Fold]:
    step = step or test
    folds: list[Fold] = []
    train_end = min_train
    while True:
        val_start = train_end + purge
        val_end = val_start + validation
        test_start = val_end + purge
        test_end = test_start + test
        if test_end > n:
            break
        folds.append(Fold(0, train_end, val_start, val_end, test_start, test_end))
        train_end += step
    return folds
