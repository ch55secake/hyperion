from dataclasses import dataclass, field


@dataclass
class HyperionConfig:
    """Central configuration for all tunable Hyperion parameters.

    Attributes:
        period: Historical data period to download (e.g. '1y', '2y', '5y').
        intervals: List of data intervals to download (e.g. ['1d', '1h']).
        test_size: Fraction of data reserved for final held-out testing (0 < test_size < 1).
        val_size: Fraction of data reserved for hyperparameter optimisation validation
            (0 < val_size < 1 - test_size).  The three splits are laid out temporally as
            train | val | test so that no future data leaks into earlier sets.
        target_days: Number of forward days used to compute the return target.
        n_trials: Number of Optuna hyperparameter-search trials per model.
        r2_save_threshold: Minimum R² required to persist a model to disk.
        r2_invalid_threshold: R² below this value causes the model to be saved as invalid.
        initial_capital: Starting cash balance for trading simulations.
        transaction_cost: Proportional transaction cost applied on each buy/sell (e.g. 0.001 = 0.1 %).
    """

    period: str = "2y"
    intervals: list[str] = field(default_factory=lambda: ["1d", "1h"])
    test_size: float = 0.2
    val_size: float = 0.1
    target_days: int = 10
    n_trials: int = 1000
    r2_save_threshold: float = 0.0012
    r2_invalid_threshold: float = -0.3
    initial_capital: float = 10000
    transaction_cost: float = 0.001
