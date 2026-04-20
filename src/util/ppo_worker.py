"""Pre-started PPO worker process for macOS dual-libomp isolation.

Background
----------
On macOS (Apple Silicon), LightGBM bundles Homebrew's ``libomp.dylib``
(resolved via ``/opt/homebrew/opt/libomp/lib``), while PyTorch bundles its
own build of ``libomp.dylib`` from ``/opt/llvm-openmp``.  When both are
loaded in the same OS process they create **named POSIX semaphores** in the
global kernel namespace.  Whichever library initialises second finds those
semaphores in a corrupted/leaked state and either deadlocks or crashes with
SIGSEGV (-11).

In the full pipeline, LightGBM training runs first, initialising its libomp
and locking those semaphores.  Any subprocess spawned *after* that point
inherits the corrupted semaphore state in the kernel and crashes when
PyTorch tries to acquire the same names on its own initialisation.

The fix
-------
Start the PPO worker subprocess **before any ML library is imported** in the
parent process.  At that point the global POSIX semaphore namespace is clean,
so the subprocess can safely initialise torch's libomp on its own without
conflict.  The subprocess then stays alive for the entire program lifetime
and processes work requests via multiprocessing queues.

Usage
-----
In ``main.py``, call :func:`initialize` as the very first action — before
importing ``src.pipeline`` or anything else that pulls in LightGBM or torch::

    import src.ppo_worker as _ppo_worker
    _ppo_worker.initialize()

    from src.pipeline.stacked_pipeline import StackedModelTrainingPipeline
    ...

From ``stacked_pipeline.simulate()`` use :func:`submit_work` / :func:`get_results`
to delegate work to the already-running subprocess::

    import src.ppo_worker as _ppo_worker
    _ppo_worker.submit_work(ticker_batches, strategy_key, initial_capital, tc)
    all_batch_results = _ppo_worker.get_results()

IMPORTANT: This module must have **zero ML imports at module level**.  Only
the standard-library ``multiprocessing`` import is safe here.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.process import BaseProcess as _BaseProcess

# ---------------------------------------------------------------------------
# Module-level singleton state — populated by initialize()
# ---------------------------------------------------------------------------
_worker_proc: _BaseProcess | None = None
_work_q: mp.Queue | None = None
_result_q: mp.Queue | None = None


# ---------------------------------------------------------------------------
# Worker subprocess entry-point
# ---------------------------------------------------------------------------


def _worker_loop(ready_q: mp.Queue, work_q: mp.Queue, result_q: mp.Queue) -> None:
    """Entry point for the pre-started worker subprocess.

    All ML imports (torch, StrategyRegistry, TradingSimulator) happen inside
    this function body, so they **never** touch the parent process's OpenMP
    state.  After imports complete a ``"ready"`` sentinel is sent on
    *ready_q* so the parent knows torch has been fully initialised.

    The loop then blocks on *work_q*, processing one work item at a time
    until the ``None`` sentinel triggers a clean shutdown.
    """
    # --- lazy ML imports (torch initialises here, in a clean process) -------
    # Importing strategy_registry triggers src/simulation/strategy/__init__.py
    # which registers all strategies (including PPO) and loads torch.
    from src.simulation.strategy.strategy_registry import StrategyRegistry  # noqa: F401
    import src.simulation.strategy  # noqa: F401 — ensures all strategies are registered
    from src.simulation.trading_simulator import TradingSimulator

    # Signal to parent that torch is fully initialised and we are ready.
    ready_q.put("ready")

    # ------------------------------------------------------------------ #
    # Main work loop                                                       #
    # ------------------------------------------------------------------ #
    while True:
        item = work_q.get()
        if item is None:
            # Sentinel — clean shutdown requested.
            break

        ticker_batches, strategy_key, initial_capital, transaction_cost = item

        all_batch_results: list[list[tuple[str, dict | None, str | None]]] = []

        for batch in ticker_batches:
            batch_results: list[tuple[str, dict | None, str | None]] = []
            for ticker_df in batch:
                symbol = str(ticker_df["symbol"].iloc[0])
                try:
                    strategy_class = StrategyRegistry.get(strategy_key)

                    if len(ticker_df) < strategy_class.get_minimum_data_points():
                        batch_results.append((symbol, None, f"insufficient data ({len(ticker_df)} rows)"))
                        continue

                    additional_data = strategy_class.get_extra_params(ticker_df.set_index("date")["price"])
                    simulator = TradingSimulator(
                        initial_capital=initial_capital,
                        transaction_cost=transaction_cost,
                    )
                    strategy = StrategyRegistry.create(
                        name=strategy_key,
                        simulator=simulator,
                        capital=initial_capital,
                        **additional_data,
                    )
                    ticker_data = ticker_df.reset_index(drop=True)
                    results = simulator.simulate(
                        predictions=ticker_data["prediction"],
                        actual_returns=ticker_data["actual_return"],
                        prices=ticker_data["price"],
                        dates=ticker_data["date"],
                        strategy=strategy,
                        threshold="auto",
                    )
                    batch_results.append((symbol, results, None))
                except Exception as exc:
                    batch_results.append((symbol, None, f"error: {exc}"))

            all_batch_results.append(batch_results)

        result_q.put(all_batch_results)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def initialize() -> None:
    """Start the PPO worker subprocess and block until it is ready.

    Must be called **before** any import that loads LightGBM or PyTorch.
    Calling this function more than once is safe (subsequent calls are no-ops).
    """
    global _worker_proc, _work_q, _result_q

    if _worker_proc is not None:
        return  # already initialised

    ctx = mp.get_context("spawn")
    ready_q: mp.Queue = ctx.Queue()
    _work_q = ctx.Queue()
    _result_q = ctx.Queue()

    _worker_proc = ctx.Process(
        target=_worker_loop,
        args=(ready_q, _work_q, _result_q),
        daemon=True,
        name="ppo-worker",
    )
    _worker_proc.start()
    # Block until the subprocess has finished importing torch and all strategies.
    ready_q.get()


def submit_work(
    ticker_batches: list,
    strategy_key: str,
    initial_capital: int,
    transaction_cost: float,
) -> None:
    """Send a batch of simulation work to the worker subprocess.

    :param ticker_batches: List of ticker-batch lists (each batch is a
        ``list[pd.DataFrame]``).
    :param strategy_key: Registered strategy name (e.g. ``"ppo"``).
    :param initial_capital: Starting capital per simulation.
    :param transaction_cost: Proportional cost per trade.
    :raises RuntimeError: If :func:`initialize` has not been called.
    """
    if _work_q is None:
        raise RuntimeError(
            "ppo_worker has not been initialised.  Call ppo_worker.initialize() "
            "before any ML library is imported (see src/ppo_worker.py for details)."
        )
    _work_q.put((ticker_batches, strategy_key, initial_capital, transaction_cost))


def get_results() -> list:
    """Retrieve results for the most recently submitted work item.

    Blocks until the worker subprocess places results on the result queue.

    :returns: ``list[list[tuple[str, dict | None, str | None]]]`` — one inner
        list per batch, each entry being ``(symbol, results, skip_reason)``.
    :raises RuntimeError: If :func:`initialize` has not been called.
    """
    if _result_q is None:
        raise RuntimeError("ppo_worker has not been initialised.  Call ppo_worker.initialize() first.")
    return _result_q.get()


def exitcode() -> int | None:
    """Return the worker process exit code, or ``None`` if still running."""
    if _worker_proc is None:
        return None
    return _worker_proc.exitcode


def shutdown() -> None:
    """Gracefully shut down the worker subprocess.

    Sends the ``None`` sentinel, waits up to 30 seconds for a clean exit,
    then kills the process if it is still alive.
    """
    global _worker_proc, _work_q, _result_q

    if _work_q is not None:
        _work_q.put(None)  # sentinel

    if _worker_proc is not None:
        _worker_proc.join(timeout=30)
        if _worker_proc.is_alive():
            _worker_proc.kill()

    _worker_proc = None
    _work_q = None
    _result_q = None
