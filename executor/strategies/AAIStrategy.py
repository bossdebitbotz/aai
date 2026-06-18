"""
AAIStrategy -- Minimal Freqtrade strategy for the AAI Trading Executor.

Accepts forced trades from the signal service via Freqtrade's REST API.
All entry/exit logic lives in the signal service; this strategy is a
pass-through that logs trade events and satisfies Freqtrade's interface.
"""

import logging

try:
    from freqtrade.strategy import IStrategy
except ImportError:
    # Stub for development without freqtrade installed
    class IStrategy:
        pass

try:
    import pandas as pd
    from pandas import DataFrame
except ImportError:
    DataFrame = None

logger = logging.getLogger(__name__)


class AAIStrategy(IStrategy):
    """Pass-through strategy that receives forced trades from the signal service."""

    INTERFACE_VERSION = 3

    # Timeframe -- must be set but irrelevant since we force trades externally
    timeframe = "1m"

    # Allow shorting (futures mode)
    can_short = True

    # Emergency stoploss
    stoploss = -0.10

    # Tight trailing stop — locks in profit, lets winners run
    trailing_stop = True
    trailing_stop_positive = 0.001      # trail by 0.1% from peak (tight)
    trailing_stop_positive_offset = 0.002  # activate once 0.2% profit reached
    trailing_only_offset_is_reached = True

    minimal_roi = {}  # disabled — exits via trailing stop or signal reversal

    # ------------------------------------------------------------------ #
    # No-op indicator / trend methods (signals come from the signal svc) #
    # ------------------------------------------------------------------ #

    def populate_indicators(self, dataframe: "DataFrame", metadata: dict) -> "DataFrame":
        """No indicators needed -- trades are forced externally."""
        return dataframe

    def populate_entry_trend(self, dataframe: "DataFrame", metadata: dict) -> "DataFrame":
        """No entry signals -- trades are forced via REST API."""
        return dataframe

    def populate_exit_trend(self, dataframe: "DataFrame", metadata: dict) -> "DataFrame":
        """No exit signals -- exits are forced via REST API."""
        return dataframe

    # ------------------------------------------------------------------ #
    # Leverage                                                            #
    # ------------------------------------------------------------------ #

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: str = None, side: str = None, **kwargs) -> float:
        """1x leverage — matches profitable backtest exactly."""
        return 1.0

    # ------------------------------------------------------------------ #
    # Confirmation hooks -- log forced trades for audit trail             #
    # ------------------------------------------------------------------ #

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:
        """Log and confirm every forced entry."""
        logger.info(
            "AAIStrategy confirm_trade_entry: pair=%s side=%s amount=%.6f rate=%.6f tag=%s",
            pair, side, amount, rate, entry_tag,
        )
        return True

    def confirm_trade_exit(
        self,
        pair: str,
        trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time,
        **kwargs,
    ) -> bool:
        """Log and confirm every forced exit."""
        logger.info(
            "AAIStrategy confirm_trade_exit: pair=%s amount=%.6f rate=%.6f reason=%s",
            pair, amount, rate, exit_reason,
        )
        return True
