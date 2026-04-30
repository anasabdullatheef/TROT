"""
Arbitrage Calculator - Calculates spreads, fees, and net profit for all pairs
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from monitor import PriceData
import config


class OpportunityType(Enum):
    NONE = "none"
    LOG_ONLY = "log_only"  # 0.08% - 0.15% spread
    TRADEABLE = "tradeable"  # > 0.15% spread


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity between two exchanges"""
    timestamp: datetime
    buy_exchange: str
    sell_exchange: str
    buy_price: float  # Ask price on buy exchange
    sell_price: float  # Bid price on sell exchange
    gross_spread_pct: float  # Raw spread percentage
    buy_fee_pct: float
    sell_fee_pct: float
    total_fee_pct: float
    net_spread_pct: float  # After fees
    opportunity_type: OpportunityType
    estimated_profit_usd: float  # At $100 trade size
    pair_name: str


class ArbitrageCalculator:
    """Calculates arbitrage opportunities between exchange pairs"""

    def __init__(self):
        self.opportunities: List[ArbitrageOpportunity] = []

    def calculate_spread(
        self,
        buy_exchange: str,
        sell_exchange: str,
        buy_price: float,  # Ask price (what you pay to buy)
        sell_price: float  # Bid price (what you receive when selling)
    ) -> Tuple[float, float, float, float]:
        """
        Calculate spread and fees between two exchanges.

        Returns:
            gross_spread_pct: Raw spread percentage
            buy_fee_pct: Fee for buying
            sell_fee_pct: Fee for selling
            net_spread_pct: Net spread after all fees
        """
        # Gross spread: (sell_price - buy_price) / buy_price
        gross_spread_pct = (sell_price - buy_price) / buy_price

        # Get fees
        buy_fee_pct = config.EXCHANGE_FEES[buy_exchange]
        sell_fee_pct = config.EXCHANGE_FEES[sell_exchange]

        # Net spread after fees
        # When buying: you pay buy_price * (1 + buy_fee)
        # When selling: you receive sell_price * (1 - sell_fee)
        # Net profit = sell_price * (1 - sell_fee) - buy_price * (1 + buy_fee)
        # Net profit % = net_profit / buy_price
        net_spread_pct = gross_spread_pct - buy_fee_pct - sell_fee_pct

        return gross_spread_pct, buy_fee_pct, sell_fee_pct, net_spread_pct

    def find_opportunities(
        self,
        prices: Dict[str, PriceData]
    ) -> List[ArbitrageOpportunity]:
        """
        Find all arbitrage opportunities across exchange pairs.

        For each pair, check both directions:
        - Buy on exchange A, sell on exchange B
        - Buy on exchange B, sell on exchange A
        """
        opportunities = []

        for ex1, ex2 in config.EXCHANGE_PAIRS:
            if ex1 not in prices or ex2 not in prices:
                continue

            price1 = prices[ex1]
            price2 = prices[ex2]

            # Direction 1: Buy on ex1 (at ask), sell on ex2 (at bid)
            opp1 = self._check_direction(
                buy_exchange=ex1,
                sell_exchange=ex2,
                buy_price=price1.ask,
                sell_price=price2.bid
            )
            if opp1:
                opportunities.append(opp1)

            # Direction 2: Buy on ex2 (at ask), sell on ex1 (at bid)
            opp2 = self._check_direction(
                buy_exchange=ex2,
                sell_exchange=ex1,
                buy_price=price2.ask,
                sell_price=price1.bid
            )
            if opp2:
                opportunities.append(opp2)

        # Sort by net spread (highest first)
        opportunities.sort(key=lambda x: x.net_spread_pct, reverse=True)

        return opportunities

    def _check_direction(
        self,
        buy_exchange: str,
        sell_exchange: str,
        buy_price: float,
        sell_price: float
    ) -> Optional[ArbitrageOpportunity]:
        """Check if there's an opportunity in this direction"""

        gross_spread, buy_fee, sell_fee, net_spread = self.calculate_spread(
            buy_exchange, sell_exchange, buy_price, sell_price
        )

        # Only log if gross spread is positive
        if gross_spread <= 0:
            return None

        # Determine opportunity type
        if net_spread >= config.MIN_SPREAD_TO_TRADE:
            opp_type = OpportunityType.TRADEABLE
        elif net_spread >= config.LOG_ONLY_SPREAD_MIN:
            opp_type = OpportunityType.LOG_ONLY
        else:
            return None  # Too small to even log

        # Estimate profit at $100 trade size
        estimated_profit = config.TRADE_SIZE_USD * net_spread

        return ArbitrageOpportunity(
            timestamp=datetime.now(),
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            buy_price=buy_price,
            sell_price=sell_price,
            gross_spread_pct=gross_spread,
            buy_fee_pct=buy_fee,
            sell_fee_pct=sell_fee,
            total_fee_pct=buy_fee + sell_fee,
            net_spread_pct=net_spread,
            opportunity_type=opp_type,
            estimated_profit_usd=estimated_profit,
            pair_name=f"{buy_exchange}->{sell_exchange}"
        )

    def get_best_opportunity(
        self,
        prices: Dict[str, PriceData]
    ) -> Optional[ArbitrageOpportunity]:
        """Get the best tradeable opportunity"""
        opportunities = self.find_opportunities(prices)

        for opp in opportunities:
            if opp.opportunity_type == OpportunityType.TRADEABLE:
                return opp

        return None

    def format_opportunity(self, opp: ArbitrageOpportunity) -> str:
        """Format opportunity for display"""
        status = "TRADE" if opp.opportunity_type == OpportunityType.TRADEABLE else "LOG"

        return (
            f"[{status}] {opp.pair_name}: "
            f"Buy ${opp.buy_price:,.2f} -> Sell ${opp.sell_price:,.2f} | "
            f"Gross: {opp.gross_spread_pct*100:.3f}% | "
            f"Fees: {opp.total_fee_pct*100:.3f}% | "
            f"Net: {opp.net_spread_pct*100:.3f}% | "
            f"Est. Profit: ${opp.estimated_profit_usd:.2f}"
        )


def test_calculator():
    """Test the arbitrage calculator with mock data"""
    from datetime import datetime

    # Mock price data
    prices = {
        "binance": PriceData(
            exchange="binance",
            symbol="BTC/USDT",
            bid=95000.0,
            ask=95010.0,
            mid=95005.0,
            timestamp=datetime.now(),
            latency_ms=50
        ),
        "bybit": PriceData(
            exchange="bybit",
            symbol="BTC/USDT",
            bid=95200.0,  # Higher bid - potential to sell here
            ask=95210.0,
            mid=95205.0,
            timestamp=datetime.now(),
            latency_ms=60
        ),
        "okx": PriceData(
            exchange="okx",
            symbol="BTC/USDT",
            bid=95050.0,
            ask=95060.0,
            mid=95055.0,
            timestamp=datetime.now(),
            latency_ms=55
        )
    }

    calc = ArbitrageCalculator()
    opportunities = calc.find_opportunities(prices)

    print("Arbitrage Opportunities:")
    print("=" * 80)

    for opp in opportunities:
        print(calc.format_opportunity(opp))

    print(f"\nTotal opportunities found: {len(opportunities)}")
    tradeable = [o for o in opportunities if o.opportunity_type == OpportunityType.TRADEABLE]
    print(f"Tradeable opportunities: {len(tradeable)}")


if __name__ == "__main__":
    test_calculator()
