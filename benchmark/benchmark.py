# Copyright 2019 QuantRocket LLC - All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from moonshot import Moonshot

class EqualWeightedIndex(Moonshot):
    """
    Strategy that buys all stocks in the universe, optionally filtering by dollar volume, 
    and weights equally.
    
    Examples
    --------
    Create equal-weighted index of Canadian stocks with average dollar volume >= 1M CAD: 
    
    >>> from codeload.benchmark import EqualWeightedIndex
    >>>
    >>> class CanadaEqualWeightedIndex(EqualWeightedIndex):
    >>>
    >>>     CODE = "canada-benchmark"
    >>>     DB = "canada-stk-1d"
    >>>     MIN_DOLLAR_VOLUME = 1e6
    """
    CODE = None
    DB = None
    DB_FIELDS = ["Close", "Volume"]
    UNIVERSES = None
    EXCLUDE_UNIVERSES = None

    # Optionally filter universe by min dollar volume, or dollar volume rank 
    # (dollar volume rank takes precedence if both defined)
    MIN_DOLLAR_VOLUME = 0
    DOLLAR_VOLUME_TOP_N_PCT = None
    DOLLAR_VOLUME_WINDOW = 30
    
    def prices_to_signals(self, prices):
          
        dollar_volumes = prices.loc["Volume"] * prices.loc["Close"]
        avg_dollar_volumes = dollar_volumes.rolling(window=self.DOLLAR_VOLUME_WINDOW).mean()
        if self.DOLLAR_VOLUME_TOP_N_PCT:
            dollar_volume_ranks = avg_dollar_volumes.rank(axis=1, ascending=False, pct=True)
            have_adequate_dollar_volumes = dollar_volume_ranks <= (self.DOLLAR_VOLUME_TOP_N_PCT/100)
        else:
            have_adequate_dollar_volumes = avg_dollar_volumes >= self.MIN_DOLLAR_VOLUME
            
        signals = have_adequate_dollar_volumes.astype(int)
        return signals

    def signals_to_target_weights(self, signals, prices):
        weights = self.allocate_equal_weights(signals)    
        return weights

    def target_weights_to_positions(self, weights, prices):
        positions = weights.shift()
        return positions

    def positions_to_gross_returns(self, positions, prices):
        closes = prices.loc["Close"]
        pct_changes = closes.pct_change()
        # Ignore gains or losses that are likely spurious data (e.g. 2->100 or 100->2) 
        pct_changes = pct_changes.where((pct_changes > -0.98) & (pct_changes < 50), 0)
        gross_returns = pct_changes * positions.shift()
        return gross_returns
    
class DollarVolumeWeightedIndex(EqualWeightedIndex):
    """
    Strategy that buys all stocks in the universe, optionally filtering by dollar volume, 
    and weights by dollar volume.
    
    Examples
    --------
    Create dollar-volume-weighted index of Japanese stocks, including only stocks in the top 
    50% of dollar volume: 
    
    >>> from codeload.benchmark import DollarVolumeWeightedIndex
    >>>
    >>> class JapanDollarVolumeWeightedIndex(DollarVolumeWeightedIndex):
    >>>
    >>>     CODE = "japan-benchmark"
    >>>     DB = "japan-stk-1d"
    >>>     DOLLAR_VOLUME_TOP_N_PCT = 50
    """
    CODE = None
    DB = None
    DB_FIELDS = ["Close", "Volume"]
    UNIVERSES = None
    EXCLUDE_UNIVERSES = None

    # Optionally filter universe by min dollar volume, or dollar volume rank 
    # (dollar volume rank takes precedence if both defined)
    MIN_DOLLAR_VOLUME = 0
    DOLLAR_VOLUME_TOP_N_PCT = None
    DOLLAR_VOLUME_WINDOW = 30

    def signals_to_target_weights(self, signals, prices):
        
        dollar_volumes = prices.loc["Volume"] * prices.loc["Close"]
        avg_dollar_volumes = dollar_volumes.rolling(window=self.DOLLAR_VOLUME_WINDOW).mean()
        avg_dollar_volumes = avg_dollar_volumes.where(signals > 0)
        
        total_daily_dollar_volumes = avg_dollar_volumes.sum(axis=1)
        weights = avg_dollar_volumes.div(total_daily_dollar_volumes, axis=0)
        return weights
