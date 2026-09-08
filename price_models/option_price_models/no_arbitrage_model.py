# Contains the NoArbitrageModel class

import numpy as np

from price_models.option_price_models.option_price_model import OptionPriceModel
from price_models.stock_price_models.stock_price_model import StockPriceModel

#################################
# NoArbitrageModel Class
#################################

class NoArbitrageModel(OptionPriceModel):

    def value(self, S0, T, *args, stock_price_model, r, N_paths=10_000, **kwargs):
        """
        Monte Carlo estimator for the arbitrage-free option value.

        Parameters
        ----------
        S0 : float
            Current price of the underlying asset.

        T : float
            Time to maturity in years.

        stock_price_model : StockPriceModel
            Stock price model used to simulate underlying price paths.

        r : float
            Risk-free interest rate.

        N_paths : int, optional
            Number of Monte Carlo simulation paths (default is 10,000).

        Returns
        -------
        float
            Estimated present value of the option, discounted at the risk-free rate.

        Raises
        ------
        TypeError
            If stock_price_model is not an instance of StockPriceModel.
        """

        if not isinstance(stock_price_model, StockPriceModel):
            raise TypeError("Invalid input. The stock price model should be an instance of StockPriceModel.")
        ST_arr = stock_price_model.simulate_paths(S0, T, *args, r=r, N_paths=N_paths, **kwargs)[:, -1]
        return np.exp(-r * T) * self.payoff(ST_arr).mean()