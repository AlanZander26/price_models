# Contains the NoArbitrageModel class

import numpy as np
import scipy as sp
from price_models.option_price_models.option_price_model import OptionPriceModel
from price_models.stock_price_models.stock_price_model import StockPriceModel

#################################
# NoArbitrageModel Class
#################################

class NoArbitrageModel(OptionPriceModel): 

    
    def __init__(self, strike, option_type, stock_price_model, contract_size=100):
        if not isinstance(stock_price_model, StockPriceModel):
            raise TypeError(f"Invalid input. The stock price model should be an instance of StockPriceModel.")
        self.stock_price_model = stock_price_model
        super().__init__(strike, option_type, contract_size=contract_size)

    def value(self, S0, T, *args, r, N_paths=10_000, **kwargs):
        """
        Monte Carlo estimator for the arbitrage-free option value.
    
        Parameters
        ----------
        S0 : float
            Current price of the underlying asset.
        T : float
            Time to maturity in years.
        r : float
            Risk-free interest rate.
        N_paths : int, optional
            Number of Monte Carlo simulation paths (default is 10,000).  
        Returns
        -------
        float
            Estimated present value of the option, discounted at the risk-free rate.
        """
        ST_arr = self.stock_price_model.simulate_paths(S0, T, *args, r=r, N_paths=N_paths, **kwargs)[:, -1] 
        return np.exp(-r*T) * self.payoff(ST_arr).mean()
        
        
