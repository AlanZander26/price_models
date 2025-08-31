# Contains the BlackScholesModels class

import numpy as np
import scipy as sp
from price_models.option_price_models import OptionPriceModel

#################################
# BlackScholes Class
#################################

class BlackScholesModel(OptionPriceModel): 
    """
    Black-Scholes option pricing model.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to expiration (in years).
    r : float
        Risk-free interest rate.
    vol : float
        Volatility of the underlying asset.
    option_type : str
        Type of option ('call' or 'put').
    contract_size : int, float
        Contract size. Default is 100, since equitiy options are based on 100 shares.

    Methods
    -------
    price(S0):
        Compute the price of the option.

    delta(S0):
        Compute the delta of the option.

    gamma(S0):
        Compute the gamma of the option.

    vega(S0):
        Compute the vega of the option.

    theta(S0):
        Compute the theta of the option.

    rho(S0):
        Compute the rho of the option.
    """
    
    def __init__(self, strike, option_type, contract_size=100):
        super().__init__(strike, option_type, contract_size=contract_size)
        
    def _d1_d2(self, S0, T, *, vol, r):
        """
        Helper method to calculate d1 and d2 for the Black-Scholes formula.

        Parameters
        ----------
        S0 : float
            Current price of the underlying asset.

        Returns
        -------
        tuple
            d1 and d2 values.
        """
        if T <= 0:
            return 0, 0
        sigma = vol * T**0.5
        d1 = (np.log(S0 / self.strike) + (r * T + 0.5 * sigma**2)) / (sigma)
        d2 = d1 - sigma
        return d1, d2
        
    def value(self, S0, T, *, vol, r):
        """
        Compute the price of the option.

        Parameters
        ----------
        S0 : float
            Current price of the underlying asset.

        Returns
        -------
        float
            Price of the option.
        """
        if T <= 0:
            return self.payoff(S0)
        S0 = np.asarray(S0)
        d1, d2 = self._d1_d2(S0, T, vol=vol, r=r)
        if self.option_type == "C":
            price = S0 * sp.stats.norm.cdf(d1) - self.strike * np.exp(-r * T) * sp.stats.norm.cdf(d2)
        elif self.option_type == "P":
            price = self.strike * np.exp(-r * T) * sp.stats.norm.cdf(-d2) - S0 * sp.stats.norm.cdf(-d1)
        else:
            raise ValueError(f"Invalid option type: {self.option_type}. Allowed types are 'call' or 'put'.")
        return self.contract_size*price
    
        
    def delta(self, S0, T, *, vol, r=0.0):
        d1, _ = self._d1_d2(S0, T, vol=vol, r=r)
        if self.option_type == "C":
            return sp.stats.norm.cdf(d1)
        elif self.option_type == "P":
            return sp.stats.norm.cdf(d1) - 1
    
    def gamma(self, S0, T, *, vol, r=0.0):
        sigma = vol * T**0.5
        d1, _ = self._d1_d2(S0, T, vol=vol, r=r)
        return sp.stats.norm.pdf(d1) / (S0 * sigma)
    
    def vega(self, S0, T, *, vol, r=0.0):
        d1, _ = self._d1_d2(S0, T, vol=vol, r=r)
        return S0 * sp.stats.norm.pdf(d1) * T**0.5
    
    def theta(self, S0, T, *, vol, r=0.0):
        sigma = vol * T**0.5
        d1, d2 = self._d1_d2(S0, T, vol=vol, r=r)
        term1 = -(S0 * sp.stats.norm.pdf(d1) * sigma) / (2 * T)
        if self.option_type == "C":
            term2 = r * self.strike * np.exp(-r * T) * sp.stats.norm.cdf(d2)
            return term1 - term2
        elif self.option_type == "P":
            term2 = r * self.strike * np.exp(-r * T) * sp.stats.norm.cdf(-d2)
            return term1 + term2
    
    def rho(self, S0, T, *, vol, r=0.0):
        _, d2 = self._d1_d2(S0, T, vol=vol, r=r)
        if self.option_type == "C":
            return T * self.strike * np.exp(-r * T) * sp.stats.norm.cdf(d2)
        elif self.option_type == "P":
            return -T * self.strike * np.exp(-r * T) * sp.stats.norm.cdf(-d2)
