# Contains the OptionPriceModel class

import numpy as np
from abc import ABC, abstractmethod
from price_models.price_model import PriceModel

#################################
# OptionPriceModel Class
#################################

class OptionPriceModel(PriceModel, ABC): # This is going to become a subclass of Option. Maybe change the name to PricedOption or ValuedOption.
    """
    Abstract base class for option pricing models.

    Methods
    -------
    price(S0):
        Abstract method to compute the price of the option given the current stock price.

    delta(S0):
        Calculates the sensitivity of the option price to changes in the underlying stock price.

    gamma(S0):
        Calculates the sensitivity of the option's delta to changes in the underlying stock price.

    vega(S0):
        Calculates the sensitivity of the option price to changes in volatility.

    theta(S0):
        Calculates the sensitivity of the option price to the passage of time.

    rho(S0):
        Calculates the sensitivity of the option price to changes in the risk-free interest rate.
    """
    def payoff(self, ST):
        if self.option_type == "C":
            return self.contract_size * np.maximum(ST - self.strike, 0)
        elif self.option_type == "P":
            return self.contract_size * np.maximum(self.strike - ST, 0)


    @abstractmethod
    def value(self, S0, T, *args, **kwargs):
        """
        Compute the price of the option.

        Parameters
        ----------
        S0 : float
            Current price of the underlying asset.

        T : float
            Time till expiration in years.

        Returns
        -------
        float
            Price of the option.
        """
        pass
        