# Contains the Geometric Brownian Motion Model class

import numpy as np
import scipy as sp
from price_models.stock_price_models.stock_price_model import StockPriceModel

#################################
# GBMModel Class
#################################

class GBMModel(StockPriceModel):
    """
    Stock price model assuming log-normal distribution.

    Attributes
    ----------
    T : float
        Time to maturity in years.
    sigma : float
        Volatility scaled by the square root of T.
    r : float
        Risk-free rate.
    well_behaved : bool
        Indicates whether numerical quadrature is used for integration or MC integration.

    Methods
    -------
    stock_pdf(S0, ST):
        Calculates the probability density function of stock price.
    
    probability_inside_range(S0, ST1, ST2):
        Calculates the probability that the stock price lies within a range.
    
    pdf_range(S0):
        Calculates the range in prices covering most of the support of the probability distribution.
    """
    def __init__(self):
        """
        Initializes the LogNormalModel.
        """
    
    def stock_pdf(self, S0, ST, T, vol, r):
        """
        Calculates the probability density function of the stock price.

        Parameters
        ----------
        S0 : float
            Initial stock price.
        ST : array-like
            Stock prices to evaluate.

        Returns
        -------
        ndarray
            PDF values for the stock prices.
        """
        sigma = vol * T**0.5
        ST = np.asarray(ST)
        result = np.zeros_like(ST, dtype=np.float64)
        valid_mask = (ST > 0) & (ST < np.inf)

        numer = np.log(S0 / ST[valid_mask]) + (r * T - 0.5 * sigma**2)
        d2 = numer / sigma
        result[valid_mask] = sp.stats.norm.pdf(d2) / (ST[valid_mask] * sigma)

        return result

    def stock_pdf_MC(self, S0, ST, N_steps, N_paths, T, vol, r):
        paths = self.simulate_paths(S0, N_steps=N_steps, N_paths=N_paths, T=T, vol=vol, r=r)
        ST_samples = paths[:, -1]
    
        # --- KDE smoothing on terminal prices ---
        kde = sp.stats.gaussian_kde(ST_samples)
        return kde(ST)

    def simulate_paths(self, S0, T, vol, r, N_steps=200, N_paths=1):
        deltaT = T / N_steps
        t_arr = np.linspace(deltaT, T, N_steps)  # time grid
        Z = sp.stats.norm.rvs(size=(N_paths, N_steps))
        W = np.cumsum(np.sqrt(deltaT) * Z, axis=1) # Brownian motion increments
        drift = (r - 0.5 * vol**2) * t_arr # Exact solution for GBM
        diffusion = vol * W
        log_paths = np.log(S0) + drift + diffusion
        paths = np.zeros((N_paths, N_steps+1))
        paths[:, 0] = S0
        paths[:, 1:] = np.exp(log_paths)
        return paths

    def prob_between(self, S0, ST1, ST2, T, vol, r, **kwargs): 
        """
        Calculates the probability that the stock price lies within a range.

        Parameters
        ----------
        S0 : float
            Initial stock price.
        ST1 : float
            Lower bound of the stock price range.
        ST2 : float
            Upper bound of the stock price range.

        Returns
        -------
        float
            Probability that the stock price is within the range.
        """
        sigma = vol * T**0.5
        if ST1 > ST2:
            raise ValueError("ST1 cannot be greater than ST2.")
        if ST2 <= 0:
            return 0.0
        elif ST2 == float('inf'):
            prob_max = 0.0
        else:
            numer_max = np.log(S0 / ST2) + (r * T - 0.5 * sigma**2)
            d2_max = numer_max / sigma
            prob_max = sp.stats.norm.cdf(d2_max)
        
        if ST1 <= 0:
            prob_min = 1.0
        elif ST1 == float('inf'):
            return 0.0
        else:
            numer_min = np.log(S0 / ST1) + (r * T - 0.5 * sigma**2)
            d2_min = numer_min / sigma
            prob_min = sp.stats.norm.cdf(d2_min)
        
        return prob_min - prob_max
    
    def pdf_range(self, S0, T, vol, r, eps=1e-3, nmax=10, *args, **kwargs):
        """
        Calculates the range in prices covering most of the support of the probability distribution.

        Parameters
        ----------
        S0 : float
            Initial stock price.

        Returns
        -------
        tuple
            Range of stock prices.
        """
        target_prob = 0.99999
        ST1, ST2 = self.price_interval(S0, T, target_prob, vol, r, eps=eps, nmax=nmax)
        return ST1, ST2

