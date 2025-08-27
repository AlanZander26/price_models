# Contains the StockPriceModel abstract class

import scipy as sp
import numpy as np
from abc import ABC, abstractmethod
import warnings
from price_models.price_model import PriceModel

#################################
# StockPriceModel Class
#################################

class StockPriceModel(PriceModel, ABC): 
    """
    Abstract base class for stock price modeling.

    Methods
    -------
    stock_pdf(S0, ST, T, *params_pdf):
        Abstract method to calculate the probability density function for the stock price.

    weighted_integral(S0, ST1, ST2, f, *params_pdf):
        Computes the integral of a weighted function using either numerical quadrature
        or Monte Carlo integration.

    prob_between(S0, ST1, ST2):
        Calculates the probability that the stock price lies within a range.

    price_interval(S0, T, coverage_prob, eps=1e-3, nmax=10, *params):
        Determines the stock price range for a target confidence level.

    pdf_range(S0):
        Returns the range in prices covering most of the support of the probability distribution.
    """
    @abstractmethod
    def stock_pdf(self, S0, ST, T, *args, **kwargs):
        pass
    
    @abstractmethod
    def simulate_paths(self, S0, N_paths, *args, **kwargs): # Make documentation!!!!!!!!
        pass

    def weighted_integral(self, S0, ST1, ST2, f, T, *args, **kwargs):
        """
        Computes the integral of a weighted function with respect to the stock 
        price distribution. Attempts numerical quadrature first; if the range is
        too large or quadrature fails, falls back to trapezoidal integration.
    
        Parameters
        ----------
        S0 : float
            Initial stock price.
        ST1 : float
            Lower bound of the stock price range.
        ST2 : float
            Upper bound of the stock price range.
        f : callable
            Weight function.
        T : float
            Time horizon in years.
        *args, **kwargs :
            Additional parameters passed to `stock_pdf`.
    
        Returns
        -------
        float
            Result of the integration.
        """
        # Heuristic: if integration range is extremely wide, warn and skip quad
        if ST2 - ST1 > 10 * max(1.0, S0):
            warnings.warn(
                f"Integration range [{ST1}, {ST2}] is very wide relative to S0={S0}. "
                "Falling back to trapezoidal integration.",
                RuntimeWarning
            )
        else:
            try:
                result, _ = sp.integrate.quad(
                    lambda ST: f(ST) * self.stock_pdf(S0, ST, T, *args, **kwargs),
                    ST1, ST2,
                    epsabs=1e-6, epsrel=1e-6
                )
                return result
            except Exception as e:
                warnings.warn(
                    f"Quadrature integration failed ({e}); falling back to trapezoidal integration.",
                    RuntimeWarning
                )
        # --- Trapezoidal fallback ---
        points_per_unit = 100
        n_samples = int(points_per_unit * (ST2 - ST1))
        n_samples = max(n_samples, 100)  # ensure a minimum resolution
        samples = np.linspace(ST1, ST2, n_samples)
        weights = f(samples) * self.stock_pdf(S0, samples, T, *args, **kwargs)
        result = np.trapz(weights, samples)
        return result


    def pdf_mean(self, S0, T, *args, **kwargs):
        ST1, ST2 = self.pdf_range(S0, T, *args, **kwargs)
        f = lambda x: x
        return self.weighted_integral(S0, ST1, ST2, f, T, *args, **kwargs)
    
    def pdf_variance(self, S0, T, *args, **kwargs):
        ST1, ST2 = self.pdf_range(S0, T, *args, **kwargs)
        mean_val = self.pdf_mean(S0, T, *args, **kwargs)
        f = lambda x: x**2
        second_moment = self.weighted_integral(S0, ST1, ST2, f, T, *args, **kwargs)
        return second_moment - mean_val**2

    def pdf_std(self, S0, T, *args, **kwargs):
        return self.pdf_variance(S0, T, *args, **kwargs)**0.5
    
    def prob_between(self, S0, ST1, ST2, T, *args, **kwargs): 
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
        f = lambda x: 1
        return np.clip(self.weighted_integral(S0, ST1, ST2, f, T, *args, **kwargs), 0.0, 1.0)

    def price_interval(self, S0, T, coverage_prob, *args, eps=1e-3, nmax=10, **kwargs):
        """
        Determines the stock price range for a target confidence level.

        Parameters
        ----------
        S0 : float
            Initial stock price.
        coverage_prob : float
            Desired probability level.
        eps : float, optional
            Convergence tolerance. Default is 1e-3.
        nmax : int, optional
            Maximum number of iterations. Default is 10.
        *params : tuple
            Additional parameters for the PDF.

        Returns
        -------
        tuple
            Lower and upper bounds of the confidence interval.
        """
        if coverage_prob > 1:
            raise ValueError(f"Invalid input: {coverage_prob}. Coverage probability must be less than 1.")
        def objective(x):
            ST1, ST2 = sorted(x)
            return abs(self.prob_between(S0, ST1, ST2, T, *args, **kwargs) - coverage_prob)

        # def objective(x):
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("ignore", category=RuntimeWarning)
        #         ST1, ST2 = sorted(x)
        #         return abs(self.prob_between(S0, ST1, ST2, T, *args, **kwargs) - coverage_prob)

        constraints = (
            {'type': 'ineq', 'fun': lambda x: x[0]},       # ST1 > 0
            {'type': 'ineq', 'fun': lambda x: x[1] - x[0]} # ST2 > ST1
        )
        diff = 1 + eps
        n = 0
        while diff > eps and n < nmax:
            percentages = np.linspace(eps, 1 - eps, nmax)
            perc = percentages[n]
            initial_guess = [S0 * (1 - perc), S0 * (1 + perc)]
            result = sp.optimize.minimize(objective, initial_guess, constraints=constraints)
            if not result.success:
                raise ValueError("Optimization failed to converge. Check your inputs or try a different initial guess.")
            diff = result.fun
            n += 1
        if n == nmax and diff > eps:
            raise ValueError(f"Maximum iterations (nmax = {nmax}) reached without achieving the required tolerance eps = {eps}. Consider increasing nmax or reducing eps.")
        ST1, ST2 = sorted(result.x)
        return ST1, ST2
    
    def pdf_range(self, S0, T, *args, **kwargs):
        """
        Returns the range in prices covering most of the support of the probability distribution.

        Returns
        -------
        tuple
            Range of stock prices.
        """
        return -np.inf, np.inf #1e-3*S0*np.sqrt(1+T), 1e3*S0*np.sqrt(1+T)

    def POT(self, S0, T, barrier, *args, N_paths=10_000, **kwargs):
        """
        Estimates the probability of touch (POT) for a given barrier level
        using Monte Carlo simulation. The probability of touch is the probability
        that the stock price path crosses the barrier at any time before maturity.
    
        Parameters
        ----------
        S0 : float
            Initial stock price.
        barrier : float
            Barrier level to check for touch. If equal to S0, returns 1.
            If greater than S0, computes the probability of touching an
            up-barrier. If less than S0, computes the probability of touching
            a down-barrier.
        T : float
            Time horizon (in years).
        n_steps : int, optional
            Number of time steps in each simulated path. Default is 252.
        N_paths : int, optional
            Number of simulated paths. Default is 10,000.
        *args, **kwargs :
            Additional parameters passed to `simulate_paths`.
    
        Returns
        -------
        float
            Estimated probability of touch, in the range [0, 1].
        """
        if barrier == S0:
            return 1.0
        paths = self.simulate_paths(S0, T, *args, N_paths=N_paths, **kwargs)
        if barrier < S0:
            touched = (paths < barrier).any(axis=1)
        else:
            touched = (paths > barrier).any(axis=1)
        return touched.mean()
