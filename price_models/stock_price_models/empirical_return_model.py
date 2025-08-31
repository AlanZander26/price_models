# Containt the EmpiricalReturnModel Class

import numpy as np
import scipy as sp
import warnings
from numpy.fft import fft, ifft, fftshift, ifftshift
from price_models.stock_price_models.stock_price_model import StockPriceModel


#################################
# EmpiricalReturnModel Class
#################################

class EmpiricalReturnModel(StockPriceModel):
    """
    Stock price model based on empirical log returns, extended to multiple periods
    using FFT convolution.

    Attributes
    ----------
    log_returns : ndarray
        Array of historical log returns.
    kde_log_returns : gaussian_kde
        Kernel density estimation of log returns.
    """

    # Class-level dictionary of periods per year
    PERIODS_PER_YEAR = {
        "year": 1,
        "month": 12,
        "week": 52,
        "day": 252,                # trading days in a year
        "hour": 252 * 6.5,         # ~6.5 trading hours per day
        "5min": 252 * 6.5 * 12,    # 12 intervals of 5 minutes per hour
    }

    def __init__(self, prices, period="day", fft_grid_size=2**15, T_default=np.linspace(1/252, 1, 252)):
        """
        Initializes the model with historical close prices.

        Parameters
        ----------
        prices : array-like
            Historical daily close prices.
        """
        prices = np.asarray(prices)
        self.log_returns = np.diff(np.log(prices))
        self.kde_log_returns = sp.stats.gaussian_kde(self.log_returns)
        if period not in self.PERIODS_PER_YEAR:
            raise ValueError(f"Invalid period: {period}. Must be one of {list(self.PERIODS_PER_YEAR)}.")
        else:
            self.period = period
        self._pdf_interp = None
        self._T_cached = None
        print("Interpolating probability distribution function...")
        self.set_cache_pdf(T_default, fft_grid_size=fft_grid_size)
        print("Interpolation finished.")

    def _logreturn_grid(self, N_periods, fft_grid_size):
        """
        Construct the log-return grid and spacing for FFT convolution.
        """
        low_q, high_q = np.quantile(self.log_returns, [0.001, 0.999])
        half_range = max(abs(low_q), abs(high_q)) * np.sqrt(N_periods)
        x = np.linspace(-half_range, half_range, fft_grid_size)
        dx = x[1] - x[0]
        return x, dx

    def set_cache_pdf(self, T_arr, fft_grid_size = 2**15, Nmax = 252):
        x, dx = self._logreturn_grid(Nmax, fft_grid_size)    
        pdf_1 = self.kde_log_returns(x)          
        cf = fft(fftshift(pdf_1)) * dx  # Till here all independent of T
        pdf_grid = []
        for T in T_arr:
            N_periods = int(round(T * self.PERIODS_PER_YEAR[self.period]))
            cf_T = cf ** N_periods
            pdf_T = fftshift(np.real(ifft(cf_T))) / dx
            pdf_grid.append(pdf_T) 
            mean_from_returns = np.exp(self.log_returns).mean() ** N_periods
            mean_from_pdf = np.trapz(pdf_T * np.exp(x), x)
            rel_error = abs(mean_from_pdf - mean_from_returns) / mean_from_returns
            if rel_error > 0.01:  # > 1%
                warnings.warn(
                    f"For T={T:.4f}, the PDF shows a relative error of {rel_error:.2%} "
                    "compared to the expected distribution. "
                    "This is likely due to aliasing or an insufficient FFT grid size. "
                    "Consider increasing `fft_grid_size` or restricting T to a smaller range.",
                    RuntimeWarning
                )   
        np.array(pdf_grid)
        self._pdf_interp = sp.interpolate.RegularGridInterpolator(
            (T_arr, x),
            pdf_grid,
            bounds_error=False,
            fill_value=0.0
        )
        self._T_cached = [T_arr[0], T_arr[-1]]       

    def stock_pdf(self, S0, ST, T, fft_grid_size=2**15):
        """
        Computes the PDF of stock price after horizon T using FFT convolution.

        Parameters
        ----------
        S0 : float
            Initial stock price.
        ST : array-like
            Stock prices to evaluate.
        T : float, optional
            Horizon length in years (default = 1.0).
        period : str, optional
            Base period for returns ("day", "week", "month", "year", "hour", "5min").
        fft_grid_size : int, optional
            Number of grid points for FFT (default = 2**12, must be power of 2).

        Returns
        -------
        ndarray
            PDF values for the stock prices at horizon T.
        """
        ST = np.asarray(ST)
        if self._T_cached[0] <= T <= self._T_cached[-1]:
            log_ST = np.log(ST / S0)
            pts = np.column_stack([np.full_like(log_ST, T), log_ST])
            f_rT = self._pdf_interp(pts)
            return f_rT / ST
        warnings.warn(
            f"Requested T={T:.4f} lies outside the cached range "
            f"[{self._T_cached[0]:.4f}, {self._T_cached[-1]:.4f}]. "
            "Falling back to direct FFT computation (slower). "
            "Consider extending the cache with `set_cache_pdf`.",
            RuntimeWarning
        )
        N_periods = int(round(T * self.PERIODS_PER_YEAR[self.period]))
        low_q, high_q = np.quantile(self.log_returns, [0.001, 0.999])
        half_range = max(abs(low_q), abs(high_q)) * np.sqrt(N_periods)
        x = np.linspace(-half_range, half_range, fft_grid_size)
        dx = x[1] - x[0]
        pdf_1 = self.kde_log_returns(x)  
        cf = fft(fftshift(pdf_1)) * dx 
        cf_T = cf**N_periods
        pdf_T = fftshift(np.real(ifft(cf_T))) / dx
        log_ST = np.log(ST / S0)
        f_rT = np.interp(log_ST, x, pdf_T, left=0.0, right=0.0)
        return f_rT / ST # Change of variables: f_ST(ST) = f_rT(log_ST) / ST


    def simulate_paths(self, S0, T, *args, N_paths=1, method="kde", **kwargs):
        """
        Simulates stock price paths based on historical returns.
    
        Parameters
        ----------
        S0 : float
            Initial stock price.
        N_periods : int
            Number of time steps in each path.
        N_paths : int, optional
            Number of simulated paths. Default is 1.
        method : {"bootstrap", "kde"}, optional
            Simulation method:
            - "bootstrap": resample directly from historical returns (empirical).
            - "kde": sample from a kernel density estimate of returns (smoothed).
    
        Returns
        -------
        ndarray
            Simulated stock price paths of shape (N_paths, N_periods+1).
        """
        N_periods = int(round(T * self.PERIODS_PER_YEAR[self.period]))
        if method == "bootstrap":
            # Directly resample from historical returns
            sampled_returns = np.random.choice(self.log_returns, size=(N_paths, N_periods))
        elif method == "kde":
            # Sample from KDE distribution (smooth version of historical returns)
            sampled_returns = self.kde_log_returns.resample(N_periods * N_paths).reshape(N_paths, N_periods)
        else:
            raise ValueError(f"Invalid method: {method}. Supported methods are either 'bootstrap' or 'kde'.")
        # Build price paths
        paths = np.zeros((N_paths, N_periods + 1))
        paths[:, 0] = S0
        paths[:, 1:] = S0 * np.exp(np.cumsum(sampled_returns, axis=1))
        return paths

    
    def pdf_range(self, S0, T):
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
        ST1, ST2 = 1e-3 * S0, 1e2 * S0
        return ST1, ST2
