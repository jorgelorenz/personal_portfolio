import numpy as np
from scipy.interpolate import interp1d
from FinanceDate import FinanceDate, DayCountConvention
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

class InterestRateCurve:
    """
    Unified interest rate curve that can handle zero rates, discount factors, and forward rates.
    Supports various interpolation methods and automatic conversion between curve types.
    """

    @classmethod
    def from_dates(cls, reference_date, dates, values, curve_type="zero_rate", interpolation_zero_rates="linear", interpolation_discount_factors="exponential",
                  interpolation_forward_rates="linear", convention=DayCountConvention.ACT_360):
        """
        Initialize the interest rate curve with given dates and values.
        
        Parameters
        ----------
        dates : list or np.array
            Time points in years.
        values : list or np.array
            Corresponding values based on curve_type.
        curve_type : str
            Type of input curve: "zero_rate", "discount_factor", "forward_rate"
        interpolation : str or callable
            Interpolation method:
            - "linear": Linear interpolation
            - "cubic": Cubic spline interpolation
            - "quadratic": Quadratic interpolation
            - "nearest": Nearest neighbor
            - "slinear": 1st order spline (same as linear)
            - "zero": Zero-order hold (step function)
            - "previous": Previous value (backward step)
            - "next": Next value (forward step)
            - callable: Custom interpolation function
        fill_value : str, float, or tuple
            How to handle extrapolation:
            - "extrapolate": Extrapolate using the interpolation method
            - float: Use constant value for extrapolation
            - (float, float): Different values for below/above range
        bounds_error : bool
            If True, raises error when extrapolating. If False, uses fill_value.
        """
        return cls(reference_date, FinanceDate.year_fraction_diff(dates, convention=convention), values, curve_type, interpolation_zero_rates=interpolation_zero_rates, interpolation_discount_factors=interpolation_discount_factors,
                  interpolation_forward_rates=interpolation_forward_rates, convention=convention)
    
    def __init__(self, reference_date, times, values, curve_type="zero_rate", interpolation_zero_rates="linear", interpolation_discount_factors="exponential",
                  interpolation_forward_rates="linear", convention=DayCountConvention.ACT_360):
        """
        Parameters
        ----------
        times : list or np.array
            Time points in years.
        values : list or np.array
            Corresponding values based on curve_type.
        curve_type : str
            Type of input curve: "zero_rate", "discount_factor", "forward_rate"
        interpolation_zero_rates : str or callable
            Interpolation method for zero rates:
            - "linear", "cubic", "quadratic", "nearest", "slinear", "zero", "previous", "next"
            - callable: Custom interpolation function
        interpolation_discount_factors : str or callable
            Interpolation method for discount factors:
            - "linear", "cubic", "quadratic", "nearest", "slinear", "zero", "previous", "next"
            - callable: Custom interpolation function
        interpolation_forward_rates : str or callable
            Interpolation method for forward rates:
            - "linear", "cubic", "quadratic", "nearest", "slinear", "zero", "previous", "next"
            - callable: Custom interpolation function
        fill_value : str, float, or tuple
            How to handle extrapolation:
            - "extrapolate": Extrapolate using the interpolation method
            - float: Use constant value for extrapolation
            - (float, float): Different values for below/above range
        bounds_error : bool
            If True, raises error when extrapolating. If False, uses fill_value.
        """
        if len(times) != len(values):
            raise ValueError("Times and values must have the same length.")
        if len(times) < 2:
            raise ValueError("At least two time points are required to create a curve.")
        
        if not isinstance(reference_date, FinanceDate) and isinstance(reference_date, int):
            reference_date = FinanceDate(reference_date)
        else:
            #TODO: lanzar error, debe ser un entero que represente la fecha o una FinanceDate
            pass

        self.reference_date_ = reference_date
        self.times = np.array(times)
        self.values = np.array(values)
        self.curve_type = curve_type.lower()
        self.interpolation_zero_rates = interpolation_zero_rates
        self.interpolation_discount_factors = interpolation_discount_factors
        self.interpolation_forward_rates = interpolation_forward_rates
        self.convention_ = convention
        
        # Validate curve type
        valid_types = ["zero_rate", "discount_factor", "forward_rate"]
        if self.curve_type not in valid_types:
            raise ValueError(f"curve_type must be one of {valid_types}")
        
        self._compute_all_curves()
    
    def _compute_all_curves(self):
        """Compute all three curve representations from the input curve."""
        if self.curve_type == "zero_rate":
            self.zero_rates = ZeroRateCurve(self.times, self.values, interpolation_method=self.interpolation_zero_rates)
            self.discount_factors = self.zero_rates.discount_factors()
            self.forward_rates = self.zero_rates.forward_rates()
            
        elif self.curve_type == "discount_factor":
            self.discount_factors = DiscountCurve(self.times, self.values, interpolation_method=self.interpolation_discount_factors)
            self.zero_rates = self.discount_factors.zero_rates(interpolation_method=self.interpolation_zero_rates)
            self.forward_rates = self.discount_factors.forward_curve(interpolation_method=self.interpolation_forward_rates)
            
        elif self.curve_type == "forward_rate":
            self.forward_rates = ForwardRateCurve(self.times, self.values, interpolation_method=self.interpolation_forward_rates)
            self.discount_factors = self.forward_rates.discount_factors()
            self.zero_rates = self.forward_rates.zero_rates()
        else:
            raise ValueError(f"Unsupported curve type: {self.curve_type}")
    
    def _compute_forwards_from_zeros(self):
        """Compute forward rates from zero rates."""
        if len(self.times) < 2:
            return self.zero_rates.copy()
        
        delta_t = np.diff(np.insert(self.times, 0, 0))
        cumulative = self.zero_rates * self.times
        forwards = np.diff(np.insert(cumulative, 0, 0))
        
        # Avoid division by zero
        forward_mask = delta_t > 0
        result = np.zeros_like(delta_t)
        result[forward_mask] = forwards[forward_mask] / delta_t[forward_mask]
        return result
       
    def _compute_discounts_from_forwards(self):
        """Compute discount factors from forward rates."""
        delta_t = np.diff(np.insert(self.times, 0, 0))
        return np.exp(-np.cumsum(self.forward_rates * delta_t))

    def zero_rate_at(self, t):
        """Return interpolated zero rate at time/date t."""

        if isinstance(t, FinanceDate):
            t = self.reference_date_.year_fraction(t, convention=self.convention_)

        return self.zero_rates.value_at(t)
    
    def discount_factor_at(self, t):
        """Return interpolated discount factor at time t."""
        if isinstance(t, FinanceDate):
            t = self.reference_date_.year_fraction(t, convention=self.convention_)

        return self.discount_factors.value_at(t)
    
    def forward_rate_between(self, t1, t2):
        """Return interpolated forward rate at time t."""
        if isinstance(t1, FinanceDate):
            t1 = self.reference_date_.year_fraction(t1, convention=self.convention_)

        if isinstance(t2, FinanceDate):
            t2 = self.reference_date_.year_fraction(t2, convention=self.convention_)

        # print(self.discount_factor_at(t1))
        # print(self.discount_factor_at(t2))
        # print(t2-t1)
        return (self.discount_factor_at(t1)/self.discount_factor_at(t2)-1)/(t2-t1)

    def display_curve(self):
        """Print all curve representations and interpolation method."""
        #TODO: Que printee el tipo de curva que diga en un argumento
        print(f"Original Curve Type: {self.curve_type}")
        print(f"Interpolation Method: {self.interpolation_method}")
        print("\nAll Curve Points:")
        print("Time\t\tZero Rate\tDiscount\tForward Rate")
        print("-" * 60)
        for i, t in enumerate(self.times):
            print(f"{t:.4f}\t\t{self.zero_rates[i]:.6f}\t{self.discount_factors[i]:.6f}\t{self.forward_rates[i]:.6f}")
        ## Use matplotlib or similar library to plot the curves if needed

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(131)
        ax1.plot(self.times, self.zero_rates, label='Zero Rates', color='blue')
        ax1.set_title('Zero Rate Curve')
        ax1.set_xlabel('Time (Years)')
        ax1.set_ylabel('Zero Rate')

    def instant_forward_rate_at(self, t, h=1e-4):
        """
        Devuelve la forward instantánea f(0,t).
        - Si el método es 'linear', se deriva analíticamente la recta entre nodos.
        - Si es 'quadratic' o 'cubic' (splines), se evalúa la derivada del spline.
        - Para otros casos se usa derivada numérica como fallback.
        """
        method = self.forward_rates.interpolation_method

        if method in ['previous', 'next', 'zero']:
            raise ValueError("Instantaneous forward rate is not defined for stepwise interpolation methods.")

        elif method == 'linear':
            # tramo lineal entre (t0, P0) y (t1, P1)
            times = np.asarray(self.discount_factors.times)
            vals  = np.asarray(self.discount_factors.values)
            i = np.searchsorted(times, t) - 1
            i = max(0, min(i, len(times)-2))
            t0, t1 = times[i], times[i+1]
            P0, P1 = vals[i], vals[i+1]
            slope = (P1 - P0) / (t1 - t0)
            P_t   = P0 + slope*(t - t0)
            return -slope / P_t

        elif method in ['quadratic', 'cubic']:
            # Asumo que self.discount_factors.spline existe y es un objeto de scipy.interpolate
            P_spline = CubicSpline(self.discount_factors.times, self.discount_factors.values)
            dP_dt = P_spline.derivative()(t)
            f_instant = -dP_dt / P_spline(t)
            return f_instant

        else:
            # fallback: diferencia central sobre log(P)
            P_plus  = self.discount_factor_at(t+h)
            P_minus = self.discount_factor_at(t-h)
            return -(np.log(P_plus) - np.log(P_minus)) / (2*h)

class Curve:
    """Forward rate curve defined at finite dates (Excel day numbers)."""

    def __init__(self, times, values, interpolation_method="cubic"):
        """
        Parameters
        ----------
        dates_excel : array-like
            Excel day numbers
        forward_rates : array-like
            Forward rates
        day_count : str
            Day count convention
        """
        if len(times) != len(values):
            raise ValueError("Times and values must have the same length.")
        if len(times) < 2:
            raise ValueError("At least two time points are required to create a curve.")
        self.times = np.array(times)
        self.values = np.array(values, dtype=float)
        self.interpolation_method = interpolation_method
        self.create_interpolators()

    def create_interpolators(self):
        """Create interpolators for forward rates."""
        if self.interpolation_method == "exponential":
            self.log_interp = interp1d(
                self.times, np.log(self.values), kind="linear",
                bounds_error=False, fill_value="extrapolate"
            )
            def f(x):
                return np.exp(self.log_interp(x))
            self.interpolator = f
        else:
            self.interpolator = interp1d(
                self.times, self.values, kind=self.interpolation_method, bounds_error=False, fill_value="extrapolate"
            )

    def value_at(self, t):
        """
        Get the forward rate at time t.
        
        Parameters
        ----------
        t : float
            Time in years
        
        Returns
        -------
        float
            Forward rate at time t
        """
        return self.interpolator(t)
    
class DiscountCurve(Curve):
    def __init__(self, times, discount_factors, interpolation_method="cubic"):
        """
        Parameters
        ----------
        times : array-like
            Time points in years
        forward_rates : array-like
            Forward rates
        interpolation_method : str
            Interpolation method: "linear", "cubic", "quadratic", etc.
        """
        super().__init__(times, discount_factors, interpolation_method)

    def forward_curve(self, interpolation_method="cubic"):
        """Compute forward rates from discount factors."""
        delta_t = np.diff(self.times)
        result = (self.values[:-1]/self.values[1:]-1)/delta_t
        return ForwardRateCurve(self.times[:-1], result, interpolation_method=interpolation_method)
    
    def zero_rates(self, interpolation_method="cubic"):
        """Compute zero rates from discount factors."""
        result = (1/self.values[1:]-1)/self.times[1:]

        return ZeroRateCurve(self.times[:-1], result, interpolation_method=interpolation_method)
    
class ForwardRateCurve(Curve):
    def __init__(self, times, forward_rates, interpolation_method="cubic"):
        """
        Parameters
        ----------
        times : array-like
            Time points in years
        forward_rates : array-like
            Forward rates
        interpolation_method : str
            Interpolation method: "linear", "cubic", "quadratic", etc.
        """
        super().__init__(times, forward_rates, interpolation_method)

class ZeroRateCurve(Curve):
    def __init__(self, times, zero_rates, interpolation_method="cubic"):
        """
        Parameters
        ----------
        times : array-like
            Time points in years
        forward_rates : array-like
            Forward rates
        interpolation_method : str
            Interpolation method: "linear", "cubic", "quadratic", etc.
        """
        super().__init__(times, zero_rates, interpolation_method)