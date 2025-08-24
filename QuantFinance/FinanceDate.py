from datetime import datetime, timedelta
from datetime import date as date_dt    
import numpy as np
from enum import Enum

class DayCountConvention:
    """Day count conventions."""
    ACT_365 = "ACT/365"
    ACT_360 = "ACT/360"
    CONV_30_360 = "30/360"
    ACT_ACT = "ACT/ACT"

class CalendarAdjustmentConvention:
    UNADJUSTED = "Unadjusted"
    ADJUSTED = "Adjusted"
    
class FinanceDate:
    """
    Represents a financial date stored as Excel-style day number.
    Can convert to years according to different day count conventions.
    Optimized with vectorized operations and reference date support.
    """
    
    # Class-level constants for performance
    _ORIGIN = datetime(1899, 12, 30)
    
    def __init__(self, excel_day_number):
        """
        Parameters
        ----------
        excel_day_number : int
            Excel-style day number (e.g., 40320)
        """
        self.excel_day_number = excel_day_number
        
        try:
            self.date = self._ORIGIN + timedelta(days=excel_day_number)
        except TypeError:
            print(excel_day_number)

    def to_datetime(self):
        """Return a standard datetime object."""
        return self.date

    @classmethod
    def from_datetime(cls, dt):
        """Create FinanceDate from datetime object."""
        delta = dt - cls._ORIGIN
        return cls(delta.days)
    
    @classmethod
    def from_date_string(cls, date_string, format="%Y-%m-%d"):
        """Create FinanceDate from date string."""
        dt = datetime.strptime(date_string, format)
        return cls.from_datetime(dt)
    
    def year_fraction_array(self, other_dates, convention=DayCountConvention.ACT_365):
        """
        Vectorized year fraction between two arrays of FinanceDate objects.
        
        Parameters
        ----------
        start_dates : array-like of FinanceDate
        end_dates : array-like of FinanceDate
        convention : str
            Day count convention
        round_up_full_years : bool
            Whether to round up full years
        
        Returns
        -------
        np.ndarray
            Array of year fractions
        """
        return [self.year_fraction(d, convention=convention) for d in other_dates]

    @classmethod
    def year_fraction_class(cls, initial_date, end_date, convention=DayCountConvention.ACT_360):
        if not isinstance(initial_date, FinanceDate):
            initial_date = FinanceDate(initial_date)
        
        if not isinstance(end_date, FinanceDate):
            end_date = FinanceDate(end_date)

        return initial_date.year_fraction(end_date, convention=convention)
    
    @classmethod
    def year_fraction_diff(cls, dates, convention=DayCountConvention.ACT_365,  with_zero=True):
        """
        Vectorized year fraction between consecutive dates in the same array.
        
        Parameters
        ----------
        dates : array-like of FinanceDate
            Array of dates
        convention : str
            Day count convention

        with_zero : bool
            If True, includes the first date in the calculation. So, 0 is returned for the first year fraction.
        Returns
        -------
        np.ndarray
            Array of year fractions of dates respect to te first date.

        """
        origin_date = dates[0]
        if not isinstance(dates, (list, tuple, np.ndarray)):
            if with_zero:
                dates = list(dates)
            else:
                dates = dates[1:]

        if len(dates) < 2:
            raise ValueError("At least two dates are required to compute year fractions.")

        origin_date = FinanceDate(origin_date)
        dates = np.array([FinanceDate(d) for d in dates])

        return origin_date.year_fraction_array(dates, convention=convention)

    def year_fraction(self, other, convention=DayCountConvention.ACT_ACT):
        """
        Compute the year fraction between this date and another FinanceDate.

        Parameters
        ----------
        other : FinanceDate
            The other financial date.
        convention : str
            Day count convention: DayCountConvention.ACT_365, "ACT/360", DayCountConvention.CONV_30_360, DayCountConvention.ACT_ACT
        round_up_full_years : bool
            If True, adds 1 for each full year difference (actual/actual).

        Returns
        -------
        float
            Fraction of year between the two dates.
        """
        if not isinstance(other, FinanceDate):
            other = FinanceDate(other)

        if self.excel_day_number == other.excel_day_number:
            return 0.0
        if self.excel_day_number > other.excel_day_number:
            return -other.year_fraction(self, convention)

        years = 0
        next_date = self.add_years(1)
        while next_date < other:
            years += 1
            next_date = self.add_years(years+1)
        
        ref_date = self.add_years(years)
        
        delta_days = other.excel_day_number - ref_date.excel_day_number

        if convention == DayCountConvention.ACT_365:
            year_fraction = delta_days / 365.0
        elif convention == DayCountConvention.ACT_360:
            year_fraction = delta_days / 360.0
        elif convention == DayCountConvention.CONV_30_360:
            # 30/360 assumes 30 days per month and 360 days per year
            start_month = ref_date.date.month
            start_day = ref_date.date.day
            end_month = other.date.month
            end_day = other.date.day
            start_year =ref_date.date.year
            end_year = other.date.year
            
            if start_day == 31 or (start_month == 2 and start_day > 28):
                start_day = 30
            if end_day == 31 or (end_month == 2 and end_day > 28):
                end_day = 30

            if start_month == end_month and start_year==end_year:
                year_fraction = (end_day - start_day) / 30.0
            else:
                if start_month == end_month:
                    year_fraction = 11/12+ (30 + end_day - start_day) / 30.0

                if start_day < end_day:
                    year_fraction = (end_day - start_day) / 360.0 
                    if start_year == end_year:
                        year_fraction += (end_month-start_month) /12
                    else:
                        year_fraction += (12-start_month+end_month)/12

                elif start_day > end_day:
                    year_fraction = (30 - start_day + end_day) / 360.0 

                    if start_year == end_year:
                        year_fraction += (end_month-start_month-1) /12
                    else:
                        year_fraction += (12-start_month+end_month-1)/12

                else:
                    if start_year == end_year:
                        year_fraction = (end_month-start_month) /12
                    else:
                        year_fraction = (12-start_month+end_month)/12

        elif convention == DayCountConvention.ACT_ACT:
            # Actual/Actual uses the actual number of days in the year
            start_year = ref_date.date.year
            end_year = other.date.year
            
            if start_year == end_year:
                year_fraction = delta_days / 365.0
            else:
                days_in_start_year = (datetime(start_year + 1, 1, 1) - datetime(start_year, 1, 1)).days
                days_in_end_year = (datetime(end_year + 1, 1, 1) - datetime(end_year, 1, 1)).days
                
                year_fraction = (days_in_start_year - ref_date.date.timetuple().tm_yday) / days_in_start_year
                
                year_fraction += other.date.timetuple().tm_yday / days_in_end_year
        else:
            raise ValueError(f"Unsupported day count convention: {convention}")
        
        return float(year_fraction) + years

    def get_day_number(self):
        return self.excel_day_number
    def __repr__(self):
        return f"FinanceDate({self.excel_day_number} -> {self.date.strftime('%Y-%m-%d')})"
    
    def __str__(self):
        return self.date.strftime('%Y-%m-%d')
    
    def __eq__(self, other):
        if isinstance(other, FinanceDate):
            return self.excel_day_number == other.excel_day_number
        return False
    
    def __lt__(self, other):
        if isinstance(other, FinanceDate):
            return self.excel_day_number < other.excel_day_number
        return False
    
    def __le__(self, other):
        return self == other or self < other
    
    def __gt__(self, other):
        return not self <= other
    
    def __ge__(self, other):
        return not self < other
    
    def __hash__(self):
        return hash(self.excel_day_number)
    
    def _restart_date(self):
        self.date = self._ORIGIN + timedelta(days=self.excel_day_number)
    
    def add_days(self, days):
        """Add days to the date and return a new FinanceDate."""
        return FinanceDate(self.excel_day_number + days)
    
    def add_months(self, months):
        """Add months to the date and return a new FinanceDate."""
        d = self.date
        year = d.year
        month = d.month + months
        day = d.day
        
        # Handle year and month overflow/underflow
        while month > 12:
            year += 1
            month -= 12
        while month < 1:
            year -= 1
            month += 12
        
        # Handle end-of-month dates
        try:
            new_date = FinanceDate.from_datetime(d.replace(year=year, month=month, day=day))
        except ValueError:
            # Day doesn't exist in target month (e.g., Jan 31 -> Feb 31)
            # Use last day of target month
            if month == 2:
                # February - handle leap years
                if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                    day = min(day, 29)
                else:
                    day = min(day, 28)
            elif month in [4, 6, 9, 11]:
                # Months with 30 days
                day = min(day, 30)
            else:
                # Months with 31 days
                day = min(day, 31)
            
            new_date = FinanceDate.from_datetime(d.replace(year=year, month=month, day=day))

        self._restart_date()

        return new_date


    def add_years(self, years):
        """
        Add years to the date and return a new FinanceDate.
        
        Parameters
        ----------
        years : int
            Number of years to add
        
        Returns
        -------
        FinanceDate
            New FinanceDate with added years
        """

        # d = self.date
        # try:
        #     d = d.replace(year = d.year + years)
        # except ValueError:
        #     d = d + (date_dt(d.year + years, 1, 1) - date_dt(d.year, 1, 1))
                    
        # aux = FinanceDate.from_datetime(d)

        # self._restart_date()
        return FinanceDate(self.excel_day_number + 365*years)
    
    def weekday(self):
        """Return the day of the week as an integer (Monday=0, Sunday=6)."""
        return self.date.weekday()
    
    def is_weekend(self):
        """Return True if the date falls on a weekend."""
        return self.weekday() >= 5  # Saturday=5, Sunday=6

# ===============================
# Business Day Calendar Classes
# ===============================

class Frequency(Enum):
    """Frequency enumeration for date generation."""
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    SEMIANNUAL = "SEMIANNUAL"
    ANNUAL = "ANNUAL"
    MATURITY_ONLY = "MATURITY_ONLY"

class BusinessDayConvention(Enum):
    """Business day adjustment conventions."""
    FOLLOWING = "FOLLOWING"
    MODIFIED_FOLLOWING = "MODIFIED_FOLLOWING"
    PRECEDING = "PRECEDING"
    MODIFIED_PRECEDING = "MODIFIED_PRECEDING"
    NO_ADJUSTMENT = "NO_ADJUSTMENT"

class HolidayCalendar:
    """
    Holiday calendar for business day calculations.
    Stores holidays as Excel day numbers for fast lookup.
    """
    
    def __init__(self, holidays=None, name="Custom"):
        """
        Parameters
        ----------
        holidays : list of int, FinanceDate, or datetime
            List of holiday dates
        name : str
            Name of the calendar
        """
        self.name = name
        self.holidays = set()
        
        if holidays:
            for holiday in holidays:
                if isinstance(holiday, int):
                    self.holidays.add(holiday)
                elif isinstance(holiday, FinanceDate):
                    self.holidays.add(holiday.excel_day_number)
                elif isinstance(holiday, (datetime, date_dt)):
                    fd = FinanceDate.from_datetime(holiday)
                    self.holidays.add(fd.excel_day_number)
                else:
                    raise ValueError(f"Unsupported holiday type: {type(holiday)}")
    
    def is_holiday(self, date):
        """Check if a date is a holiday."""
        if isinstance(date, FinanceDate):
            return date.excel_day_number in self.holidays
        elif isinstance(date, int):
            return date in self.holidays
        else:
            raise ValueError(f"Unsupported date type: {type(date)}")
    
    def is_business_day(self, date):
        """Check if a date is a business day (not weekend or holiday)."""
        if isinstance(date, int):
            date = FinanceDate(date)
        return not date.is_weekend() and not self.is_holiday(date)
    
    def add_holiday(self, holiday):
        """Add a holiday to the calendar."""
        if isinstance(holiday, int):
            self.holidays.add(holiday)
        elif isinstance(holiday, FinanceDate):
            self.holidays.add(holiday.excel_day_number)
        elif isinstance(holiday, (datetime, date_dt)):
            fd = FinanceDate.from_datetime(holiday)
            self.holidays.add(fd.excel_day_number)
    
    def add_holidays(self, holidays):
        """Add multiple holidays to the calendar."""
        for holiday in holidays:
            self.add_holiday(holiday)
    
    @classmethod
    def create_target_eur_calendar(cls):
        """Create a basic US holiday calendar (for demonstration)."""
        # This is a simplified version - real implementation would include all US holidays
        holidays = [
            FinanceDate.from_date_string("2025-01-01"),
            FinanceDate.from_date_string("2025-04-18"),
            FinanceDate.from_date_string("2025-04-21"),
            FinanceDate.from_date_string("2025-05-01"),
            FinanceDate.from_date_string("2026-12-24"),
            FinanceDate.from_date_string("2026-12-25"),
            FinanceDate.from_date_string("2026-01-01"),
            FinanceDate.from_date_string("2025-04-03"),
            FinanceDate.from_date_string("2025-04-06"),
            FinanceDate.from_date_string("2026-05-01"),
            FinanceDate.from_date_string("2026-12-24"),
            FinanceDate.from_date_string("2026-12-25")
            # TODO:Add more holidays as needed
        ]
        return cls(holidays, "US")
    
    def __repr__(self):
        return f"HolidayCalendar('{self.name}', {len(self.holidays)} holidays)"

class DateGenerator:
    """
    Advanced date generator with business day conventions and holiday calendar support.
    Optimized for financial calculations with proper business day handling.
    """
    
    def __init__(self, calendar=None, anchor='start'):
        """
        Parameters
        ----------
        calendar : HolidayCalendar, optional
            Holiday calendar for business day calculations
        """
        self.calendar = calendar or HolidayCalendar()
        self.anchor = anchor
    
    def generate_schedule(self, start_date, end_date, frequency, 
                      business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING,
                      include_start=False, include_end=True):
        """
        Generate a schedule of dates between start and end dates.
        
        Parameters
        ----------
        start_date : FinanceDate or int
            Start date of the schedule
        end_date : FinanceDate or int  
            End date of the schedule
        frequency : Frequency
            Frequency of the schedule
        business_day_convention : BusinessDayConvention
            Business day adjustment convention
        include_start : bool
            Whether to include the start date in the schedule
        include_end : bool
            Whether to include the end date in the schedule
        anchor : str
            "start" → stub at the end (build forward)
            "end"   → stub at the beginning (build backward)
            
        Returns
        -------
        list of FinanceDate
            Generated schedule of adjusted dates
        """
        # Convert to FinanceDate objects
        if isinstance(start_date, int):
            start_date = FinanceDate(start_date)
        if isinstance(end_date, int):
            end_date = FinanceDate(end_date)
        
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        # Generate unadjusted dates
        unadjusted_dates = self._generate_unadjusted_dates(
            start_date, end_date, frequency, include_start, include_end
        )
        
        # Apply business day adjustments
        adjusted_dates = []
        for date in unadjusted_dates:
            adjusted_date = self.adjust_business_day(date, business_day_convention)
            adjusted_dates.append(adjusted_date)
        
        # Remove duplicates while preserving order
        unique_dates = []
        seen = set()
        for date in adjusted_dates:
            if date.excel_day_number not in seen:
                unique_dates.append(date)
                seen.add(date.excel_day_number)
        
        return unique_dates
    
    def _generate_unadjusted_dates(self, start_date, end_date, frequency, 
                                  include_start, include_end):
        """Generate unadjusted dates based on frequency."""
        dates = []

        if not isinstance(start_date, FinanceDate):
            start_date = FinanceDate(start_date)

        if not isinstance(end_date, FinanceDate):
            end_date = FinanceDate(end_date)  

        if frequency == Frequency.MATURITY_ONLY:
            if include_start:
                dates.append(start_date)
            if include_end:
                dates.append(end_date)
            return dates

        # Paso de frecuencia en días/meses
        def step(date):
            if frequency == Frequency.WEEKLY:
                return date.add_days(7)
            elif frequency == Frequency.MONTHLY:
                return date.add_months(1)
            elif frequency == Frequency.QUARTERLY:
                return date.add_months(3)
            elif frequency == Frequency.SEMIANNUAL:
                return date.add_months(6)
            elif frequency == Frequency.ANNUAL:
                return date.add_years(1)
            else:
                raise ValueError(f"Unsupported frequency: {frequency}")

        if self.anchor == "start":
            # Construcción hacia adelante (stub al final)
            if include_start:
                dates.append(start_date)

            current_date = start_date
            while True:
                next_date = step(current_date)
                if next_date < end_date:
                    dates.append(next_date)
                    current_date = next_date
                else:
                    break

            if include_end and (not dates or dates[-1] != end_date):
                dates.append(end_date)

        elif self.anchor == "end":
            # Construcción hacia atrás (stub al principio)
            rev_dates = []
            if include_end:
                rev_dates.append(end_date)

            current_date = end_date
            while True:
                prev_date = step(current_date)
                # Invertimos el step hacia atrás
                if frequency == Frequency.WEEKLY:
                    prev_date = current_date.add_days(-7)
                elif frequency == Frequency.MONTHLY:
                    prev_date = current_date.add_months(-1)
                elif frequency == Frequency.QUARTERLY:
                    prev_date = current_date.add_months(-3)
                elif frequency == Frequency.SEMIANNUAL:
                    prev_date = current_date.add_months(-6)
                elif frequency == Frequency.ANNUAL:
                    prev_date = current_date.add_years(-1)

                if prev_date > start_date:
                    rev_dates.append(prev_date)
                    current_date = prev_date
                else:
                    break

            if include_start and (not rev_dates or rev_dates[-1] != start_date):
                rev_dates.append(start_date)

            # Invertimos para devolver en orden cronológico
            dates = list(reversed(rev_dates))

        else:
            raise ValueError("anchor must be 'start' or 'end'")

        return dates
    
    def adjust_business_day(self, date, convention):
        """
        Adjust a date according to business day convention.
        
        Parameters
        ----------
        date : FinanceDate
            Date to adjust
        convention : BusinessDayConvention
            Business day adjustment convention
            
        Returns
        -------
        FinanceDate
            Adjusted date
        """
        if convention == BusinessDayConvention.NO_ADJUSTMENT:
            return date
        
        if self.calendar.is_business_day(date):
            return date
        
        if convention == BusinessDayConvention.FOLLOWING:
            return self._find_next_business_day(date)
        
        elif convention == BusinessDayConvention.PRECEDING:
            return self._find_previous_business_day(date)
        
        elif convention == BusinessDayConvention.MODIFIED_FOLLOWING:
            adjusted = self._find_next_business_day(date)
            # If adjustment changes month, use preceding instead
            if adjusted.date.month != date.date.month:
                adjusted = self._find_previous_business_day(date)
            return adjusted
        
        elif convention == BusinessDayConvention.MODIFIED_PRECEDING:
            adjusted = self._find_previous_business_day(date)
            # If adjustment changes month, use following instead
            if adjusted.date.month != date.date.month:
                adjusted = self._find_next_business_day(date)
            return adjusted
        
        else:
            raise ValueError(f"Unsupported business day convention: {convention}")
    
    def _find_next_business_day(self, date, max_days=10):
        """Find the next business day after the given date."""
        current = date
        for _ in range(max_days):
            current = current.add_days(1)
            if self.calendar.is_business_day(current):
                return current
        raise ValueError(f"Could not find next business day within {max_days} days of {date}")
    
    def _find_previous_business_day(self, date, max_days=10):
        """Find the previous business day before the given date."""
        current = date
        for _ in range(max_days):
            current = current.add_days(-1)
            if self.calendar.is_business_day(current):
                return current
        raise ValueError(f"Could not find previous business day within {max_days} days of {date}")
    
    def is_business_day(self, date):
        """Check if a date is a business day."""
        return self.calendar.is_business_day(date)
    
    def add_business_days(self, date, days):
        """
        Add business days to a date.
        
        Parameters
        ----------
        date : FinanceDate
            Starting date
        days : int
            Number of business days to add (can be negative)
            
        Returns
        -------
        FinanceDate
            Date after adding business days
        """
        current = date
        remaining = abs(days)
        direction = 1 if days >= 0 else -1
        
        while remaining > 0:
            current = current.add_days(direction)
            if self.calendar.is_business_day(current):
                remaining -= 1
        
        return current
    
    def business_days_between(self, start_date, end_date):
        """
        Count business days between two dates (exclusive of start, inclusive of end).
        
        Parameters
        ----------
        start_date : FinanceDate
            Start date (excluded from count)
        end_date : FinanceDate  
            End date (included in count)
            
        Returns
        -------
        int
            Number of business days
        """
        if start_date >= end_date:
            return 0
        
        count = 0
        current = start_date.add_days(1)
        
        while current <= end_date:
            if self.calendar.is_business_day(current):
                count += 1
            current = current.add_days(1)
        
        return count


# ===============================
# Example Usage and Testing
# ===============================

if __name__ == "__main__":
    # Create a sample holiday calendar
    holidays = [
        FinanceDate.from_date_string("2024-01-01"),  # New Year's Day
        FinanceDate.from_date_string("2024-07-04"),  # Independence Day
        FinanceDate.from_date_string("2024-12-25"),  # Christmas
    ]
    calendar = HolidayCalendar(holidays, "Sample")
    
    # Create date generator
    generator = DateGenerator(calendar)
    
    # Test date generation
    start = FinanceDate.from_date_string("2024-01-15")
    end = FinanceDate.from_date_string("2024-06-15")
    
    print("=== DATE GENERATION EXAMPLES ===\n")
    
    # Monthly schedule with different conventions
    for convention in [BusinessDayConvention.FOLLOWING, 
                      BusinessDayConvention.MODIFIED_FOLLOWING,
                      BusinessDayConvention.PRECEDING,
                      BusinessDayConvention.MODIFIED_PRECEDING]:
        
        schedule = generator.generate_schedule(
            start, end, Frequency.MONTHLY, 
            business_day_convention=convention,
            include_start=True, include_end=True
        )
        
        print(f"{convention.value}:")
        for date in schedule:
            is_business = "✓" if generator.is_business_day(date) else "✗"
            print(f"  {date} ({date.date.strftime('%A')}) {is_business}")
        print()
    
    # Quarterly schedule
    print("QUARTERLY SCHEDULE (Modified Following):")
    quarterly_schedule = generator.generate_schedule(
        start, FinanceDate.from_date_string("2025-01-15"), 
        Frequency.QUARTERLY,
        business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING
    )
    
    for date in quarterly_schedule:
        is_business = "✓" if generator.is_business_day(date) else "✗"
        print(f"  {date} ({date.date.strftime('%A')}) {is_business}")
    
    print(f"\n=== BUSINESS DAY UTILITIES ===")
    test_date = FinanceDate.from_date_string("2024-01-13")  # Saturday
    print(f"Original date: {test_date} ({test_date.date.strftime('%A')})")
    print(f"Is business day: {generator.is_business_day(test_date)}")
    
    # Test different adjustments
    following = generator.adjust_business_day(test_date, BusinessDayConvention.FOLLOWING)
    preceding = generator.adjust_business_day(test_date, BusinessDayConvention.PRECEDING)
    print(f"Following: {following} ({following.date.strftime('%A')})")
    print(f"Preceding: {preceding} ({preceding.date.strftime('%A')})")
    
    # Business days calculations
    bd_count = generator.business_days_between(start, end)
    print(f"\nBusiness days between {start} and {end}: {bd_count}")
    
    future_date = generator.add_business_days(start, 10)
    print(f"10 business days after {start}: {future_date}")

if __name__ == "__main__":
    # Example usage
    fd1 = FinanceDate(40320)  # Represents 2020-01-01
    fd2 = FinanceDate(40870)  # Represents 2020-03-01
    
    print(fd1)  # Output: 2020-01-01
    print(fd2)  # Output: 2020-03-01
    
    print('ACT/ACT', fd1.year_fraction(fd2, convention='ACT/ACT'))  # Output: Year fraction between the two dates
    print('ACT/365', fd1.year_fraction(fd2, convention='ACT/365'))  # Output: Year fraction between the two dates
    print('ACT/360', fd1.year_fraction(fd2, convention='ACT/360'))  # Output: Year fraction between the two dates
    print('30/360', fd1.year_fraction(fd2, convention='30/360'))  # Output: Year fraction between the two dates

    dates = [
    45890, 45897, 45904, 45911, 45918, 45925, 45932,
    45939, 45946, 45953, 45960, 45967, 45974, 45981,
    45988, 45995, 46002, 46009, 46016, 46023, 46030]
    year_fractions = FinanceDate.year_fraction_diff(dates, convention='ACT/ACT')
    print("Year fractions between consecutive dates:", year_fractions)


    
