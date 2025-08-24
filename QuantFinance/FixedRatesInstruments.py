from Instruments import Instrument, Option, OneStrikeOption
from Payoffs import EuropeanCallPayOff, EuropeanPutPayOff
from SimulatorEngine import Black76Simulator, BachelierSimulator
import numpy as np
import scipy.stats as st
from FinanceDate import FinanceDate, Frequency, BusinessDayConvention, HolidayCalendar, DateGenerator, DayCountConvention, CalendarAdjustmentConvention
from InteresRates import InterestRateCurve

class InterestRateInstrument(Instrument):
    def __init__(self, valuation_date, irCurve):
        super().__init__(valuation_date)
        self.irCurve_ = irCurve

class NotionalType:
    FLAT = 'Flat'
    AMORTIZING = 'Amortizing'
    SCHEDULE = 'Schedule'


class FixedLeg(InterestRateInstrument):
    def __init__(self, valuation_date, discount_curve, fixed_coupon, initial_period_dates, payment_dates, final_period_dates=None, convention=DayCountConvention.ACT_360,
                 notional=1.0, type_notional=NotionalType.FLAT, notional_schedule=None):
        """
        
        """
        #TODO:check len(initial_period_dates) = len(payment_dates)
        #TODO: if notional_schedule is not None, check len(notional_schedule) = len(payment_dates)

        super().__init__(valuation_date, discount_curve)
        self.payment_dates_ = [p if isinstance(p, FinanceDate) else FinanceDate(p) for p in payment_dates]  # List of FinanceDate objects
        self.fixed_coupon_ = fixed_coupon      # Fixed coupon rate (annualized)
        self.initial_period_dates_ = initial_period_dates
        self.final_period_dates_ = payment_dates if final_period_dates == None else final_period_dates
        self.convention_ = convention
        self.cash_flows = {}
        self.notional_ = notional
        if type_notional not in [NotionalType.FLAT, NotionalType.AMORTIZING, NotionalType.SCHEDULE]:
            raise ValueError("type_notional must be NotionalType.FLAT or NotionalType.SCHEDULE")
        
        if notional_schedule is None:
            if type_notional == NotionalType.FLAT:
                self.notional_schedule_ = [notional] * len(self.payment_dates_)
            elif type_notional == NotionalType.AMORTIZING:
                self.notional_schedule_ = [notional * (1 - i / len(self.payment_dates_)) for i in range(len(self.payment_dates_))]
            else:
                raise ValueError("type_notional must be NotionalType.FLAT or NotionalType.AMORTIZING if notional_schedule is None")
        else:
            if len(notional_schedule) != len(self.payment_dates_):
                raise ValueError("Length of notional_schedule must match number of payment dates")
            self.notional_schedule_ = notional_schedule
            

    def valuate(self):
        """Calculate the present value of the fixed leg."""
        pv = 0.0
        for p, i, fp, n in zip(self.payment_dates_, self.initial_period_dates_, self.final_period_dates_, self.notional_schedule_):
            if p < self.valuation_date_:
                continue
            disc = self.irCurve_.discount_factor_at(p)
            delta = FinanceDate.year_fraction_class(i, fp, convention=self.convention_)
            pv += self.fixed_coupon_ * delta * disc
            self.cash_flows[p] = self.fixed_coupon_ * delta * disc * n
        return pv

class FloatingLeg(InterestRateInstrument):

    STANDAR_IRS = 'Standar IRS'
    IRS_CAP_FLOOR = 'IRS with caps and/or floors'

    def __init__(self, valuation_date, irCurve, initial_period_dates, payment_dates, fixing_dates, final_period_dates=None, known_rates={}, reference_weight = 1.0, spread=0.0, risk_free_curve = None, 
                 caps=None, floors=None, callable_calendar=None, putable_calendar=None, notional=1.0, type_notional=NotionalType.FLAT, notional_schedule=None,
                 valuation_cascade=None, caps_floors_volatilities = None, stripped_caps_floors_volatilities=None, convention=DayCountConvention.ACT_360):
        """
        
        """
        super().__init__(valuation_date, irCurve)
        self.known_rates_ = known_rates#TODO: check is dictionary
        self.initial_period_dates_ = [d if isinstance(d, FinanceDate) else FinanceDate(d) for d in initial_period_dates]
        self.payment_dates_ = [p if isinstance(p, FinanceDate) else FinanceDate(p) for p in payment_dates]
        self.fixing_dates_ = [d if isinstance(d, FinanceDate) else FinanceDate(d) for d in fixing_dates] 
        self.spread_ = spread  # Spread over the floating rate
        self.reference_weight_ = reference_weight
        self.risk_free_curve_ = risk_free_curve
        self.caps_ = caps
        self.floors_ = floors
        self.callable_calendar_ = callable_calendar
        self.putable_calendar_ = putable_calendar
        self.caps_floors_volatilities = caps_floors_volatilities
        self.stripped_caps_floors_volatilities = stripped_caps_floors_volatilities
        self.convention_ = convention
        self.final_period_dates_ = payment_dates if final_period_dates == None else final_period_dates

        STANDAR_VALUATION_CASCADE = {
            FloatingLeg.STANDAR_IRS : self.forwardRateDiscount,
            FloatingLeg.IRS_CAP_FLOOR : self.BlackModel
        }

        self.cash_flows = {}
        self.valuation_cascade_ = STANDAR_VALUATION_CASCADE if valuation_cascade == None else valuation_cascade

        self.notional_ = notional
        if notional_schedule is None:
            if type_notional == NotionalType.FLAT:
                self.notional_schedule_ = [notional] * len(self.payment_dates_)
            elif type_notional == NotionalType.AMORTIZING:
                self.notional_schedule_ = [notional * (1 - i / len(self.payment_dates_)) for i in range(len(self.payment_dates_))]
            else:
                raise ValueError("type_notional must be NotionalType.FLAT or NotionalType.AMORTIZING if notional_schedule is None")
        else:
            if len(notional_schedule) != len(self.payment_dates_):
                raise ValueError("Length of notional_schedule must match number of payment dates")
            self.notional_schedule_ = notional_schedule

    def valuate(self):
        if self.callable_calendar_ == None and self.putable_calendar_ == None and self.caps_ == None and self.floors_ == None:
            return self.valuation_cascade_[FloatingLeg.STANDAR_IRS]()
        elif self.callable_calendar_ == None and self.putable_calendar_ == None:
            #TODO: check caps_floors_volatilities or stripped_caps_floors_volatilities is not None
            return self.valuation_cascade_[self.IRS_CAP_FLOOR]()
        elif self.caps_ == None and self.floors_ == None:
            #TODO: hacer valoracion de esto y parámetros necesarios
            pass
        else:
            #TODO: hacer valoracion de esto y parámetros necesarios
            pass

    def forwardRateDiscount(self):
        if self.risk_free_curve_ == None:
            disc_curve = self.irCurve_
        else:
            disc_curve = self.risk_free_curve_

        pv = 0.0
        for p, i, f, fp, n in zip(self.payment_dates_, self.initial_period_dates_, self.fixing_dates_, self.final_period_dates_, self.notional_schedule_):
            if p < self.valuation_date_:
                continue
            if f < self.valuation_date_:
                try:
                    rate = self.known_rates_[f.get_day_number()]
                except: 
                    raise ValueError(f"For date {f.get_day_number()} there should be a known rate")
            else:
                rate = self.irCurve_.forward_rate_between(i, p)

            rate = rate*self.reference_weight_ + self.spread_ 
            delta = FinanceDate.year_fraction_class(i, fp, self.convention_)
            disc = disc_curve.discount_factor_at(p)
            pv += disc*delta*rate * n
            self.cash_flows[p] = disc*delta*rate*n
        return pv
    
    def BlackModel(self):
        #TODO: valorar como forwardRateDiscount + el valor de cada floor comprado - el valor de cada cap vendido -> a través de sus propias clases(ya la tengo en otro python hecha)
        pass


class InterestRateSwap(InterestRateInstrument):
    def __init__(self, leg_pay, leg_receive):
        if leg_pay.valuation_date_ != leg_receive.valuation_date_:
            raise ValueError("Both legs must have the same valuation date.")
        super().__init__(leg_pay.valuation_date_, leg_pay.irCurve_)
        self.leg_pay_ = leg_pay
        self.leg_receive_ = leg_receive

    def valuate(self):
        return self.leg_receive_.valuate() - self.leg_pay_.valuate()

class InterestRateSwapFixedVSFloating(InterestRateSwap):
    @classmethod
    def shortDefinition(cls, valuation_date, irCurve, fixed_coupon, initial_date, end_date,
                        frequency_float, frequency_fixed=Frequency.ANNUAL,  
                        convention_fixed=DayCountConvention.ACT_360, convention_float=DayCountConvention.ACT_360,
                        business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING, 
                        anchor='end', calendar=HolidayCalendar.create_target_eur_calendar(),
                        calendar_convention="Adjusted", fixing_lag=2,
                        known_rates={}, reference_weight=1.0, spread=0.0, risk_free_curve=None,
                        caps=None, floors=None, callable_calendar=None, putable_calendar=None,
                        valuation_cascade=None,
                        caps_floors_volatilities=None, 
                        stripped_caps_floors_volatilities=None,
                        notional=1.0, type_notional=NotionalType.FLAT, notional_schedule=None):
        
        generator = DateGenerator(calendar, anchor=anchor)

        # Flotante
        float_dates = generator.generate_schedule(initial_date, end_date, frequency_float,
                                                  business_day_convention, include_start=True, include_end=True)
        float_dates_unadjusted = generator._generate_unadjusted_dates(initial_date, end_date, frequency_float,
                                                                      include_start=True, include_end=True)
        payment_dates_float = float_dates[1:]
        initial_period_dates_float = float_dates[:-1]
        final_period_dates_float = None if calendar_convention == CalendarAdjustmentConvention.ADJUSTED else float_dates_unadjusted[1:]
        fixing_dates = [generator.adjust_business_day(d.add_days(fixing_lag), 
                                                      convention=BusinessDayConvention.PRECEDING)  
                        for d in payment_dates_float]
        # Fijo
        fixed_dates = generator.generate_schedule(initial_date, end_date, frequency_fixed,
                                                  business_day_convention, include_start=True, include_end=True)
        fixed_dates_unadjusted = generator._generate_unadjusted_dates(initial_date, end_date, frequency_fixed,
                                                                      include_start=True, include_end=True)
        payment_dates_fixed = fixed_dates[1:]
        initial_period_dates_fixed = fixed_dates[:-1]
        final_period_dates_fixed = None if calendar_convention ==CalendarAdjustmentConvention.ADJUSTED else fixed_dates_unadjusted[1:]

        # Defaults dinámicos
        n_float_payments = len(payment_dates_float)
        if not caps is None:
            caps = [caps] * n_float_payments
        if not floors is None:
            floors = [floors] * n_float_payments

        return cls(
            valuation_date=valuation_date,
            irCurve=irCurve,
            fixed_coupon=fixed_coupon,
            initial_period_dates_fixed=initial_period_dates_fixed,
            payment_dates_fixed=payment_dates_fixed,
            initial_period_dates_float=initial_period_dates_float,
            payment_dates_float=payment_dates_float,
            fixing_dates=fixing_dates,
            known_rates=known_rates,
            reference_weight=reference_weight,
            spread=spread,
            risk_free_curve=risk_free_curve,
            final_period_dates_fixed=final_period_dates_fixed,
            final_period_dates_float=final_period_dates_float,
            caps=caps,
            floors=floors,
            callable_calendar=callable_calendar,
            putable_calendar=putable_calendar,
            valuation_cascade=valuation_cascade,
            caps_floors_volatilities=caps_floors_volatilities,
            stripped_caps_floors_volatilities=stripped_caps_floors_volatilities,
            convention_fixed=convention_fixed,
            convention_float=convention_float,
            notional=notional,
            type_notional=type_notional,
            notional_schedule=notional_schedule
        )




    def __init__(self, valuation_date, irCurve, fixed_coupon, initial_period_dates_fixed, payment_dates_fixed, initial_period_dates_float, 
                payment_dates_float, fixing_dates, type='payer', known_rates = {}, reference_weight = 1.0, spread=0.0, risk_free_curve = None, 
                notional=1.0, type_notional=NotionalType.FLAT, notional_schedule=None,
                final_period_dates_fixed = None, final_period_dates_float = None,
                caps=None, floors=None, callable_calendar=None, putable_calendar=None, valuation_cascade=None,
                caps_floors_volatilities = None, stripped_caps_floors_volatilities=None, convention_fixed=DayCountConvention.ACT_360,
                convention_float=DayCountConvention.ACT_360):
        
        floating_leg = FloatingLeg(valuation_date, irCurve, initial_period_dates_float, payment_dates_float, fixing_dates, known_rates=known_rates, reference_weight=reference_weight,
                                    spread=spread, risk_free_curve=risk_free_curve, caps=caps, floors=floors, callable_calendar=callable_calendar, putable_calendar=putable_calendar,
                                    valuation_cascade=valuation_cascade, caps_floors_volatilities=caps_floors_volatilities, stripped_caps_floors_volatilities=stripped_caps_floors_volatilities,
                                    convention=convention_float, final_period_dates=final_period_dates_float,
                                    notional=notional, type_notional=type_notional, notional_schedule=notional_schedule)
        
        if risk_free_curve == None:
            disc_curve = irCurve
        else:
            disc_curve = risk_free_curve

        fixed_leg = FixedLeg(valuation_date, disc_curve, fixed_coupon, initial_period_dates_fixed, 
                                   payment_dates_fixed, convention=convention_fixed, final_period_dates=final_period_dates_fixed,
                                   notional=notional, type_notional=type_notional, notional_schedule=notional_schedule)
        
        if type.lower() == 'payer':
            super().__init__(fixed_leg,floating_leg)
        elif type.lower() == 'receiver':
            super().__init__(floating_leg,fixed_leg)
        else:
            raise ValueError("type must be 'payer' or 'receiver'")
    


######## Caps, Floors y Swaptions #######

class FixedRateEuropeanSingleOption(OneStrikeOption):
    def __init__(self, R0, K, r, sigma, T, c_p, deltat, notional):
        #TODO: meter shifted log normal
        super().__init__(R0, K, r, 0, sigma, T)
        self.R0_ = R0
        self.d1_ = (np.log(self.R0_ / self.K_) + (0.5 * self.sigma_flat_**2) * self.T_) / (self.sigma_flat_ * np.sqrt(self.T_))
        self.d2_ = self.d1_ - self.sigma_flat_ * np.sqrt(self.T_)
        if c_p == 'c':
            self.payoff_ = EuropeanCallPayOff(self.K_)
        elif c_p == 'p':
            self.payoff_ = EuropeanPutPayOff(self.K_)
        else: 
            pass #TODO: lanzar error

        self.deltat_ = deltat
        self.notional_ = notional
        self.c_p = c_p
        self.simulatorBS_ = Black76Simulator(self.R0_, self.sigma_flat_)
        self.simulatorBach_ = BachelierSimulator(self.R0_, 0, 0, sigma=self.sigma_flat_)
        self.models = {'BS': self.priceBS, 'Bachelier':self.priceBachelier}

    def priceBS(self, sigma=None, simulation=False, n_sims=10000):
        if sigma != None:
            d1 = (np.log(self.R0_ / self.K_) + (0.5 * sigma**2) * self.T_) / (sigma * np.sqrt(self.T_))
            d2 = d1 - sigma * np.sqrt(self.T_)
        else:
            sigma = self.sigma_flat_
            d1 = self.d1_
            d2 = self.d2_

        if self.c_p == 'p':
            w = -1
        else:
            w = 1

        if simulation:
            ###We had to adapt results from MC pricer because the discount is deltat time after the option execution(this case isn't in-arrears)
            p = self.priceBSMC(sigma, n_sims=n_sims)*self.discount(self.deltat_)
        else:
            p = w*self.maturityDiscount()*(self.R0_ * st.norm.cdf(w*d1) - self.K_ * st.norm.cdf(w*d2))
        
        return p*self.deltat_*self.notional_
        # res = super().priceBS(sigma, simulation, n_sims)*self.deltat_*self.notional_
        # return res*self.discount(self.deltat_) if simulation else res
    
    def priceBSMC(self, sigma=None, n_sims=10000):
        if sigma != None:
            simulator = Black76Simulator(self.R0_, sigma)
        else:
            simulator = self.simulatorBS_

        return self.priceMC(simulator, self.payoff_, [self.T_], n_sims=n_sims)
    
    def priceBachelier(self, sigma=None, simulation=False, n_sims=10000):
        if sigma != None:
                simulatorBach = BachelierSimulator(self.R0_, 0, 0, sigma=sigma)
        else:
            sigma = self.sigma_flat_
            simulatorBach = self.simulatorBach_

        d = (self.R0_-self.K_)/(sigma*np.sqrt(self.T_))


        if simulation:
            ###We had to adapt results from MC pricer because the discount is deltat time after the option execution(this case isn't in-arrears)
            return self.priceMC(simulatorBach, self.payoff_, [self.T_], n_sims=n_sims)*self.deltat_*self.notional_*self.discount(self.deltat_)
        else:
            w = 1 if self.c_p == 'c' else -1
            return self.notional_*self.deltat_*self.maturityDiscount()*(w*(self.R0_-self.K_)*st.norm.cdf(w*d)+sigma*np.sqrt(self.T_)*st.norm.pdf(d))
    
    def maturityDiscount(self):
        return self.discount(self.T_+self.deltat_)
    
    def discount(self, date):
        return np.exp(-self.r_*date)
    
class Caplet(FixedRateEuropeanSingleOption):
    def __init__(self, R0, K, r, sigma, T, deltat, notional):
        super().__init__(R0, K, r, sigma, T, 'c', deltat, notional)
    
class Floorlet(FixedRateEuropeanSingleOption):
    def __init__(self, R0, K, r, sigma, T, deltat, notional):
        super().__init__(R0, K, r, sigma, T, 'p', deltat, notional)

class Cap(Option):
    def __init__(self, R0, K, r, sigma, T, notional, deltat=1/2, freq='Annual'):
        #TODO: Usar frecuencias en el futuro
        #TODO:Sigma podría ser un entero o una lista de enteros para valorar con volas forward
        #TODO: Hacer caps amortising(lista de notionals)
        #TODO: Hacer caps in arrears
        #TODO: que los deltat puedan ser una lista
        self.caps_ = [Caplet(R0, K, r, sigma, t, deltat, notional) for t in np.linspace(0, T, round(T/deltat))[1:]]
        self.models = {'BS': self.priceBS, 'Bachelier':self.priceBachelier}
    
    def priceBS(self, simulation=False, n_sims=10000):
        return np.sum( [cap.priceBS(simulation=simulation, n_sims=n_sims) for cap in self.caps_] )
    
    def priceBachelier(self, simulation=False, n_sims=10000):
        return np.sum( [cap.priceBachelier(simulation=simulation, n_sims=n_sims) for cap in self.caps_] )
    
class Floor(Option):
    def __init__(self, R0, K, r, sigma, T, deltat=1/2, freq='Annual'):
        #TODO: Usar frecuencias en el futuro
        self.floors_ = [Floorlet(R0, K, r, sigma, t, deltat) for t in np.linspace(0, T, round(T/deltat))[1:]]
        self.models = {'BS': self.priceBS, 'Bachelier':self.priceBachelier}
    
    def priceBS(self, simulation=False, n_sims=10000):
        return np.sum( [floor.priceBS(simulation=simulation, n_sims=n_sims) for floor in self.floors_] )
    
    def priceBachlier(self, simulation=False, n_sims=10000):
        return np.sum( [floor.priceBachelier(simulation=simulation, n_sims=n_sims) for floor in self.floors_] )

#TODO: stripping de vola usando getImpliedVol de los caps a distintos plazos

#TODO: swaptions

#TODO: crear constantes de clase genéricas para frecuencias

if __name__ == '__main__':
    r = 0.05
    sigma = 0.2
    T = 1

    ##### Test modelo bachelier con caplets y swaptions
    notional = 1000000
    deltat = 1
    R0 = 0.01
    strike_r = 0.01

    frop = Caplet(R0, strike_r, r, sigma, T, deltat, notional)
    
    print("Opciones caplet: ")
    print("Con Black Scholes: ")
    print('Analítico: ', frop.priceBS())
    print('MonteCarlo: ', frop.priceBS(simulation=True, n_sims=1000000))
    
    priceBach = frop.priceBachelier(sigma*R0)
    print("Con Bachelier(otra vola): ")
    print('Analítico: ',frop.price(model='Bachelier', sigma=sigma*R0))
    print('MonteCarlo: ',frop.price(model='Bachelier',sigma=sigma*R0, simulation=True, n_sims=1000000))

    print(frop.getImpliedVol(priceBach, model='Bachelier'))

    ## TODO: comprobar caps y floors con excel

    ## TODO: comprobar swaptions
    
    ############ Prueba valoración swap con examen del master #####################

    fv = 45657
    fechas_estr = [45657,45663,45667,45674,45691,45719,45750,45782,45811,45841,45873,45903,45933,45964,45994,46027,46206,46391,46755,47121,47486,47851,48218,48582,48947,49312,49677,50045,51138,52965,54791,56618,60272,63922]
    valores_estr = [1,0.999516,0.999192,0.998626,0.997255,0.995199,0.99314,0.991182,0.989529,0.987915,0.986231,0.984684,0.983171,0.981661,0.980203,0.978596,0.969929,0.960734,0.941642,0.921557,0.901559,0.88111,0.860384,0.840072,0.820523,0.798701,0.778146,0.757801,0.700782,0.626543,0.57159,0.527079,0.456181,0.401673]
    estr = InterestRateCurve.from_dates(fv, fechas_estr, valores_estr, curve_type="discount_factor")

    fechas_eur6M = [45657,45658,45659,45666,45673,45680,45691,45719,45749,45779,45810,45840,46024,46205,46391,46755,47120,47485,47850,48215,48582,48946,49311,51137,52964,54791,56618,58442,60269,62095,63921]
    valores_eur6m = [1,0.999918062269897,0.999836131253586,0.999282362277824,0.998721447149518,0.99810109884343,0.997181412737011,0.995051591517705,0.992910579958377,0.990901502704203,0.988906380827598,0.986543012243062,0.97648884585673,0.966919103253629,0.956792576017485,0.936296353160604,0.915277061902381,0.894434634667742,0.873359080366994,0.852359507361842,0.831563699487058,0.811287877313173,0.790774820106363,0.696767382654296,0.626822332676576,0.57596634982958,0.535855347717537,0.499599747977576,0.469777072011112,0.446121298194756,0.426638948270295]
    eur6m = InterestRateCurve.from_dates(fv, fechas_eur6M, valores_eur6m, curve_type="discount_factor")

    fixed_coupon = 0.0221 #2.18% en el excel del examen, forward ligeramente distintos

    initial_date = 45819
    end_date = 47280

    frequency_float = Frequency.SEMIANNUAL
    frequency_fixed = Frequency.SEMIANNUAL

    fixing_lag = 2

    convention_float = DayCountConvention.ACT_360
    convention_fixed = DayCountConvention.CONV_30_360

    business_day_convention = BusinessDayConvention.MODIFIED_FOLLOWING
    calendar_convention =CalendarAdjustmentConvention.UNADJUSTED

    swap = InterestRateSwapFixedVSFloating.shortDefinition(fv, eur6m, fixed_coupon, initial_date, end_date, frequency_float,frequency_fixed=frequency_fixed,
                                                         convention_fixed=convention_fixed, convention_float=convention_float, business_day_convention=business_day_convention,
                                                         calendar_convention=calendar_convention, risk_free_curve=estr, fixing_lag=fixing_lag)
    
    print('Swap valuation should be near 0:')
    print(swap.valuate())

