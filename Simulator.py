from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as opt

#######  Simuladores ########

class Simulator(ABC):
    @abstractmethod
    def simulate(self, dates, n_sims):
        ##TODO: Dates should be a list of positive floats
        pass

class ShiftedLognormalSimulator(Simulator):
    def __init__(self, S0, r, q, sigma, shift):
        super().__init__()
        if S0 + shift < 0:
            #TODO: lanzar excepción
            pass 

        self.S0_ = S0
        self.r_ = r
        self.q_ = q
        self.sigma_ = sigma
        self.shift_ = shift

    def simulate(self, dates, n_sims):
        #TODO: Dates should be a list of floats
        if len(dates) > 1:
            deltas = np.array(dates[1:])-np.array(dates[:-1])
            deltas = np.append(deltas, dates[0])
        else:
            deltas = np.array(dates)
        lognorm = np.exp( (self.r_ - self.q_- self.sigma_**2/2)*deltas+ np.multiply(np.random.standard_normal(  (n_sims, len(deltas)) ), np.tile(np.sqrt(deltas), (n_sims,1)))*self.sigma_)
        observations = (self.S0_+self.shift_) * np.cumprod( lognorm, axis=1 ) - self.shift_

        return observations
    
class BlackScholesSimulator(ShiftedLognormalSimulator):
    def __init__(self, S0, r, q, sigma):
        super().__init__(S0, r, q, sigma, shift=0)

class Black76Simulator(BlackScholesSimulator):
    def __init__(self, S0, sigma):
        super().__init__(S0, 0, 0, sigma)

class BachelierSimulator(Simulator):
    def __init__(self, S0, r, q, sigma):
        super().__init__()
        self.S0_ = S0
        self.r_ = r
        self.q_ = q
        self.sigma_ = sigma
    
    def simulate(self, dates, n_sims):
        #Dates should be a list of floats
        if len(dates) > 1:
            deltas = np.array(dates[1:])-np.array(dates[:-1])
            deltas = np.insert(deltas, 0, dates[0])
        else:
            deltas = np.array(dates)
        norm = (self.r_ - self.q_)*deltas+ np.multiply(np.random.standard_normal(  (n_sims, len(deltas)) ), np.tile(np.sqrt(deltas), (n_sims,1)))*self.sigma_
        observations = self.S0_ + np.cumsum( norm, axis=1 )

        return observations 

#TODO: Meter simuladores local vol, vol estocástica, bachelier, garch, arima...
#TODO: Simuladores multi-asset con matriz de correl

class GarchSimulator(Simulator):
    pass

#######  Simuladores de Árbol ########
#TODO: Meter simuladores en árbol para que devuelva datos intermedios y poder valorar americanas, bermuda, etc

def TreeSimulator(Simulator):
    def __init__(self, probabilities):
        #Probabilities is a list of positive floats which sum is 1
        #TODO: check sum is 1
        self.probabilities_ = probabilities
        pass

    @abstractmethod
    def simulate(self, dates, n_sims=10000, deltat=1/252):
        #Dates should be after init(i.e. >0)?
        #Dates are t times
        pass

    @abstractmethod
    def generateTree(self, dates, n_sims=10000, deltat=1/252, convergenceEnhacement = True):
        ### Meter suavización en último salto con BS si convergenceEnhacement
        pass

#######  PayOffs ########

class Payoff(ABC):
    @abstractmethod
    def value(self, observations, **kwargs):
        pass

####### Europeas #########
class EuropeanVanillaPayoff(Payoff):
    def __init__(self, K):
        self.K_ = K

class EuropeanCallPayOff(EuropeanVanillaPayoff):
    def value(self, observation):
        return max(0, observation-self.K_)
    
class EuropeanPutPayOff(EuropeanVanillaPayoff):
    def value(self, observation):
        return max(0, self.K_ - observation)

####### Asiáticas #########
class AsianPayoff(Payoff):
    def __init__(self, K):
        self.K_ = K

class AsianCallPayoff(AsianPayoff):
    def value(self, observations):
        return max(0, np.mean(observations)-self.K_)

class AsianPutPayoff(AsianPayoff):
    def value(self, observations):
        return max(0, self.K_-np.mean(observations))

####### Digitales #########
class DigitalPayoff(Payoff):
    def __init__(self, K):
        self.K_ = K
    
class DigitalCallPayoff(DigitalPayoff):
    def value(self, observation):
        return 1 if observation > self.K_ else 0

class DigitalPutPayoff(DigitalPayoff):
    def value(self, observation):
        return 1 if observation < self.K_ else 0
    
####### Barrera en observaciones ########

class BarrierPayoff(Payoff):
    def __init__(self, K, barrier, u_d='u', activation=True):
        self.K_ = K
        self.barrier_ = barrier
        self.u_d_ = u_d
        self.activation_ = activation

    def isActiveAtMaturity(self, observations):
        mini = np.min(observations)
        maxi = np.max(observations)
        if self.activation_:
            if self.u_d_ == 'u':
                return self.barrier_ < maxi
            else:
                return self.barrier_ > mini
        else:
            if self.u_d_ == 'u':
                return self.barrier_ > maxi
            else:
                return self.barrier_ < mini
    
class BarrierCallPayoff(BarrierPayoff):
    def value(self, observations):
        if isinstance(observations, np.float64):
            last = observations
        else:
            last = observations[-1]

        if self.isActiveAtMaturity(observations):
            return max(last - self.K_, 0)
        else:
            return 0

class BarrierPutPayoff(BarrierPayoff):
    def value(self, observations):
        if isinstance(observations, np.float64):
            last = observations
        else:
            last = observations[-1]

        
        if self.isActiveAtMaturity(observations):
            return max(self.K_ - last, 0)
        else:
            return 0

##### Clases ayuda ######

##### Call #####
class DownOutCallPayoff(BarrierCallPayoff):
    def __init__(self, K, barrier):
        super().__init__(K, barrier, u_d='d', activation=False)

class DownInCallPayoff(BarrierCallPayoff):
    def __init__(self, K, barrier):
        super().__init__(K, barrier, u_d='d', activation=True)
    
class UpOutCallPayoff(BarrierCallPayoff):
    def __init__(self, K, barrier):
        super().__init__(K, barrier, u_d='u', activation=False)

class UpInCallPayoff(BarrierCallPayoff):
    def __init__(self, K, barrier):
        super().__init__(K, barrier, u_d='u', activation=True)

##### Put #####
class DownOutPutPayoff(BarrierPutPayoff):
    def __init__(self, K, barrier):
        super().__init__(K, barrier, u_d='d', activation=False)

class DownInPutPayoff(BarrierPutPayoff):
    def __init__(self, K, barrier):
        super().__init__(K, barrier, u_d='d', activation=True)
    
class UpOutPutPayoff(BarrierPutPayoff):
    def __init__(self, K, barrier):
        super().__init__(K, barrier, u_d='u', activation=False)

class UpInPutPayoff(BarrierPutPayoff):
    def __init__(self, K, barrier):
        super().__init__(K, barrier, u_d='u', activation=True)

####Meter payoffs de estructurados, etc

####### PayOff Decision (para bermuda y americanas) ######### 
#TODO: tiene que ser evaluación de un nodo, luego para valorar con esto tiene que recorrer arbol, O una listade  momentos del tiempo de simulación

#######  Opciones ########

class Option(ABC):
    def __init__(self):
        ### TODO: meter opciones genéricas de opciones como fijación de strike en el futuro, una función entonces para calcular s0 o ver cómo hago
        ### Forward starting etc
        pass

    def priceMC(self, simulator, payoff, dates, n_sims):
        #Meter comprobaciones de tipo de objetos
        sims = simulator.simulate(dates, n_sims=n_sims)
        if len(dates) == 1:
            return np.mean( [payoff.value( sims[i,0] ) for i in range(n_sims)])*self.maturityDiscount()
        else:
            return np.mean( [payoff.value( sims[i,:] ) for i in range(n_sims)])*self.maturityDiscount()
        
    
    @abstractmethod
    def maturityDiscount(self):
        pass

####### Europeas ########
class EuropeanVanillaOption(Option):
    def __init__(self, S0, K, r, q, sigma, T, c_p):
        super().__init__()
        self.S0_ = S0
        self.K_ = K
        self.r_ = r
        self.q_ = q
        self.T_ = T
        self.sigma_flat_ = sigma
        self.d1_ = (np.log(self.S0_ / self.K_) + (self.r_ - self.q_ + 0.5 * self.sigma_flat_**2) * self.T_) / (self.sigma_flat_ * np.sqrt(self.T_))
        self.d2_ = self.d1_ - self.sigma_flat_ * np.sqrt(self.T_)
        self.simulatorBS_ = BlackScholesSimulator(self.S0_, self.r_, self.q_, self.sigma_flat_)
        if c_p == 'c':
            self.payoff_ = EuropeanCallPayOff(self.K_)
        elif c_p == 'p':
            self.payoff_ = EuropeanPutPayOff(self.K_)
        else: 
            pass #TODO: lanzar error

        self.c_p = c_p  

    def getBSImpliedVol(self, price):
        return opt.fsolve( lambda x: (self.priceBS(sigma=x) - price)**2, 0.1)[0]

    def priceBS(self, sigma=None, simulation=False, n_sims=10000):
        if sigma != None:
            d1 = (np.log(self.S0_ / self.K_) + (self.r_ - self.q_ + 0.5 * sigma**2) * self.T_) / (sigma * np.sqrt(self.T_))
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
            return self.priceBSMC(sigma, n_sims=n_sims)
        else:
            return w*self.S0_ * np.exp(-self.q_ * self.T_) * st.norm.cdf(w*d1) - w*self.K_ * np.exp(-self.r_ * self.T_) * st.norm.cdf(w*d2)


    def priceBSMC(self, sigma=None, n_sims=10000):
        if sigma != None:
            simulator = BlackScholesSimulator(self.S0_, self.r_, self.q_, sigma)
        else:
            simulator = self.simulatorBS_

        return self.priceMC(simulator, self.payoff_, [self.T_], n_sims=n_sims)  
    
    def maturityDiscount(self):
        return np.exp(-self.r_*self.T_)

class EuropeanCallOption(EuropeanVanillaOption):
    def __init__(self, S0, K, r, q, sigma, T):
        super().__init__(S0, K, r, q, sigma, T, c_p='c')

class EuropeanPutOption(EuropeanVanillaOption):
    def __init__(self, S0, K, r, q, sigma, T):
        super().__init__(S0, K, r, q, sigma, T, c_p='p')

### TODO: price con superficie de vol: en tiempo solo o en tiempo y nivel subyacente 
    ### -> definición de simuladores y calibración

##Hacer un enum o así con las griegas que se pueden devolver 

######## Asiáticas ########
class AsianOption(Option):
    def __init__(self, S0, K, r, q, sigma, dates, T, c_p):
        super().__init__()
        self.S0_ = S0
        self.K_ = K
        self.r_ = r
        self.q_ = q
        self.T_ = T
        self.dates_ = dates
        self.sigma_flat_ = sigma
        self.simulatorBS_ = BlackScholesSimulator(self.S0_, self.r_, self.q_, self.sigma_flat_)
        if c_p == 'c':
            self.payoff_ = AsianCallPayoff(self.K_)
        elif c_p == 'p':
            self.payoff_ = AsianPutPayoff(self.K_)
        else: 
            pass #TODO: lanzar error

        self.c_p = c_p  

    def getBSImpliedVol(self, price):
        return opt.fsolve( lambda x: (self.priceBS(sigma=x) - price)**2, 0.1)

    def priceBS(self, sigma=None, n_sims=10000):
        if sigma == None:
            sigma = self.sigma_flat_
        return self.priceBSMC(sigma, n_sims=n_sims)

    def priceBSMC(self, sigma=None, n_sims=10000):
        if sigma != None:
            simulator = BlackScholesSimulator(self.S0_, self.r_, self.q_, sigma)
        else:
            simulator = self.simulatorBS_

        return self.priceMC(simulator, self.payoff_, self.dates_, n_sims=n_sims)
    
    def maturityDiscount(self):
        return np.exp(-self.r_*self.T_)
    
    ### TODO: price con superficie de vol: en tiempo solo o en tiempo y nivel subyacente 
    ### -> definición de simuladores y calibración
    
class AsianCallOption(AsianOption):
    def __init__(self, S0, K, r, q, sigma, dates, T):
        super().__init__(S0, K, r, q, sigma, dates, T, c_p='c')

class AsianPutOption(AsianOption):
    def __init__(self, S0, K, r, q, sigma, dates, T):
        super().__init__(S0, K, r, q, sigma, dates, T, c_p='p')

######## Digitales ########

class DigitalOption(Option):
    def __init__(self, S0, K, r, q, sigma, T, c_p):
        super().__init__()
        self.S0_ = S0
        self.K_ = K
        self.r_ = r
        self.q_ = q
        self.T_ = T
        self.sigma_flat_ = sigma
        self.d1_ = (np.log(self.S0_ / self.K_) + (self.r_ - self.q_ + 0.5 * self.sigma_flat_**2) * self.T_) / (self.sigma_flat_ * np.sqrt(self.T_))
        self.d2_ = self.d1_ - self.sigma_flat_ * np.sqrt(self.T_)
        self.simulatorBS_ = BlackScholesSimulator(self.S0_, self.r_, self.q_, self.sigma_flat_)
        if c_p == 'c':
            self.payoff_ = DigitalCallPayoff(self.K_)
        elif c_p == 'p':
            self.payoff_ = DigitalPutPayoff(self.K_)
        else: 
            pass #TODO: lanzar error

        self.c_p = c_p  

    def getBSImpliedVol(self, price):
        return opt.fsolve( lambda x: (self.priceBS(sigma=x) - price)**2, 0.1)

    def priceBS(self, sigma=None, simulation=False, n_sims=10000):
        if sigma == None:
            sigma = self.sigma_flat_
            d1 = self.d1_
            d2 = self.d2_
        else:
            d1 = (np.log(self.S0_ / self.K_) + (self.r_ - self.q_ + 0.5 * sigma**2) * self.T_) / (sigma * np.sqrt(self.T_))
            d2 = d1 - sigma * np.sqrt(self.T_)

        if simulation:
            return self.priceBSMC(sigma=sigma, n_sims=n_sims)
        else:
            return st.norm.cdf(d2) if self.c_p == 'c' else 1-st.norm.cdf(d2)

    def priceBSMC(self, sigma=None, n_sims=10000):
        if sigma != None:
            simulator = BlackScholesSimulator(self.S0_, self.r_, self.q_, sigma)
        else:
            simulator = self.simulatorBS_

        return self.priceMC(simulator, self.payoff_, [self.T_], n_sims=n_sims)
    
    def maturityDiscount(self):
        return np.exp(-self.r_*self.T_)

class DigitalCallOption(DigitalOption):
    def __init__(self, S0, K, r, q, sigma, T):
        super().__init__(S0, K, r, q, sigma, T, 'c')

class DigitalPutOption(DigitalOption):
    def __init__(self, S0, K, r, q, sigma, T):
        super().__init__(S0, K, r, q, sigma, T, 'p')

###### BarrierOption ######

class BarrierOption(Option):
    def __init__(self, S0, K, barrier, r, q, sigma, T, c_p, u_d, activation=True):
        super().__init__()
        self.S0_ = S0
        self.K_ = K
        self.r_ = r
        self.q_ = q
        self.T_ = T
        self.sigma_flat_ = sigma
        self.d1_ = (np.log(self.S0_ / self.K_) + (self.r_ - self.q_ + 0.5 * self.sigma_flat_**2) * self.T_) / (self.sigma_flat_ * np.sqrt(self.T_))
        self.d2_ = self.d1_ - self.sigma_flat_ * np.sqrt(self.T_)
        self.simulatorBS_ = BlackScholesSimulator(self.S0_, self.r_, self.q_, self.sigma_flat_)
        if c_p == 'c':
            self.payoff_ = BarrierCallPayoff(self.K_, barrier, u_d=u_d, activation=activation)
        elif c_p == 'p':
            self.payoff_ = BarrierPutPayoff(self.K_, barrier, u_d=u_d, activation=activation)
        else: 
            pass #TODO: lanzar error

        self.c_p = c_p

    def priceBS(self, dates, sigma=None, n_sims=10000):
        return self.priceBSMC(self, dates, sigma=sigma, n_sims=n_sims)
    
    def priceBSMC(self, dates, sigma=None, n_sims=10000):
        ### Dates should be a list and contain maturity at last element
        if abs(dates[-1] - self.T_) > 0.001:
            #TODO: lanzar error
            pass

        if sigma != None:
            simulator = BlackScholesSimulator(self.S0_, self.r_, self.q_, sigma)
        else:
            simulator = self.simulatorBS_
        return self.priceMC(simulator, self.payoff_, dates, n_sims=n_sims)
    
    def maturityDiscount(self):
        return np.exp(-self.r_*self.T_)
    
class BarrierCallOption(BarrierOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T, u_d, activation=True):
        super().__init__(S0, K, barrier, r, q, sigma, T, c_p='c', u_d=u_d, activation=activation)
        

class BarrierPutOption(BarrierOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T, u_d, activation=True):
        super().__init__(S0, K, barrier, r, q, sigma, T, c_p='p', u_d=u_d, activation=activation)

##### Clases ayuda ######

##### Call #####
class DownOutCallPayoff(BarrierCallOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T):
        super().__init__(S0, K, barrier, r, q, sigma, T, u_d='d', activation=False)

class DownInCallPayoff(BarrierCallOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T):
        super().__init__(S0, K, barrier, r, q, sigma, T, u_d='d', activation=True)
    
class UpOutCallPayoff(BarrierCallOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T):
        super().__init__(S0, K, barrier, r, q, sigma, T, u_d='u', activation=False)

class UpInCallPayoff(BarrierCallOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T):
        super().__init__(S0, K, barrier, r, q, sigma, T, u_d='u', activation=True)

##### Put #####
class DownOutPutPayoff(BarrierPutOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T):
        super().__init__(S0, K, barrier, r, q, sigma, T, u_d='d', activation=False)

class DownInPutPayoff(BarrierPutOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T):
        super().__init__(S0, K, barrier, r, q, sigma, T, u_d='d', activation=True)
    
class UpOutPutPayoff(BarrierPutOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T):
        super().__init__(S0, K, barrier, r, q, sigma, T, u_d='u', activation=False)

class UpInPutPayoff(BarrierPutOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T):
        super().__init__(S0, K, barrier, r, q, sigma, T, u_d='u', activation=True)

##### Opciones vanilla sobre futuros ####
class FutureEuropeanVanillaOption(Option):
    def __init__(self, F0, K, r, sigma, T, c_p):
        super().__init__()
        self.F0_ = F0
        self.K_ = K
        self.r_ = r
        self.T_ = T
        self.sigma_flat_ = sigma
        self.d1_ = (np.log(self.F0_ / self.K_) + (0.5 * self.sigma_flat_**2) * self.T_) / (self.sigma_flat_ * np.sqrt(self.T_))
        self.d2_ = self.d1_ - self.sigma_flat_ * np.sqrt(self.T_)
        self.simulatorBS_ = Black76Simulator(self.F0_, self.sigma_flat_)
        if c_p == 'c':
            self.payoff_ = EuropeanCallPayOff(self.K_)
        elif c_p == 'p':
            self.payoff_ = EuropeanPutPayOff(self.K_)
        else: 
            pass #TODO: lanzar error

        self.c_p = c_p  

    def getBSImpliedVol(self, price):
        return opt.fsolve( lambda x: (self.priceBS(sigma=x) - price)**2, 0.1)[0]

    def priceBS(self, sigma=None, simulation=False, n_sims=10000):
        if sigma != None:
            d1 = (np.log(self.F0_ / self.K_) + (0.5 * sigma**2) * self.T_) / (sigma * np.sqrt(self.T_))
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
            return self.priceBSMC(sigma, n_sims=n_sims)
        else:
            return  w*self.maturityDiscount()*(self.F0_ * st.norm.cdf(w*d1) - self.K_ * st.norm.cdf(w*d2))


    def priceBSMC(self, sigma=None, n_sims=10000):
        if sigma != None:
            simulator = BlackScholesSimulator(self.F0_, 0, 0, sigma)
        else:
            simulator = self.simulatorBS_

        return self.priceMC(simulator, self.payoff_, [self.T_], n_sims=n_sims)  
    
    def maturityDiscount(self):
        return np.exp(-self.r_*self.T_)

class FutureEuropeanCallOption(FutureEuropeanVanillaOption):
    def __init__(self, S0, K, r, sigma, T):
        super().__init__(S0, K, r, sigma, T, c_p='c')

class FutureEuropeanPutOption(FutureEuropeanVanillaOption):
    def __init__(self, S0, K, r, sigma, T):
        super().__init__(S0, K, r, sigma, T, c_p='p')

######## Caps, Floors y Swaptions #######

######## Americanas #######

######## Bermuda ##########

######## Estructurados ####

#### Cosas renta fija: bonos, swaps, ... 

##### Tendré que hacer wrappers con fechas concretas que usen esto por debajo

if __name__=='__main__':
    n_sims = 100000
    spot = 100
    strike = 100
    r = 0.02
    q = 0.01
    sigma = 0.2
    T = 1


    ##### Test europeas
    op = EuropeanCallOption(spot, strike , r, q, sigma, T)
    op2 = EuropeanPutOption(spot, strike , r, q, sigma, T)

    price = op.priceBS(0.2)

    print('Opciones europeas:')
    print(op.priceBS(0.2))
    print(op.priceBS(0.2, simulation=True, n_sims=n_sims))
    print(op2.priceBS(0.2))
    print(op2.priceBS(0.2, simulation=True, n_sims=n_sims))
    print(op.getBSImpliedVol(price))


    ##### Test asiáticas
    dates = [1]
    aop = AsianCallOption(spot, strike , r, q, sigma, dates, T)
    aop2 = AsianPutOption(spot, strike , r, q, sigma, dates, T)

    print('Opciones asiáticas con solo fecha 1:')
    print(aop.priceBS(n_sims=n_sims))
    print(aop2.priceBS(n_sims=n_sims))

    dates = [0.25,0.5,0.75,1]
    aop = AsianCallOption(spot, strike , r, q, sigma, dates, T)
    aop2 = AsianPutOption(spot, strike , r, q, sigma, dates, T)
    print('Opciones asiáticas con fechas 0.25,0.5,0.75,1:')
    print(aop.priceBS(n_sims=n_sims))
    print(aop2.priceBS(n_sims=n_sims))


    ##### Test digitales
    strike = 120
    dop = DigitalCallOption(spot, strike , r, q, sigma, T)
    dop2 = DigitalPutOption(spot, strike , r, q, sigma, T)

    print('Opciones digitales:')
    print(dop.priceBS(0.2))
    print(dop.priceBS(0.2, simulation=True, n_sims=n_sims))
    print(dop2.priceBS(0.2))
    print(dop2.priceBS(0.2, simulation=True, n_sims=n_sims))

    ##### Test opciones sobre futuros
    strike = 120
    fop = FutureEuropeanCallOption(spot, strike, r, sigma, T)
    fop2 = FutureEuropeanPutOption(spot, strike, r, sigma, T)

    print("Opciones sobre futuros:")
    print(fop.priceBS(0.2))
    print(fop.priceBS(0.2, simulation=True, n_sims=n_sims*3))
    print(fop2.priceBS(0.2))
    print(fop2.priceBS(0.2, simulation=True, n_sims=n_sims*3))

    ##### Test barrera
    strike = 100
    barrier = 120

    bop = UpInCallPayoff(spot, strike, barrier, r, q, sigma, T)
    aux11 = DigitalCallOption(spot, barrier, r, q, sigma, T)
    aux12 = EuropeanCallOption(spot, barrier, r,q,sigma,T)

    barrier2 = 80
    bop2 = DownInPutPayoff(spot, strike, barrier2, r, q, sigma, T)
    aux21 = DigitalPutOption(spot, barrier2, r, q, sigma, T)
    aux22 = EuropeanPutOption(spot, barrier2, r,q,sigma,T)

    print("Opciones con barrera: ")
    print("Call up&in a vencimiento es como digital+call en la barrera:")
    print(bop.priceBSMC([T], n_sims=n_sims*3))
    print( (barrier-strike)*aux11.priceBS( )+aux12.priceBS())

    print("Put down&in a vencimiento es como digital+call en la barrera:")
    print(bop2.priceBSMC([T], n_sims=n_sims*3))
    print((strike-barrier2)*aux21.priceBS( )+aux22.priceBS())

    ##### Test modelo bachelier con caplets y swaptions