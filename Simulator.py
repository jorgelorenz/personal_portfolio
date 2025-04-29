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

class MultiBrownianMotion(Simulator):
    def __init__(self, correl):
        #TODO: correl should be a square matrix

        self.correl_ = correl
        self.n_undr_ = correl.shape[0]

    def simulate(self, dates, n_sims):
        #TODO:
        pass
        
class MultiAssetSimulator(Simulator):
    pass #TODO: usar movimiento browniano

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

#TODO: Meter simuladores local vol, vol estocástica, garch, arima...
#TODO: Simuladores multi-asset con matriz de correl

class GarchSimulator(Simulator):
    pass

#######  Simuladores de Árbol ########
#TODO: Meter simuladores en árbol para que devuelva datos intermedios y poder valorar americanas, bermuda, etc

class TreeSimulator(Simulator):
    def __init__(self):
        #Probabilities is a list of positive floats which sum is 1
        #TODO: check sum is 1
        pass

    @abstractmethod
    def simulate(self, dates, n_sims=10000):
        #Dates should be after init(i.e. >0)?
        #Dates are t times
        pass

    @abstractmethod
    def generateTree(self, dates):
        ### Meter suavización en último salto con BS si convergenceEnhacement
        pass

#TODO: arbol con volatilidades cambiantes -> se cambia deltat

class BinomialTreeSimulator(TreeSimulator):
    def __init__(self, S0, r, q, sigma, deltat):
        self.S0_ = S0
        self.r_ = r
        self.q_ = q
        self.sigma_ = sigma
        self.deltat_ = deltat

        self.p_ = self.getProb()

        self.aux_fun_u = np.vectorize(lambda x: 1 if x>self.p_ else 0)
        self.aux_fun_d = np.vectorize(lambda x: 0 if x>self.p_ else 1)

    def getDeltat(self):
        return self.deltat_

    @abstractmethod
    def getProb(self):
        pass

    @abstractmethod
    def getJump(self):
        pass
    
    def simulate(self, dates, n_sims=10000):
        #TODO: dates should be a list of ascendent integers which we assume n-th step of the simulation,
        #time is i*deltat

        u,d = self.getJump()
        sims = np.random.random((n_sims, dates[-1]))
        observations = self.S0_*np.cumprod(np.hstack((np.ones(shape=(n_sims,1)), self.aux_fun_u(sims)*u+self.aux_fun_d(sims)*d)), axis=1)
        
        ret = np.zeros((n_sims, len(dates)))
        i = 0
        for d in dates:
            ret[:, i] = observations[:, d]
            i+=1

        return ret

    # def simulate(self):
    #     pass
    #     #TODO: simulaciones con otro formato de fechas ?

    #TODO: convergenceEnhacement=True en valoración sobre arbol

    def generateTree(self, dates):
        N = dates[-1]+1
        tree = np.zeros((N, N))
        tree[0,0] = self.S0_

        u,d = self.getJump()

        for i in range(1, N):
            for j in range(i+1):
                if j==0:
                    tree[j,i] = tree[j,i-1]*u
                else:
                    tree[j,i] = tree[j-1,i-1]*d

        return tree

class JarrowRuddTree(BinomialTreeSimulator):
    def __init__(self, S0, r, q, sigma, deltat):
        super().__init__(S0, r, q, sigma, deltat)
        self.p_ = 0.5

    def getJump(self):
        deltat = self.deltat_
        u = np.exp((self.r_-self.q_)*deltat+self.sigma_*np.sqrt(deltat))*(2/(np.exp(self.sigma_*np.sqrt(deltat))+np.exp(-self.sigma_*np.sqrt(deltat))))
        d = np.exp((self.r_-self.q_)*deltat-self.sigma_*np.sqrt(deltat))*(2/(np.exp(self.sigma_*np.sqrt(deltat))+np.exp(-self.sigma_*np.sqrt(deltat))))
        return u,d

    def getProb(self):
        return 0.5
    
#TODO: Black-Derman-Toy tree simulator

#######  PayOffs ########

class Payoff(ABC):
    @abstractmethod
    def value(self, observations, **kwargs):
        pass

    def hasDecision(self, date, deltat):
        return False

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

####### Americanas ########
class AmericanPayoff(Payoff):
    def __init__(self, K):
        self.K_ = K

    def hasDecision(self, date, deltat):
        return True
    
class AmericanCallPayOff(AmericanPayoff):
    def value(self, observation):
        return max(0, observation-self.K_)
    
class AmericanPutPayOff(AmericanPayoff):
    def value(self, observation):
        return max(0, self.K_ - observation)

####### Bermuda ##########
class BermudanPayoff(Payoff):
    def __init__(self, K, dates):
            #TODO: dates is a list with dates when decision can be taken
            self.K_ = K
            self.dates_ = np.array(dates)

    def hasDecision(self, date, deltat):
        return  (np.absolute(self.dates_ - date) < deltat/2).any()
    
class BermudanCallPayOff(BermudanPayoff):
    def value(self, observation):
        return max(0, observation-self.K_)
    
class BermudanPutPayOff(BermudanPayoff):
    def value(self, observation):
        return max(0, self.K_ - observation)

####Meter payoffs de estructurados, etc

#######  Opciones ########

class Option(ABC):
    def __init__(self):
        ### TODO: meter opciones genéricas de opciones como fijación de strike en el futuro, una función entonces para calcular s0 o ver cómo hago
        ### Forward starting etc
        self.models = dict()
        pass

    def price(self, model='BS', **kwargs):
        if model not in self.models:
            #TODO: lanzar error
            pass
        
        return self.models[model](**kwargs)
    
    def priceMC(self, simulator, payoff, dates, n_sims):
        #Meter comprobaciones de tipo de objetos
        sims = simulator.simulate(dates, n_sims=n_sims)
        if len(dates) == 1:
            return np.mean( [payoff.value( sims[i,0] ) for i in range(n_sims)])*self.maturityDiscount()
        else:
            return np.mean( [payoff.value( sims[i,:] ) for i in range(n_sims)])*self.maturityDiscount()
    
    def priceTree(self, simulator, dates, payoff, funLast=None):
        tree = simulator.generateTree(dates)
        p = simulator.getProb()
        deltat = simulator.getDeltat()

        N = tree.shape[0]

        disc = self.discountStep(deltat)

        value_tree = np.full((N,N), np.nan)
        for i in range(N):
            value_tree[i,N-1] = payoff.value( tree[i,N-1] )
        
        if funLast!= None:
            for i in range(N):
                val = funLast(tree[i, N-2])
                if payoff.hasDecision(deltat*(N-2), deltat):
                    value_tree[i, N-2] = max(payoff.value(tree[i,N-2]), val) 
                else:
                    value_tree[i, N-2] = val

            for j in range(N-3, -1, -1):
                for i in range(j+1):
                    val = (p*value_tree[i,j+1]+(1-p)*value_tree[i+1,j+1])*disc
                    if payoff.hasDecision(deltat*j, deltat):
                        value_tree[i,j] = max(payoff.value(tree[i,j]), val)
                    else:
                        value_tree[i,j] = val
        else:
            for j in range(N-2, -1, -1):
                for i in range(j+1):
                    val = (p*value_tree[i,j+1]+(1-p)*value_tree[i+1,j+1])*disc
                    if payoff.hasDecision(deltat*j, deltat):
                        value_tree[i,j] = max(payoff.value(tree[i,j]), val)
                    else:
                        value_tree[i,j] = val

        return value_tree[0,0]


    def getImpliedVol(self, price, model='BS', **kwargs):
        ## TODO: comprobar si el modelo tiene parámetro único de vola implícita
        return opt.fsolve( lambda x: (self.price(model=model, sigma=x, **kwargs) - price)**2, 0.1)[0]
    
    def getBSImpliedVol(self, price):
        return self.getImpliedVol(price, model='BS')
    
    def getBachelierImpliedVol(self, price):
        return self.getImpliedVol(price, model='Bachelier')
    
    @abstractmethod
    def maturityDiscount(self):
        pass

class OneStrikeOption(Option):
    def __init__(self, S0, K, r, q, sigma, T):
        
        #TODO: quitar la sigma como atributo de clase ? 
        self.S0_ = S0
        self.K_ = K
        self.r_ = r
        self.q_ = q
        self.T_ = T
        self.sigma_flat_ = sigma

####### Europeas ########
class EuropeanVanillaOption(OneStrikeOption):
    def __init__(self, S0, K, r, q, sigma, T, c_p):
        super().__init__(S0, K, r, q, sigma, T)
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
        self.models = {'BS': self.priceBS, 'JarrowRuddTree': self.priceJarrowRuddTree}

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
    
    def priceJarrowRuddTree(self, sigma=None, simulation=True, deltat=1/252, n_sims=10000):
        if simulation:
            return self.priceJarrowRuddTreeMC(sigma=sigma, deltat=deltat, n_sims=n_sims)
        else:
            simulator = JarrowRuddTree(self.S0_, self.r_, self.q_, sigma, deltat=deltat)
            return self.priceTree(simulator,[round(self.T_/deltat)], self.payoff_)
        
    def priceJarrowRuddTreeMC(self, sigma=None, deltat=1/252, n_sims=10000):
        if sigma == None:
            sigma = self.sigma_flat_
        
        simulator = JarrowRuddTree(self.S0_, self.r_, self.q_, sigma, deltat=deltat)

        return self.priceMC(simulator, self.payoff_, [round(self.T_/deltat)], n_sims=n_sims)

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
class AsianOption(OneStrikeOption):
    def __init__(self, S0, K, r, q, sigma, dates, T, c_p):
        super().__init__(S0, K, r, q, sigma, T)
        self.dates_ = dates
        self.simulatorBS_ = BlackScholesSimulator(self.S0_, self.r_, self.q_, self.sigma_flat_)
        if c_p == 'c':
            self.payoff_ = AsianCallPayoff(self.K_)
        elif c_p == 'p':
            self.payoff_ = AsianPutPayoff(self.K_)
        else: 
            pass #TODO: lanzar error

        self.c_p = c_p
        self.models = {'BS': self.priceBS}

    def priceBS(self, sigma=None, n_sims=10000):
        #TODO: calcular con integración numérica de número de observaciones  ?
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

class DigitalOption(OneStrikeOption):
    def __init__(self, S0, K, r, q, sigma, T, c_p):
        super().__init__(S0, K, r, q, sigma, T)
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
        self.models = {'BS': self.priceBS}

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

class BarrierOption(OneStrikeOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T, c_p, u_d, activation=True):
        super().__init__(S0, K, r, q, sigma, T)
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
        self.models = {'BS': self.priceBS}

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
class FutureEuropeanVanillaOption(OneStrikeOption):
    def __init__(self, F0, K, r, sigma, T, c_p):
        super().__init__(F0, K, r, q, sigma, T)
        self.F0_ = F0
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
        self.models = {'BS': self.priceBS}

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

class FixedRateEuropeanSingleOption(FutureEuropeanVanillaOption):

    def __init__(self, R0, K, r, sigma, T, c_p, deltat, notional):
        #TODO: meter shifted log normal
        super().__init__(R0, K, r, sigma, T, c_p)
        self.deltat_ = deltat
        self.notional_ = notional
        self.simulatorBach_ = BachelierSimulator(self.F0_, 0, 0, sigma=self.sigma_flat_)
        self.models = {'BS': self.priceBS, 'Bachelier':self.priceBachelier}

    def priceBS(self, sigma=None, simulation=False, n_sims=10000):
        #comprobado que usa el maturity discount de esta clase
        return super().priceBS(sigma, simulation, n_sims)*self.deltat_*self.notional_
    
    
    def priceBachelier(self, sigma=None, simulation=False, n_sims=10000):
        if sigma != None:
                simulatorBach = BachelierSimulator(self.F0_, 0, 0, sigma=sigma)
        else:
            sigma = self.sigma_flat_
            simulatorBach = self.simulatorBach_

        d = (self.F0_-self.K_)/(sigma*np.sqrt(self.T_))


        if simulation:
            return self.priceMC(simulatorBach, self.payoff_, [self.T_], n_sims=n_sims)*self.deltat_*self.notional_
        else:
            w = 1 if self.c_p == 'c' else -1
            return self.notional_*self.deltat_*self.maturityDiscount()*(w*(self.F0_-self.K_)*st.norm.cdf(w*d)+sigma*np.sqrt(self.T_)*st.norm.pdf(d))
    
    def maturityDiscount(self):
        return np.exp(-self.r_ * (self.T_+self.deltat_))
    
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

######## Americanas #######

class AmericanVanillaOption(OneStrikeOption):
    def __init__(self, S0, K, r, q, sigma, T, c_p):
        super().__init__(S0, K, r, q, sigma, T)
        if c_p=='c':
            self.payoff_ = AmericanCallPayOff(self.K_)
        elif c_p=='p':
            self.payoff_ = AmericanPutPayOff(self.K_)
        else:
            #TODO: lanzar error
            pass

        self.c_p = c_p
        self.models = {'JarrowRuddTree': self.priceJarrowRuddTree}
    
    def price(self, model='JarrowRuddTree', **kwargs):
        return super().price(model, **kwargs)
    
    def priceJarrowRuddTree(self, sigma=None, deltat=1/252):
        simulator = JarrowRuddTree(self.S0_, self.r_, self.q_, sigma, deltat=deltat)
        return self.priceTree(simulator, [round(self.T_/deltat)], self.payoff_)

    def maturityDiscount(self):
        return np.exp(-self.r_*self.T_)
    
    def discountStep(self,deltat):
        return np.exp(-self.r_*deltat)
    
class AmericanCallOption(AmericanVanillaOption):
    def __init__(self, S0, K, r, q, sigma, T):
        super().__init__(S0, K, r, q, sigma, T, c_p='c')

class AmericanPutOption(AmericanVanillaOption):
    def __init__(self, S0, K, r, q, sigma, T):
        super().__init__(S0, K, r, q, sigma, T, c_p='p')

######## Bermuda ##########
class BermudanVanillaOption(OneStrikeOption):
    def __init__(self, S0, K, r, q, sigma, T, dates, c_p):
        super().__init__(S0, K, r, q, sigma, T)
        self.dates_ = dates
        if c_p=='c':
            self.payoff_ = BermudanCallPayOff(self.K_, dates)
        elif c_p=='p':
            self.payoff_ = BermudanPutPayOff(self.K_, dates)
        else:
            #TODO: lanzar error
            pass

        self.c_p = c_p
        self.models = {'JarrowRuddTree': self.priceJarrowRuddTree}
    
    def price(self, model='JarrowRuddTree', **kwargs):
        return super().price(model, **kwargs)
    
    def priceJarrowRuddTree(self, sigma=None, deltat=1/252):
        simulator = JarrowRuddTree(self.S0_, self.r_, self.q_, sigma, deltat=deltat)
        return self.priceTree(simulator, [round(self.T_/deltat)], self.payoff_)

    def maturityDiscount(self):
        return np.exp(-self.r_*self.T_)
    
    def discountStep(self,deltat):
        return np.exp(-self.r_*deltat)
    
class BermudanCallOption(BermudanVanillaOption):
    def __init__(self, S0, K, r, q, sigma, T, dates):
        super().__init__(S0, K, r, q, sigma, T, dates, c_p='c')

class BermudanPutOption(BermudanVanillaOption):
    def __init__(self, S0, K, r, q, sigma, T, dates):
        super().__init__(S0, K, r, q, sigma, T, dates, c_p='p')

######## Estructurados ####

#### Cosas renta fija: bonos, swaps, ... 

##### Tendré que hacer wrappers con fechas concretas que usen esto por debajo

if __name__=='__main__':
    n_sims = 100000
    spot = 100
    strike = 100
    r = 0.05
    q = 0.01
    sigma = 0.2
    T = 1


    ##### Test europeas
    op = EuropeanCallOption(spot, strike , r, q, sigma, T)
    op2 = EuropeanPutOption(spot, strike , r, q, sigma, T)

    price = op.price(model='BS', sigma=0.2)

    print('Opciones europeas:')
    print(op.priceBS(0.2))
    print(op.priceBS(0.2, simulation=True, n_sims=n_sims))
    print(op2.priceBS(0.2))
    print(op2.priceBS(0.2, simulation=True, n_sims=n_sims))
    print(op.getImpliedVol(price))


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
    notional = 1000000
    deltat = 1
    R0 = 0.01
    strike_r = 0.01

    frop = Caplet(R0, strike_r, r, sigma, T, deltat, notional)
    
    print("Opciones caplet: ")
    print("Con Black Scholes: ")
    print(frop.priceBS())
    print(frop.priceBS(simulation=True, n_sims=1000000))
    
    priceBach = frop.priceBachelier(sigma*R0)
    print("Con Bachelier(otra vola): ")
    print(frop.price(model='Bachelier', sigma=sigma*R0))
    print(frop.price(model='Bachelier',sigma=sigma*R0, simulation=True, n_sims=1000000))

    print(frop.getImpliedVol(priceBach, model='Bachelier'))

    ## TODO: comprobar caps y floors con excel

    ## TODO: comprobar swaptions

    #Test árbol
    print('JarrowRudd simulator:')
    print(JarrowRuddTree(spot, r, q, sigma, deltat=1/252).simulate(dates=[5], n_sims=1))
    
    print('Valoración simulación árbol:')
    print('Black-scholes teórico: ', op.priceBS(0.2))
    print(op.price(model='JarrowRuddTree', sigma=0.2, simulation=True, n_sims=n_sims))

    print('Valoración en árbol:')
    print(op.price(model='JarrowRuddTree', sigma=0.2, n_sims=n_sims))

    #Test americanas: comprobar con un excel
    print('Valoración call americana:')
    op = EuropeanCallOption(spot, strike , r, 0, sigma, T)
    op2 = EuropeanPutOption(spot, strike , r, 0, sigma, T)

    aop = AmericanCallOption(spot, strike, r, 0, sigma, T)

    print('Europea: ',op.priceBS(0.2))

    print('Americana: ',aop.price(model='JarrowRuddTree', deltat=1/252, sigma=0.2))

    print('Valoración put americana:')
    aop2 = AmericanPutOption(spot, strike, r, 0, sigma, T)
    
    price =aop2.price(model='JarrowRuddTree', deltat=1/12, sigma=0.2)
    print('Europea: ', op2.priceBS(0.2))
    print('Americana: ', aop2.price(model='JarrowRuddTree', deltat=1/252, sigma=0.2))

    print('Vola implícita: ', aop2.getImpliedVol(price, model='JarrowRuddTree', deltat=1/12))

    #Test bermuda
    bermop = BermudanCallOption(spot, strike, r, 0, sigma, T, [0.5])
    bermop2 = BermudanPutOption(spot, strike, r, 0, sigma, T, [0.5])
    
    print('Bermudas con fecha de ejercicio intermedia en 0.5: ')
    print('Call: ')
    print('Europea: ',op.priceBS(0.2))
    print('Bermuda: ',bermop.price(model='JarrowRuddTree', deltat=1/12, sigma=0.2))

    print('Put: ')
    print('Europea: ',op2.priceBS(0.2))
    print('Bermuda: ',bermop2.price(model='JarrowRuddTree', deltat=1/12, sigma=0.2))

    print('Bermudas con fechas de ejercicio 0.25, 0.5, 0.75: ')
    bermop = BermudanCallOption(spot, strike, r, 0, sigma, T, [0.25, 0.5, 0.75])
    bermop2 = BermudanPutOption(spot, strike, r, 0, sigma, T, [0.25, 0.5, 0.75])

    print('Call: ',bermop.price(model='JarrowRuddTree', deltat=1/12, sigma=0.2))
    print('Put: ',bermop2.price(model='JarrowRuddTree', deltat=1/12, sigma=0.2))
