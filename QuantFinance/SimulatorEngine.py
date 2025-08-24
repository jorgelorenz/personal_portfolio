from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as opt
from scipy.interpolate import RegularGridInterpolator, interp1d

class CalibrationStatus:
    NOT_CALIBRATED = 'NOT_CALIBRATED'
    CALIBRATED = 'CALIBRATED'
    CALIBRATED_CAP_FLOORLET = 'CALIBRATED_CAP_FLOORLET'
    CALIBRATED_SWAPTION = 'CALIBRATED_SWAPTION'

#######  Simuladores ########

class Simulator(ABC):
    @abstractmethod
    def __init__(self, n_vars, correlMatrix=None):
        super().__init__()
        self.n_vars_ = n_vars
        self.correlMatrix_ = correlMatrix
        if n_vars > 1 and correlMatrix is None:
            pass #TODO: lanzar error            
        #TODO: heston has 2 vars, vol and underlying

    def simulate(self, dates, n_sims):
        ##TODO: Dates should be a list of positive floats
        if len(dates) > 1:
            deltas = np.array(dates[1:])-np.array(dates[:-1])
            deltas = np.append(deltas, dates[0])
        else:
            deltas = np.array(dates)
        
        if self.n_vars_ == 1:
            return self.simulateWithNoise(dates, n_sims, np.random.standard_normal(  (n_sims, len(deltas))))
        
        elif self.n_vars_ > 1:
            L = np.linalg.cholesky(self.correlMatrix_)
            Z = np.random.standard_normal((n_sims, len(deltas), self.n_vars_))
            Z_correladas = Z @ L.T
            return Z_correladas
    
    @abstractmethod
    def simulateWithNoise(self, dates, n_sims, noise):
        #TODO: check noise is in shape n_sims x dates
        
        pass

class ShiftedLognormalSimulator(Simulator):
    def __init__(self, S0, r, q, sigma, shift, n_assets=1, correlMatrix=None):
        super().__init__(n_vars=n_assets, correlMatrix=correlMatrix)
        if S0 + shift < 0:
            #TODO: lanzar excepción
            pass 

        if n_assets > 1 and correlMatrix is None:
            pass #TODO: lanzar excepción
    
        if n_assets == 1:
            self.S0_ = S0
            self.r_ = r
            self.q_ = q
            self.sigma_ = sigma
            self.shift_ = shift
        else:
            self.S0_ = np.array(S0)
            self.r_ = np.array(r)
            self.q_ = np.array(q)
            self.sigma_ = np.array(sigma)
            self.shift_ = np.array(shift)

    def simulateWithNoise(self, dates, n_sims, noise):
        #TODO: check noise is in shape n_sims x dates
        if len(dates) > 1:
            deltas = np.array(dates[1:])-np.array(dates[:-1])
            deltas = np.append(deltas, dates[0])
        else:
            deltas = np.array(dates)

        if self.n_vars_ == 1:
            lognorm = np.exp( (self.r_ - self.q_- self.sigma_**2/2)*deltas+ np.multiply( noise, np.tile(np.sqrt(deltas), (n_sims,1)))*self.sigma_)
            observations = (self.S0_+self.shift_) * np.cumprod( lognorm, axis=1 ) - self.shift_
            return observations
        else:
            sigma = self.sigma_.reshape(1, 1, -1)
            r = self.r_.reshape(1, 1, -1)
            q = self.q_.reshape(1, 1, -1)
            S0 = self.S0_.reshape(1, 1, -1)
            shift = self.shift_.reshape(1, 1, -1)
            deltas = deltas.reshape(1, -1, 1)
            sqrt_deltas = np.sqrt(deltas)

            drifts = (r - q - 0.5 * sigma ** 2) * deltas
            diffusion = noise * sigma * sqrt_deltas
            lognorm = np.exp(drifts + diffusion)

            observations = (S0 + shift) * np.cumprod(lognorm, axis=1) - shift
            return observations

            '''
            @njit(parallel=True)
            def parallel_cumprod(lognorm):
                n_sims, n_steps, n_assets = lognorm.shape
                result = np.empty_like(lognorm)
                for i in prange(n_sims):
                    for j in prange(n_assets):
                        acc = 1.0
                        for t in range(n_steps):
                            acc *= lognorm[i, t, j]
                            result[i, t, j] = acc
                return result
            '''
    
class BlackScholesSimulator(ShiftedLognormalSimulator):
    def __init__(self, S0, r, q, sigma, n_assets=1):
        super().__init__(S0, r, q, sigma, shift=0, n_assets=n_assets)

class Black76Simulator(BlackScholesSimulator):
    def __init__(self, S0, sigma):
        super().__init__(S0, 0, 0, sigma)

class BachelierSimulator(Simulator):
    def __init__(self, S0, r, q, sigma, n_assets=1):
        super().__init__(n_vars=n_assets)
        if n_assets == 1:
            self.S0_ = S0
            self.r_ = r
            self.q_ = q
            self.sigma_ = sigma
        else:
            self.S0_ = np.array(S0)
            self.r_ = np.array(r)
            self.q_ = np.array(q)
            self.sigma_ = np.array(sigma)
    
    def simulateWithNoise(self, dates, n_sims, noise):
        #Dates should be a list of floats
        if len(dates) > 1:
            deltas = np.array(dates[1:])-np.array(dates[:-1])
            deltas = np.insert(deltas, 0, dates[0])
        else:
            deltas = np.array(dates)
        
        if self.n_vars_ == 1:
            norm = (self.r_ - self.q_)*deltas+ np.multiply(noise, np.tile(np.sqrt(deltas), (n_sims,1)))*self.sigma_
            observations = self.S0_ + np.cumsum( norm, axis=1 )

            return observations 
        else:
        # Asegurar que noise tiene shape correcto
            if noise.shape != (n_sims, len(dates), self.n_vars_):
                #TODO: lanzar error en ingles
                raise ValueError(f"Shape esperado del ruido: ({n_sims}, {len(dates)}, {self.n_vars_}), recibido: {noise.shape}")
            
            # Preparar parámetros en forma (1, 1, n_vars) para broadcasting
            sigma = self.sigma_.reshape(1, 1, -1)
            r = self.r_.reshape(1, 1, -1)
            q = self.q_.reshape(1, 1, -1)
            S0 = self.S0_.reshape(1, 1, -1)

            deltas = deltas.reshape(1, -1, 1)  # shape (1, T, 1)
            sqrt_deltas = np.sqrt(deltas)     # shape (1, T, 1)

            # Calcular incrementos normales
            norm = (r - q) * deltas + noise * sigma * sqrt_deltas

            # Acumular con suma para obtener paths
            observations = S0 + np.cumsum(norm, axis=1)
            return observations
    
class DupireSimulator(Simulator):
    def __init__(self, S0, r_dicts, q_dicts, n_assets=1, correlMatrix=None):
        
        # TODO: check if vols_surfaces is a list of 2D arrays
        # TODO: check if S0 , etc are lists of the same length as n_assets

        super().__init__(n_vars=n_assets, correlMatrix=correlMatrix)

        self.n_assets = n_assets
        self.S0_ = np.array(S0)
        self.state_ = CalibrationStatus.NOT_CALIBRATED
        # Guardamos lista de interpoladores para cada activo
        self.vol_interpolators_ = []
        self.r_interps_ = []
        self.q_interps_ = []


    def simulateWithNoise(self, dates, n_sims, noise):
        #TODO: check calibrated
        pass

    def calibrateLocalVol(self, market_data, strikes_list, tenors_list):
        pass

#TODO: Meter simuladores local vol, vol estocástica, garch, arima...
#TODO: Simuladores multi-asset con matriz de correl

class GarchSimulator(Simulator):
    pass

#######  Simuladores de Árbol ########
#TODO: Meter simuladores en árbol para que devuelva datos intermedios y poder valorar americanas, bermuda, etc

class TreeSimulator:
    def __init__(self):
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

        self.isJumpConstant_ = True

    def getDeltat(self):
        return self.deltat_

    @abstractmethod
    def getProb(self):
        pass

    @abstractmethod
    def getJump(self):
        pass
    
    def simulate(self, dates_integer, n_sims=10000):
        #TODO: dates should be a list of ascendent integers which we assume n-th step of the simulation,
        #time is i*deltat

        if self.isJumpConstant_:
            u,d = self.getJump(0)
            sims = np.random.random((n_sims, dates_integer[-1]))
            observations = self.S0_*np.cumprod(np.hstack((np.ones(shape=(n_sims,1)), self.aux_fun_u(sims)*u+self.aux_fun_d(sims)*d)), axis=1)
            
            ret = np.zeros((n_sims, len(dates_integer)))
            i = 0
            for d in dates_integer:
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

        if self.isJumpConstant_:
            u,d = self.getJump(0)

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

        if isinstance(sigma, float):
            self.isJumpConstant_ = True
        elif isinstance(sigma, dict):
            self.isJumpConstant_ = False
        else:
            #TODO: lanzar error
            pass
        

    def getJump(self, date):
        deltat = self.deltat_
        if self.isJumpConstant_:
            u = np.exp((self.r_-self.q_)*deltat+self.sigma_*np.sqrt(deltat))*(2/(np.exp(self.sigma_*np.sqrt(deltat))+np.exp(-self.sigma_*np.sqrt(deltat))))
            d = np.exp((self.r_-self.q_)*deltat-self.sigma_*np.sqrt(deltat))*(2/(np.exp(self.sigma_*np.sqrt(deltat))+np.exp(-self.sigma_*np.sqrt(deltat))))
        else:
            pass
            
        return u,d

    def getProb(self):
        return 0.5

    
#TODO: Black-Derman-Toy tree simulator
#### Cosas renta fija: bonos, swaps, ... 

##### Tendré que hacer wrappers con fechas concretas que usen esto por debajo
