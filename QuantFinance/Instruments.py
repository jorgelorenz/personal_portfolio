from abc import ABC, abstractmethod
import numpy as np
import scipy.optimize as opt


class Instrument:
    def __init__(self, valuation_date):
        self.valuation_date_ = valuation_date   

class Option(ABC, Instrument):
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
    
    #Cosas que añadir: fechas pago != fechass_observacion , cómo hacerlo? en la opción se pone tal cosa. se simula a observaciones
    #y se descuenta a pagos
    #Algunos datos pueden estar ya fijados
    #Crear una clase con info de mercado(monedas, subyacentes(volas, dividendos), curvas(y fijaciones),
    #correlaciones: de ahí se sacan los tipos, volas, dividendos
    def priceMC(self, simulator, payoff, dates, n_sims, dates_obs=None, **kwargs):
        #Meter comprobaciones de tipo de objetos
        if dates_obs == None:
            dates_obs = dates

        discounts = [self.discount(date) for date in dates]
        sims = simulator.simulate(dates_obs, n_sims=n_sims)
        
        if len(dates) == 1:
            return np.mean( np.dot([payoff.cash_flows(sims[i,0], **kwargs) for i in range(n_sims)], discounts) )
        else:
            return np.mean( np.dot([payoff.cash_flows(sims[i,:], **kwargs) for i in range(n_sims)], discounts) )
        
    def priceTreeMC(self, simulator, payoff, dates, n_sims, **kwargs):
        #Meter comprobaciones de tipo de objetos
        deltat = simulator.getDeltat()
        dates_integer = [round(date/deltat) for date in dates]
        discounts = [self.discount(date) for date in dates]
        sims = simulator.simulate(dates_integer, n_sims=n_sims)
        
        if len(dates) == 1:
            return np.mean( np.dot([payoff.cash_flows(sims[i,0], **kwargs) for i in range(n_sims)], discounts) )
        else:
            return np.mean( np.dot([payoff.cash_flows(sims[i,:], **kwargs) for i in range(n_sims)], discounts) )

        # if len(dates) == 1:
        #     return np.mean( [payoff.value( sims[i,0], **kwargs ) for i in range(n_sims)])*self.maturityDiscount()
        # else:
        #     return np.mean( [payoff.value( sims[i,:], **kwargs ) for i in range(n_sims)])*self.maturityDiscount()
    
    def priceTree(self, simulator, dates, payoff, funLast=None):
        tree = simulator.generateTree(dates)
        p = simulator.getProb()
        deltat = simulator.getDeltat()

        N = tree.shape[0]

        disc = self.discount(deltat)

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
    
    def getBSImpliedVol(self, price, **kwargs):
        return self.getImpliedVol(price, model='BS', **kwargs)
    
    def getBachelierImpliedVol(self, price):
        return self.getImpliedVol(price, model='Bachelier')
    
    @abstractmethod
    def maturityDiscount(self):
        pass

    @abstractmethod
    def discount(self, date):
        pass
    

#######  Opciones ########

class OneStrikeOption(Option):
    def __init__(self, S0, K, r, q, sigma, T):
        
        #TODO: quitar la sigma como atributo de clase ? 
        self.S0_ = S0
        self.K_ = K
        self.r_ = r
        self.q_ = q
        self.T_ = T
        self.sigma_flat_ = sigma