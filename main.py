import numpy as np
import simulation


class MonteCarlo:

    # setup monte carlo variables
    def __init__(self,
            simulationTime,
            numberOfFirms,
            numberOfBanks,
            alpha, # mean of firms price
            varpf, # variance of firms price
            gamma, # interest rate parameter
            chi, # number of potential partners on credit market
            lambd, # intensity of choice
            adj, # leverage adjustment
            phi, # production function parameter
            beta, # production function parameter
            rCB, # central bank interest rate
            cB, # banks costs
            ):
        self.simulationTime = simulationTime
        self.numberOfFirms = numberOfFirms
        self.numberOfBanks = numberOfBanks
        self.alpha = alpha
        self.varpf = varpf
        self.gamma = gamma
        self.chi = chi # number of potential partners on credit market
        self.lambd = lambd # intensity of choice
        self.adj = adj # leverage adjustment
        self.phi = phi # production function parameter
        self.beta = beta # production function parameter
        self.rCB = rCB # central bank interest rate
        self.cB = cB # banks costs

        # output variables: Each col for monte carlo run, each row for sim round
        self.changeFB = np.array([]) # switching rate report
        self.YF = np.array([]) # output report
        self.AB = np.array([]) # banks NW report
        self.AF = np.array([]) # firms NW report
        self.BF = np.array([]) # debt report
        self.RBF = np.array([]) # average interest rate report
        self.BAD = np.array([]) # bad debt report
        self.FALLF = np.array([]) # firms defaults report
        self.FALLB = np.array([]) # banks defaults report
        self.LEV = np.array([]) # leverage (BF/AF) report
        self.PRF = np.array([]) # firms profits report
        self.PRB = np.array([]) # banks profits report
        self.GR = np.array([]) # growth rate report
        self.PR = np.array([]) # average price
        self.PBF = np.array([]) # firms probability of default report
        self.PBB = np.array([]) # banks probability of default report

    def run(self):
        s = simulation.Simulation(simulationTime, self.numberOfFirms,
                self.numberOfBanks, self.alpha, self.varpf, self.gamma,
                self.chi, self.lambd, self.adj, self.phi, self.beta,
                self.rCB, self.cB)
        s.run()

if __name__=="__main__":
    
    simulationTime = 1000
    numberOfFirms = 500
    numberOfBanks = 50

    alpha = 0.1 # mean of firm price
    varpf = 0.4 # variance of firm price
    gamma = 0.02 # interest rate parameter
    chi = 5 # number of potential partners on credit market
    lambd = 4 # intensity of choice
    adj = 0.1 # leverage adjustment
    phi = 3 # production function parameter
    beta = 0.7 # production function parameter
    rCB = 0.02 # central bank interest rate
    cB = 0.01 # banks costs

    monteCarlo = MonteCarlo(simulationTime,
                            numberOfFirms,
                            numberOfBanks,
                            alpha,
                            varpf,
                            gamma,
                            chi,
                            lambd,
                            adj,
                            phi,
                            beta,
                            rCB,
                            cB)
    monteCarlo.run()
