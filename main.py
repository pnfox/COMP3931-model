import numpy as np


class MonteCarlo:

    # setup monte carlo variables
    def __init__(self,
            simulationTime,
            numberOfFirms,
            numberOfBanks,
            alpha,
            varpf,
            gamma):
        chi = 5 # number of potential partners on credit market
        lambd = 4 # intensity of choice
        adj = 0.1 # leverage adjustment
        phi = 3 # production function parameter
        beta = 0.7 # production function parameter
        rBC = 0.02 # central bank interest rate
        cB = 0.01 # banks costs

        # output variables: Each col for monte carlo run, each row for sim round
        self.changeFB = np.array() # switching rate report
        self.YF = np.array() # output report
        self.AB = np.array() # banks NW report
        self.AF = np.array() # firms NW report
        self.BF = np.array() # debt report
        self.RBF = np.array() # average interest rate report
        self.BAD = np.array() # bad debt report
        self.FALLF = np.array() # firms defaults report
        self.FALLB = np.array() # banks defaults report
        self.LEV = np.array() # leverage (BF/AF) report
        self.PRF = np.array() # firms profits report
        self.PRB = np.array() # banks profits report
        self.GR = np.array() # growth rate report
        self.PR = np.array() # average price
        self.PBF = np.array() # firms probability of default report
        self.PBB = np.array() # banks probability of default report

if __name__=="__main__":
    
    simulationTime = 1000
    numberOfFirms = 500
    numberOfBanks = 50

    alpha = 0.1 # mean of firm price
    varpf = 0.4 # variance of firm price
    gamma = 0.02 # interest rate parameter

    monteCarlo = MonteCarlo(simulationTime,
                            numberOfFirms,
                            numberOfBanks,
                            alpha,
                            varpf,
                            gamma)
