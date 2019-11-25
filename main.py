import numpy as np
import simulation


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

    model = simulation.Simulation(simulationTime,
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
    model.run()
