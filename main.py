import numpy as np
import simulation
import sys
import argparse

parser = argparse.ArgumentParser(description="Economic ABM model simulation")
parser.add_argument("-t", "--timesteps", action='store', dest='simulationSteps', \
        required=True, help="Steps simulation is ran for")
parser.add_argument("-f", "--firms", action='store', dest='firms',
        required=True, help="Number of Firms in Simulation")
parser.add_argument("-b", "--banks", action='store', dest='banks',
        required=True, help="Number of Banks in Simulation")
parser.add_argument("-alpha", action='store', dest='priceMean', required=False, \
        default=0.1, help="firm price mean")
parser.add_argument("-var", action='store', dest='priceVariance', required=False, \
        default=0.4, help="firm price variance")
parser.add_argument("-gamma", action='store', dest='gamma', required=False, \
        default=0.02, help="interest rate parameter")
parser.add_argument("-partners", action='store', dest='chi', required=False, \
        default=5, help="number of potential partners on credit market")
parser.add_argument("-intensity", action='store', dest='choiceIntensity', required=False, \
        default=4, help="Intensity of partner choice")
parser.add_argument("-adjustment", action='store', dest='adjustment', required=False, \
        default=0.1, help="Leverage adjustment")
parser.add_argument("-phi", action='store', dest='phi', required=False, \
        default=3, help="Firm production function parameter")
parser.add_argument("-beta", action='store', dest='beta', required=False, \
        default=0.7, help="Firm production function parameter")
parser.add_argument("-rCB", action='store', dest='rCB', required=False, \
        default=0.02, help="Central bank interest rate")
parser.add_argument("-bankcost", action='store', dest='bankCost', required=False, \
        default=0.01, help="Cost of maintaining banks")
parser.add_argument("-i", "--interactive", required=False, default=False, \
        action='store_true', dest='interactive', \
        help="interactive mode - Useful for debugging")
parser.add_argument("-s", "--seed", action='store', dest='seed', required=False, \
        help="Simulation seed")
parser.add_argument("-o", "--output", action='store', dest='folder', required=False, \
        default=None, help="Directory location where results are written")


if __name__=="__main__":

    # Get command line arguments
    args = parser.parse_args()

    simulationTime = args.simulationSteps 
    numberOfFirms = args.firms
    numberOfBanks = args.banks
    alpha = args.priceMean # mean of firm price
    varpf = args.priceVariance # variance of firm price
    gamma = args.gamma # interest rate parameter
    chi = args.chi # number of potential partners on credit market
    lambd = args.choiceIntensity # intensity of choice
    adj = args.adjustment # leverage adjustment
    phi = args.phi # production function parameter
    beta = args.beta # production function parameter
    rCB = args.rCB # central bank interest rate
    cB = args.bankCost # banks costs
    output = args.folder

    if(args.interactive):
        mode = "Interactive"
    else:
        mode = "Default"

    if(args.seed):
        seed = args.seed
    else:
        seed = None

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
                            cB,
                            mode=mode,
                            seed=seed,
                            outputFolder=output)
    model.run()
    sys.exit() # terminate program
