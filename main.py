


class MonteCarlo:

    # setup monte carlo variables
    def __init__(self):
        numberOfSimulations = 1000
        numberOfFirms = 500
        numberOfBanks = 50
        gamma = 0.02 # interest rate parameter
        chi = 5 # number of potential partners on credit market
        lambd = 4 # intensity of choice
        adj = 0.1 # leverage adjustment
        phi = 3 # production function parameter
        beta = 0.7 # production function parameter
        alpha = 0.1 # mean of firm price
        varpf = 0.4 # variance of firm price
        rBC = 0.02 # central bank interest rate
        cB = 0.01 # banks costs

        # output variables: Each entry in arrays is for 1 simulation
        changeFB = np.array() # switching rate report
        YF = np.array() # output report
        AB = np.array() # banks NW report
        AF = np.array() # firms NW report
        BF = np.array() # debt report
        RBF = np.array() # average interest rate report
        BAD = np.array() # bad debt report
        FALLF = np.array() # firms defaults report
        FALLB = np.array() # banks defaults report
        LEV = np.array() # leverage (BF/AF) report
        PRF = np.array() # firms profits report
        PRB = np.array() # banks profits report
        GR = np.array() # growth rate report
        PR = np.array() # average price
        PBF = np.array() # firms probability of default report
        PBB = np.array() # banks probability of default report
