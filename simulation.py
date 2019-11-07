import numpy as np

class Simulation:

    def __init__(self):
        Rb = np.array() # banks interest rate
        Ab = np.array() # banks net wealth
        link_fb = np.array() # firms-banks credit matching
        Rbf = np.array() # firms interest rate on loans
        lev = np.array() # firms leverage
        pf = np.array() # firms price
        Bf = np.array() # firms net debt
        Af = np.array() # firms net wealth
        fallf = np.array() # firms defaults (1=defaulted, 0=surviving)
        fallb = np.array() # banks defaults (1=defaulted, 0=surviving)
        LGDF = np.array() # loss-given-default ratio
        D = np.array() # deposits
        Badb = np.array() # banks non performing loans
        Prb = np.array() # banks profits
        creditDegree = np.array() # banks credit link degree

        Af[0:numberOfFirms] = 10
        Ab[0:numberOfBanks] = 10
        pf[0:numberOfFirms] = np.random.normal(alpha, varpf, numberOfFirms)
        lev[0:numberOfFirms] = 1
        link_fb[0:numberOfFirms] = ceiling(runif(numberOfFirms)*numberOfBanks)

    def run(self, time):
        for i in range(time):
            # update banks interest rates

            # find bank-firm matchings

            # firms update leverage target

            # determine demand for loans

            # compute total financial capital

            # compute output

            # update price
            pf = np.random.normal(alpha, varpf, numberOfFirms)

            # compute interest rate charged to firms

            # compute firms price

            # update firms net worth and check wether defaulted

            # compute loss given default ratio

            # compute deposits

            # update banks net worth and check if defaulted
