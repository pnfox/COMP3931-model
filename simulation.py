import numpy as np

class Simulation:

    def __init__(self,
            numberOfFirms,
            numberOfBanks,
            alpha,
            varpf):
        self.numberOfFirms = numberOfFirms
        self.numberOfBanks = numberOfBanks

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
        link_fb[0:numberOfFirms] = np.ceil(np.random.uniform(size=numberOfFirms)*numberOfBanks)

    # Find bank-firm links that form credit network
    def findMatchings(self, time):
        for f in firms:
            # select potential partners
            newFallBack = np.ceil(np.random.uniform(size=chi)*self.numberOfBanks)

            # select best bank
            newBank = min(Rb(newFallBack))

            # pick up interest of old partner
            oldBank = Rb(link_fb[f])

            #compare old and new
            if (np.random.uniform(size=1) < (1-exp(lambd*(newBank - oldBank) / newBank))):
                #switch
                changeFb[t] = changeFb[t] + 1
                research = which(Rb[newFallBack] == min(Rb[newFallBack]))

                #check if multiple best interests
                if (len(research) > 1):
                    research = research[np.ceil(np.random.uniform(size=1)*length(research))]

                link_fb[f] = newFallBack[research[1]]
            else:
                link_fb[f]=link_fb[f]

        changeFB[t]=changeFB[t]/numberOfFirms

    def run(self, time):
        for t in range(time):
            # replace defaulted firms and banks

            # update banks interest rates
            Rb = gamma * Ab **(-gamma)

            # find bank-firm matchings
            self.findMatchings(t)

            # firms update leverage target
            u = np.random.uniform(size=numberOfFirms)
            lev[pf > Rbf] = lev[pf > Rbf] * (1 + adj*u[pj>Rbf])
            lev[pf <= Rbf] = lev[pf <= Rbf] * (1 - adj*u[pj<=Rbf])

            # determine demand for loans
            Bf = lev * Af

            # compute total financial capital
            Kf = Af + Bf

            # compute output
            Yf = phi * Kf ** beta

            # update price
            pf = np.random.normal(alpha, varpf, numberOfFirms)

            # compute interest rate charged to firms
            Rbf = rCB + Rb(link_fb) + gamma*(lev) / ((1+Af/max(Af)))

            # compute firms price
            Prf = pf * Yf - Rbf * Bf

            # update firms net worth and check wether defaulted
            Af = Af + Prf
            fallf[Af > 0] = 0
            fallf[Af <= 0] = 1

            # compute loss given default ratio
            LGDf[0:numberOfFirms] = -(Af) / (Bf)
            LGDf[LGDf > 1] = 1
            LGDf[LGDf < 0] = 0

            # compute deposits

            # update banks net worth and check if defaulted
