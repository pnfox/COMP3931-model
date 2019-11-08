import numpy as np
import agents

class Simulation:

    def __init__(self,
            numberOfFirms,
            numberOfBanks,
            alpha, # mean of firms price
            varpf, # variance of firms price
            gamma, # interest rate parameter
            ):
        self.numberOfFirms = numberOfFirms
        self.numberOfBanks = numberOfBanks
        self.alpha = alpha
        self.varpf = varpf
        self.gamma = gamma

        # store firms in array
        self.firms = np.array([])
        for i in range(numberOfFirms):
            f = Firm(alpha, varpf)
            self.firms = np.append(self.firms, f)

        # store banks in array
        self.banks = np.array([])
        for i in range(numberOfBanks):
            b = Bank()
            self.banks = np.append(self.banks, b)


        Rb = np.array([]) # banks interest rate
        Ab = np.array([]) # banks net wealth
        self.link_fb = np.array([0]*numberOfFirms) # firms-banks credit matching
        Rbf = np.array([]) # firms interest rate on loans
        lev = np.array([]) # firms leverage
        pf = np.array([]) # firms price
        Bf = np.array([]) # firms net debt
        Af = np.array([]) # firms net wealth
        fallf = np.array([]) # firms defaults (1=defaulted, 0=surviving)
        fallb = np.array([]) # banks defaults (1=defaulted, 0=surviving)
        LGDF = np.array([]) # loss-given-default ratio
        D = np.array([]) # deposits
        Badb = np.array([]) # banks non performing loans
        Prb = np.array([]) # banks profits
        creditDegree = np.array([]) # banks credit link degree

        self.link_fb[0:numberOfFirms] = np.ceil(np.random.uniform(size=numberOfFirms)*numberOfBanks)

    def findBestBank(self, potentialPartners):
        lowestInterest = 100000
        for i in potentialPartners:
            if self.banks[i].interestRate < lowestInterest:
                lowestInterest = self.banks[i].interestRate
                best = i

        return best

    # Find bank-firm links that form credit network
    def findMatchings(self, time):
        for f in len(self.firms):
            # select potential partners, this is newFallBack
            potentialPartners = np.ceil(np.random.uniform(size=chi)*self.numberOfBanks)

            # select best bank
            bestBankIndex = self.findBestBank(potentialPartners)
            newInterest = self.bank[bestBankIndex].interestRate

            # pick up interest of old partner
            oldInterest = self.bank[self.link_fb[f]].interestRate

            #compare old and new
            if (np.random.uniform(size=1) < \
                    (1-exp(lambd*(newInterest - oldInterest) / newInterest))):
                #switch
                changeFb[t] = changeFb[t] + 1

                # TODO: check for multiple best banks

                self.link_fb[f] = bestBankIndex
            else:
                self.link_fb[f]=self.link_fb[f]

        changeFB[t]=changeFB[t]/numberOfFirms

    def calculateDeposits(self):
        for bank in range(numberOfBanks):
            D[bank] = np.sum(Bf[self.link_fb==bank])-Ab[bank]
            if D[bank] < 0:
                D[bank] = 0
            # compute bad debt
            Badb[bank] = np.sum(LGDf[fallf==1 and self.link_fb==bank] *
                                Bf[fallf==1 and self.link_fb==bank])
            # compute bank profits
            Pjb[bank] = Bank[self.link_fb==bank and fallf==0] * \
                        Rbf[self.link_fb==bank and fallf==0] - \
                        rCB * D[bank] - cB * Ab[bank]-Badb[bank]

    def maxFirmWealth(self):
        maxWealth = -100000
        for f in self.firms():
            if f.networth > maxWealth:
                maxWealth = f.networth

        return maxWealth

    # replace bankrupt banks and firms with new ones
    def replaceDefaults(self):
        fnum = 0
        for firm in self.firms:
            if firm.default == 1:
                firm.networth = 2 * np.random.uniform()
                firm.leverage = 1
                firm.price = np.random.normal(self.alpha, self.varpf, 1)
                self.link_fb[fnum] = np.ceil(np.random.uniform(size=1)*self.numberOfBanks)
                maxFirmWealth = self.maxFrimWealth()
                firm.interestRate = rCB + self.banks[self.link_fb[fnum]].interestRate + \
                                    self.gamma*(firm.leverage) / \
                                    ((1+firm.networth / maxFirmWealth))
            fnum += 1

        for bank in self.banks:
            if(bank.default == 1):
                bank.networth = 2 * np.random.uniform()

    def run(self, time):
        for t in range(time):
            # replace defaulted firms and banks
            self.replaceDefaults()

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
            Rbf = rCB + Rb(self.link_fb) + gamma*(lev) / ((1+Af/max(Af)))

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
            self.calculateDeposits()

            # update banks net worth and check if defaulted
            Ab=Ab+Prb
            fallb[Ab>0] = 0
            fallb[Ab<=0] = 1
