import numpy as np
import agents

class Simulation:

    def __init__(self,
            time, # time simulation is ran for
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
        self.time = time
        self.numberOfFirms = numberOfFirms
        self.numberOfBanks = numberOfBanks
        self.alpha = alpha
        self.varpf = varpf
        self.gamma = gamma
        self.chi = chi
        self.lambd = lambd
        self.adj = adj
        self.phi = phi
        self.beta = beta
        self.rBC = rCB
        self.cB = cB

        # store firms in array
        self.firms = np.array([])
        for i in range(self.numberOfFirms):
            f = agents.Firm(self.alpha, self.varpf)
            self.firms = np.append(self.firms, f)

        # store banks in array
        self.banks = np.array([])
        for i in range(self.numberOfBanks):
            b = agents.Bank()
            self.banks = np.append(self.banks, b)


        Rb = np.array([]) # banks interest rate
        Ab = np.array([]) # banks net wealth
        self.link_fb = np.array([0]*self.numberOfFirms) # firms-banks credit matching
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

        self.link_fb[0:numberOfFirms] = np.ceil(np.random.uniform( \
                                        size=self.numberOfFirms)*self.numberOfBanks)

        # Output variables
        self.changeFB = np.array([0]*self.time)

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
            potentialPartners = np.ceil(np.random.uniform(size=self.chi)*self.numberOfBanks)

            # select best bank
            bestBankIndex = self.findBestBank(potentialPartners)
            newInterest = self.bank[bestBankIndex].interestRate

            # pick up interest of old partner
            oldInterest = self.bank[self.link_fb[f]].interestRate

            #compare old and new
            if (np.random.uniform(size=1) < \
                    (1-exp(self.lambd*(newInterest - oldInterest) / newInterest))):
                #switch
                self.changeFb[t] = self.changeFb[t] + 1

                # TODO: check for multiple best banks

                self.link_fb[f] = bestBankIndex
            else:
                self.link_fb[f] = self.link_fb[f]

        self.changeFB[t] = self.changeFB[t] / self.numberOfFirms

    def calculateDeposits(self):
        for bank in range(numberOfBanks):

            banksTotalLoans = 0
            for i in bank.customers:
                bankTotalLoans += self.firms[i].debt

            bank.deposit = banksTotalLoans - bank.networth
            # bank has gone bankrupt
            if bank.deposit < 0:
                bank.deposit = 0

            # compute bad debt
            bankBadDebt = 0
            for i in bank.customers:
                if self.firms[i].default:
                    bankBadDebt += self.firms[i].lgdf * self.firms[i].debt

            bank.badDebt = bankBadDebt

            # compute bank profits
            bankProfit = 0
            for i in bank.customers:
                customer = self.firm[i]
                if customer.default:
                    bankProfit += customer.debt * customer.interestRate - \
                                self.rCB * bank.deposit - \
                                self.cB * bank.networth - bank.badDebt
            bank.profit = bankProfit

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
                firm.interestRate = self.rCB + self.banks[self.link_fb[fnum]].interestRate + \
                                    self.gamma*(firm.leverage) / \
                                    ((1+firm.networth / maxFirmWealth))
            fnum += 1

        for bank in self.banks:
            if(bank.default == 1):
                bank.networth = 2 * np.random.uniform()

    def updateInterestRates(self):
        for b in self.banks:
            b.interestRate = self.gamma * b.networth ** (-self.gamma)

    def updateFrimDebt(self):
        for f in self.firms:
            f.debt = f.leverage * f.networth

    def updateFirmCapital(self):
        for f in self.firms:
            f.totalCapital = f.networth + f.debt 

    def updateFirmOutput(self):
        for f in self.firms:
            f.output = self.phi * self.totalCapital ** self.beta

    def updateFirmPrice(self):
        for f in self.firms:
            f.price = np.random.normal(self.alpha, self.varpf, self.numberOfFirms)

    def updateLossRatio(self):
        for f in self.firms:
            lossGivenRation = -f.networth / f.debt
            if lossGivenRation > 1:
                lossGivenRatio = 1
            if lossGivenRatio < 0:
                lossGivenRatio = 0
            f.lgdf = lossGivenRatio

    def run(self):
        for t in range(self.time):
            # replace defaulted firms and banks
            self.replaceDefaults()

            # update banks interest rates
            self.updateInterestRates()

            # find bank-firm matchings
            self.findMatchings(t)

            # firms update leverage target
            u = np.random.uniform(size=numberOfFirms)
            lev[pf > Rbf] = lev[pf > Rbf] * (1 + adj*u[pj>Rbf])
            lev[pf <= Rbf] = lev[pf <= Rbf] * (1 - adj*u[pj<=Rbf])

            # determine demand for loans
            self.updateFirmDebt()

            # compute total financial capital
            self.updateFirmCapital()

            # compute output
            self.updateFrimOutput()

            # update price
            self.updateFirmPrice()

            # compute interest rate charged to firms
            Rbf = rCB + Rb(self.link_fb) + self.gamma*(lev) / ((1+Af/max(Af)))

            # compute firms price
            Prf = pf * Yf - Rbf * Bf

            # update firms net worth and check wether defaulted
            Af = Af + Prf
            fallf[Af > 0] = 0
            fallf[Af <= 0] = 1

            # compute loss given default ratio
            self.updateLossRatio()

            # compute deposits
            self.calculateDeposits()

            # update banks net worth and check if defaulted
            Ab=Ab+Prb
            fallb[Ab>0] = 0
            fallb[Ab<=0] = 1
