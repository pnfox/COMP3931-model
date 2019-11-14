import numpy as np
#import matplotlib.pyplot as plt
import math
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
        self.rCB = rCB
        self.cB = cB
        self.bestFirm = 0

        self.firms = agents.Firms(numberOfFirms, self.alpha, self.varpf)

        self.banks = agents.Banks(numberOfBanks)


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
        self.changeFB = np.array([0]*self.time, dtype=float)
        self.firmOutputReport = np.array([0]*self.time, dtype=float)
        self.firmCapitalReport = np.array([0]*self.time, dtype=float)

    def findBestBank(self, potentialPartners):
        bestInterest = np.amin(self.banks.interestRate)
        if np.isnan(bestInterest):
            print("Error: no bank with lowest interest")
            print(self.banks.interestRate)
            exit()
        best = np.where(self.banks.interestRate == np.amin(self.banks.interestRate))[0]
        if len(best) > 1:
            best = best[np.random.randint(0, len(best))]
        elif len(best) == 0:
            print("Error: non bank with best interest rate ", bestInterest)
            exit()

        return best

    # Find bank-firm links that form credit network
    def findMatchings(self, time):
        for f in range(self.numberOfFirms):
            # select potential partners, this is newFallBack
            potentialPartners = np.ceil(np.random.uniform(size=self.chi)*self.numberOfBanks)

            # select best bank
            bestBankIndex = self.findBestBank(potentialPartners)
            newInterest = self.banks.interestRate[bestBankIndex]

            # pick up interest of old partner
            oldInterest = self.banks.interestRate[self.link_fb[f]-1]

            #compare old and new
            if (np.random.uniform(size=1) < \
                    (1-math.exp(self.lambd*(newInterest - oldInterest) / newInterest))):
                #switch
                self.changeFB[time] = self.changeFB[time] + 1

                # TODO: check for multiple best banks

                # update link
                self.link_fb[f] = bestBankIndex
            else:
                self.link_fb[f] = self.link_fb[f]

        self.changeFB[time] = self.changeFB[time] / self.numberOfFirms

    # find who is using bank i
    def findBankCustomers(self, i):
        return np.where(self.link_fb == i)[0]

    def calculateDeposits(self):
        bankIndex = 0
        for bank in range(self.numberOfBanks):
            # find who is using bank
            self.banks.deposit[bank] = np.sum(self.firms.debt[self.link_fb == bank] - self.banks.networth[bank])

            # bank has gone bankrupt
            if self.banks.deposit[bank] < 0:
                self.banks.deposit[bank] = 0

            # compute bad debt
            defaultedFirmsWithBank = np.where(self.firms.default == 1) and np.where(self.link_fb == bank)
            self.banks.badDebt[bank] = np.sum(self.firms.lgdf[defaultedFirmsWithBank] * \
                                        self.firms.debt[defaultedFirmsWithBank])

            # compute bank profits
            nonDefaultedFirmsWithBank = np.where(self.firms.default == 0) and np.where(self.link_fb == bank)
            p = np.dot(self.firms.debt[nonDefaultedFirmsWithBank], self.firms.interestRate[nonDefaultedFirmsWithBank]) - \
                                self.rCB * self.banks.deposit[bank] - self.cB * \
                                self.banks.networth[bank] - self.banks.badDebt[bank]
            self.banks.profit[bank] = p
            bankIndex += 1

    def maxFirmWealth(self):
        return np.amax(self.firms.networth) 

    # replace bankrupt banks and firms with new ones
    def replaceDefaults(self):
        maxFirmWealth = self.maxFirmWealth()
        defaulted = np.where(self.firms.default == 1)
        self.firms.networth[defaulted] = 2 * np.random.uniform(size=len(defaulted))
        self.firms.leverage[defaulted] = 1
        self.firms.price[defaulted] = np.random.normal(self.alpha, self.varpf, size=len(defaulted))
        self.link_fb[defaulted] = np.ceil(np.random.uniform(size=len(defaulted))*self.numberOfBanks)
        self.firms.interestRate[defaulted] = self.rCB + self.banks.interestRate[ \
                self.link_fb[defaulted]-1] + self.gamma * \
                (self.firms.leverage[defaulted] / ((1+self.firms.networth[defaulted] / maxFirmWealth)))
        self.firms.default[defaulted] = 0

        defaulted = np.where(self.banks.default == 1)
        self.banks.networth[defaulted] = 2 * np.random.uniform(size=len(defaulted))
        self.banks.default[defaulted] = 0
        for i in self.banks.networth:
            if i < 0:
                print("Error: Banks with negative net worth")
            if np.isnan(i):
                print("Error: Banks with NAN networth exist")
                print("Defaulted Banks: ", defaulted)
                print("Networth: ", i)
                exit()

    def updateInterestRates(self):
        self.banks.interestRate = self.gamma * np.float_power(self.banks.networth, -self.gamma)

    def updateFrimDebt(self):
        self.firms.debt = self.firms.leverage * self.firms.networth

    def updateFirmCapital(self):
        self.firms.totalCapital = self.firms.networth + self.firms.debt

    def updateFirmOutput(self):
        self.firms.output = self.phi * np.float_power(self.firms.totalCapital, self.beta)

    def updateFirmPrice(self):
        self.firms.price = np.random.normal(self.alpha, self.varpf, size=self.numberOfFirms)

    def updateFirmInterestRate(self):
        for f in range(self.numberOfFirms):
            # interest of bank that firm uses
            bankInterest = self.banks.interestRate[self.link_fb[f]-1]
            bestFirmWorth = self.maxFirmWealth()
            self.firms.interestRate[f] = self.rCB + bankInterest + \
                             self.gamma*(self.firms.leverage[f]) / \
                             ((1+self.firms.networth[f]/bestFirmWorth))
        
    def updateFirmProfit(self):
        self.firms.profit = self.firms.price * self.firms.output - self.firms.interestRate * self.firms.debt

    def updateFirmNetWorth(self):
        self.firms.networth += self.firms.profit
        # check if bankrupt
        self.firms.default[self.firms.networth > 0] = 0
        self.firms.default[self.firms.networth <= 0] = 1

    def updateBankNetWorth(self):
        self.banks.networth += self.banks.profit
        # check if bankrupt
        self.banks.default[self.banks.networth > 0] = 0
        self.banks.default[self.banks.networth <= 0] = 1
        
    def updateFirmLeverage(self):
        u = np.random.uniform(size=self.numberOfFirms)
        firmsPriceGreaterInterest = self.firms.price > self.firms.interestRate
        firmsPriceLessInterest = self.firms.price <= self.firms.interestRate

        self.firms.leverage[firmsPriceGreaterInterest] = \
                self.firms.leverage[firmsPriceGreaterInterest] * \
                (1+self.adj * u[firmsPriceGreaterInterest])

        self.firms.leverage[firmsPriceLessInterest] = \
                self.firms.leverage[firmsPriceLessInterest] * \
                (1-self.adj * u[firmsPriceLessInterest])

    def updateFirmDebt(self):
        self.firms.debt = self.firms.leverage * self.firms.networth

    def updateLossRatio(self):
        self.firms.lgdf = -self.firms.networth / self.firms.debt
        self.firms.lgdf[self.firms.lgdf > 1] = 1
        self.firms.lgdf[self.firms.lgdf < 0] = 0

    def run(self):
        for t in range(self.time):
            # replace defaulted firms and banks
            self.replaceDefaults()

            if np.any(self.firms.networth <= 0):
                print("Firm negative networth", np.where(self.firms.networth <= 0))
            #    exit()
        #    if np.any(self.firms.price <= 0):
        #        print("Firm price negative", np.where(self.firms.price <= 0))
        #        print(self.firms.price[np.where(self.firms.price <= 0)[0]])
        #        print(self.firms.default[np.where(self.firms.price <= 0)[0]])
        #        exit()
            if np.any(self.banks.networth <= 0):
                print("Banks negative networth at ", np.where(self.banks.networth <= 0))
            #    exit()
            if np.any(self.firms.totalCapital < 0):
                print("Firms with negative total capital", len(np.where(self.firms.totalCapital <= 0)[0]))
                print("Print at time: ", t)
                exit()

            # update banks interest rates
            self.updateInterestRates()

            # find bank-firm matchings
            self.findMatchings(t)

            # firms update leverage target
            self.updateFirmLeverage()

            # determine demand for loans
            self.updateFirmDebt()

            # compute total financial capital
            self.updateFirmCapital()

            # compute output
            self.updateFirmOutput()

            # update price
            self.updateFirmPrice()

            # find best firm

            # compute interest rate charged to firms
            self.updateFirmInterestRate()

            # compute firms profit
            self.updateFirmProfit()

            # update firms net worth and check wether defaulted
            self.updateFirmNetWorth()

            # compute loss given default ratio
            self.updateLossRatio()

            # compute deposits
            self.calculateDeposits()

            # update banks net worth and check if defaulted
            self.updateBankNetWorth()

            totalCapital = np.sum(self.firms.totalCapital)
            totalOutput = np.sum(self.firms.output)
            self.firmOutputReport[t] = totalOutput
            self.firmCapitalReport[t] = totalCapital
#        plt.plot(self.firmCapitalReport)
#        plt.show()
