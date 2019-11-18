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

        # firms-banks credit matching adjacency matrix
        self.link_fb = np.zeros((numberOfFirms, numberOfBanks))
        banksWithFirms = np.ceil(np.random.uniform(0, self.numberOfBanks-1, self.numberOfFirms))
        for i in range(self.numberOfFirms):
            self.link_fb[i][int(banksWithFirms[i])] = 1


        # contains banks that firms use as lookup for interestRates
        self.bankPools = np.zeros((self.numberOfFirms, self.chi))

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
        self.bankPools = np.ceil(np.random.uniform(0, self.numberOfBanks, \
                            self.chi*self.numberOfFirms).reshape(self.numberOfFirms, self.chi))
        for f in range(self.numberOfFirms):
            # select potential partners, this is newFallBack
            potentialPartners = self.bankPools[f]

            # select best bank
            bestBankIndex = self.findBestBank(potentialPartners)
            newInterest = self.banks.interestRate[bestBankIndex]

            # pick up interest of old partner
            currentBank = np.nonzero(self.link_fb[f])
            if not currentBank[0]:
                oldInterest = np.nan
            else:
                oldInterest = self.banks.interestRate[currentBank[0][0]]

            #compare old and new
            if (newInterest < oldInterest):
                #switch
                self.changeFB[time] = self.changeFB[time] + 1

                # TODO: check for multiple best banks

                # update link
                self.link_fb[f] = 0
                self.link_fb[f][bestBankIndex] = 1

        self.changeFB[time] = self.changeFB[time] / self.numberOfFirms

    # find who is using bank i
    def findBankCustomers(self, i):
        return np.nonzero(self.link_fb.transpose()[i])[0]

    def calculateDeposits(self):
        bankIndex = 0
        for bank in range(self.numberOfBanks):
            # find who is using bank
            bankCustomers = self.findBankCustomers(bank)
            self.banks.deposit[bank] = np.sum(self.firms.debt[bankCustomers] - self.banks.networth[bank])

            # bank has gone bankrupt
            if self.banks.deposit[bank] < 0:
                self.banks.deposit[bank] = 0

            # compute bad debt
            defaultedFirmsWithBank = np.where(self.firms.default == 1) and bankCustomers
            self.banks.badDebt[bank] = np.sum(self.firms.lgdf[defaultedFirmsWithBank] * \
                                        self.firms.debt[defaultedFirmsWithBank])

            # compute bank profits
            nonDefaultedFirmsWithBank = np.where(self.firms.default == 0) and bankCustomers
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
        defaulted = np.where(self.firms.default == 1)[0]
        if defaulted.size == 0:
            return

        # find new partners for defaulted firms
        banks = np.ceil(np.random.uniform(0, self.numberOfBanks-1, len(defaulted)))
        self.link_fb[defaulted] = 0
        j = 0
        for i in defaulted:
            self.link_fb[i][int(banks[j])] = 1
            j += 1

        # update firm variables
        self.firms.networth[defaulted] = np.random.uniform(2, size=len(defaulted))
        self.firms.leverage[defaulted] = 1
        self.firms.price[defaulted] = np.random.normal(self.alpha, np.sqrt(self.varpf), size=len(defaulted))
        self.firms.interestRate[defaulted] = self.rCB + self.banks.interestRate[ \
                np.nonzero(self.link_fb[defaulted])[0][0]] + self.gamma * \
                (self.firms.leverage[defaulted] / ((1+self.firms.networth[defaulted] / maxFirmWealth)))
        self.firms.default[defaulted] = 0

        # replace defaulted banks
        defaulted = np.where(self.banks.default == 1)
        self.banks.networth[defaulted] = np.random.uniform(2, size=len(defaulted))
        self.banks.default[defaulted] = 0

    def updateInterestRates(self):
        self.banks.interestRate = self.gamma * np.float_power(self.banks.networth, -self.gamma)

    def updateFrimDebt(self):
        self.firms.debt = self.firms.leverage * self.firms.networth

    def updateFirmCapital(self):
        self.firms.totalCapital = self.firms.networth + self.firms.debt

    def updateFirmOutput(self):
        self.firms.output = self.phi * np.float_power(self.firms.totalCapital, self.beta)

    def updateFirmPrice(self):
        self.firms.price = np.random.normal(self.alpha, np.sqrt(self.varpf), size=self.numberOfFirms)

    def updateFirmInterestRate(self):
        for f in range(self.numberOfFirms):
            # interest of bank that firm uses
            currentBank = np.nonzero(self.link_fb[f])[0]
            bankInterest = self.banks.interestRate[currentBank[0]]
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
        firmsPriceGreaterInterest = []
        firmsPriceLessInterest = []

        for i in range(self.numberOfFirms):
            if self.firms.price[i] > self.firms.interestRate[i]:
                firmsPriceGreaterInterest.append(i)
            else:
                firmsPriceLessInterest.append(i)

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
        print("Running Simulation...")
        for t in range(self.time):
            # replace defaulted firms and banks
            self.replaceDefaults()

            if np.any(self.firms.networth <= 0):
                print("Firm negative networth", np.where(self.firms.networth <= 0))
                exit()
            if np.any(self.banks.networth <= 0):
                print("Banks negative networth at ", np.where(self.banks.networth <= 0))
                exit()
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
