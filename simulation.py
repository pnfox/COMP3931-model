import numpy as np
import matplotlib.pyplot as plt
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


        self.link_fb = np.array([0]*self.numberOfFirms) # firms-banks credit matching

        self.link_fb[0:numberOfFirms] = np.ceil(np.random.uniform( \
                                        size=self.numberOfFirms)*self.numberOfBanks)

        # Output variables
        self.changeFB = np.array([0]*self.time)
        self.firmOutputReport = np.array([0]*self.time)

    def findBestBank(self, potentialPartners):
        lowestInterest = 100000
        for i in potentialPartners:
            index = int(i)-1
            if self.banks[index].interestRate < lowestInterest:
                lowestInterest = self.banks[index].interestRate
                best = index

        return best

    # Find bank-firm links that form credit network
    def findMatchings(self, time):
        for f in range(len(self.firms)):
            # select potential partners, this is newFallBack
            potentialPartners = np.ceil(np.random.uniform(size=self.chi)*self.numberOfBanks)

            # select best bank
            bestBankIndex = self.findBestBank(potentialPartners)
            newInterest = self.banks[bestBankIndex].interestRate

            # pick up interest of old partner
            oldInterest = self.banks[self.link_fb[f]-1].interestRate

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
        for bank in self.banks:
            # find who is using bank
            customers = self.findBankCustomers(bankIndex)
            banksTotalLoans = 0
            for i in customers:
                banksTotalLoans += self.firms[i].debt

            bank.deposit = banksTotalLoans - bank.networth

            # bank has gone bankrupt
            if bank.deposit < 0:
                bank.deposit = 0

            # compute bad debt
            bankBadDebt = 0
            for i in customers:
                if self.firms[i].default:
                    bankBadDebt += self.firms[i].lgdf * self.firms[i].debt

            bank.badDebt = bankBadDebt

            # compute bank profits
            bankProfit = 0
            for i in customers:
                customer = self.firms[i]
                if customer.default:
                    bankProfit += customer.debt * customer.interestRate - \
                                self.rCB * bank.deposit - \
                                self.cB * bank.networth - bank.badDebt
            bank.profit = bankProfit
            bankIndex += 1

    def maxFirmWealth(self):
        maxWealth = -100000
        for f in self.firms:
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
                maxFirmWealth = self.maxFirmWealth()
                firm.interestRate = self.rCB + self.banks[self.link_fb[fnum]-1].interestRate + \
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
            f.output = self.phi * f.totalCapital ** self.beta

    def updateFirmPrice(self):
        for f in self.firms:
            f.price = np.random.normal(self.alpha, self.varpf)

    def updateFirmInterestRate(self):
        firmIndex = 0
        for f in self.firms:
            # interest of bank that firm uses
            bankInterest = self.banks[self.link_fb[firmIndex]-1].interestRate
            bestFirmWorth = self.bestFirm.networth
            f.interestRate = self.rCB + bankInterest + \
                             self.gamma*(f.leverage) / ((1+f.networth/bestFirmWorth))
            firmIndex += 1

    def updateFirmProfit(self):
        for f in self.firms:
            f.profit = f.price * f.output - f.interestRate * f.debt

    def updateFirmNetWorth(self):
        for f in self.firms:
            f.networth = f.networth + f.profit
            if f.networth > 0:
                f.default = 0
            elif f.networth <= 0:
                f.default = 1

    def updateBankNetWorth(self):
        for b in self.banks:
            b.networth += b.profit
            if b.networth > 0:
                b.default = 0
            elif b.networth <= 0:
                b.default = 1

    def updateFirmLeverage(self):
        for f in self.firms:
            u = np.random.uniform()
            if f.price > f.interestRate:
                f.leverage = f.leverage * (1 + self.adj * u)
            elif f.price <= f.interestRate:
                f.leverage = f.leverage * (1 - self.adj * u)

    def updateFirmDebt(self):
        for f in self.firms:
            f.debt = f.leverage * f.networth

    def updateLossRatio(self):
        for f in self.firms:
            lossGivenRatio = -f.networth / f.debt
            if lossGivenRatio > 1:
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
            bestNetWorthFirm = -100000
            for f in self.firms:
                if f.networth > bestNetWorthFirm:
                    self.bestFirm = f
                    bestNetWorthFirm = f.networth

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

            totalOutput = 0
            for f in self.firms:
                totalOutput += f.output

            self.firmOutputReport[t] = totalOutput
        plt.plot(self.firmOutputReport)
        plt.show()
