import numpy as np

class Firms:

    def __init__(self,
                numberOfFirms,
                alpha, # mean of firms price
                varpf # variance of firms price
                ):
        self.numberOfFirms = numberOfFirms
        self.price = np.random.normal(size=numberOfFirms)*alpha + varpf
        self.debt = np.zeros(numberOfFirms)
        self.networth = np.full_like(np.arange(numberOfFirms), 10, dtype=float)
        self.profit = np.zeros(numberOfFirms, dtype=float)
        self.interestRate = np.zeros(numberOfFirms) # firm interestRate on loans
        self.leverage = np.ones(numberOfFirms)
        self.totalCapital = np.zeros(numberOfFirms)
        self.output = np.zeros(numberOfFirms)
        self.lgdf = np.zeros(numberOfFirms) # loss-given default ratio
        self.default = np.zeros(numberOfFirms) # (True=defaulted, False=surviving)

    def isDefaulted(self, i):
        return np.bool(self.default[i])

class Banks:

    def __init__(self,
                numberOfBanks):
        self.numberOfBanks = numberOfBanks
        self.interestRate = np.zeros(numberOfBanks)
        self.networth = np.full_like(np.arange(numberOfBanks), 10, dtype=float)
        self.deposit = np.zeros(numberOfBanks)
        self.badDebt = np.zeros(numberOfBanks)
        self.profit = np.zeros(numberOfBanks)
        self.creditLinkDegree = np.zeros(numberOfBanks)
        self.nonPerformingLoans = np.zeros(numberOfBanks)
        self.default = np.zeros(numberOfBanks)

    def isDefaulted(self):
        return np.bool(self.default[i])
