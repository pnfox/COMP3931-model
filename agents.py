import numpy as np

class Firms:

    def __init__(self,
                numberOfFirms,
                alpha, # mean of firms price
                varpf # variance of firms price
                ):
        self.price = np.random.normal(alpha, varpf, numberOfFirms)
        self.debt = np.zeros(numberOfFirms)
        self.networth = np.full_like(np.arange(numberOfFirms), 10)
        self.profit = np.zeros(numberOfFirms)
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
        self.interestRate = np.zeros(numberOfFirms)
        self.networth = np.full_like(np.arange(numberOfFirms), 10)
        self.deposit = np.zeros(numberOfFirms)
        self.badDebt = np.zeros(numberOfFirms)
        self.profit = np.zeros(numberOfFirms)
        self.creditLinkDegree = np.zeros(numberOfFirms)
        self.nonPerformingLoans = np.zeros(numberOfFirms)
        self.default = np.zeros(numberOfFirms)

    def isDefaulted(self):
        return np.bool(self.default[i])
