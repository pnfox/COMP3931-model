import numpy as np

class Firms:

    def __init__(self,
                numberOfFirms=100,
                alpha=0.1, # mean of firms price
                varpf=0.4 # variance of firms price
                ):
        self.numberOfFirms = numberOfFirms
        self.price = np.random.normal(alpha, np.sqrt(varpf), size=numberOfFirms)
        self.debt = np.zeros(numberOfFirms)
        self.networth = np.full_like(np.arange(numberOfFirms), 10, dtype=float)
        self.profit = np.zeros(numberOfFirms, dtype=float)
        self.interestRate = np.zeros(numberOfFirms) # firm interestRate on loans
        self.leverage = np.ones(numberOfFirms)
        self.capital = np.zeros(numberOfFirms)
        self.output = np.zeros(numberOfFirms)
        self.lgdf = np.zeros(numberOfFirms) # loss-given default ratio
        self.default = np.zeros(numberOfFirms) # (True=defaulted, False=surviving)

    def isDefaulted(self, i):
        return np.bool(self.default[i])

class IndividualFirm:

    def __init__(self):
        self.output = 0
        self.capital = 0
        self.price = 0
        self.networth = 0
        self.debt = 0
        self.profit = 0
        self.default = 0
        self.interest = 0

class Banks:

    def __init__(self,
                numberOfBanks=10):
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

class Economy:

    def __init__(self,
                time):
        if time is None:
            raise ValueError("Must provide economy with number of time steps")

        self.time = time
        self.GDP = np.array([0]*self.time, dtype=float)
        self.avgInterest = np.array([0]*self.time, dtype=float)
        self.leverage = np.array([0]*self.time, dtype=float)
