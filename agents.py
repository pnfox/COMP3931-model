import numpy as np

class Firms:

    def __init__(self,
                numberOfFirms,
                alpha, # mean of firms price
                varpf # variance of firms price
                ):
        self.numberOfFirms = numberOfFirms
        self.price = (np.random.normal(size=numberOfFirms)*alpha + varpf).astype(np.float64) 
        self.debt = np.zeros(numberOfFirms).astype(np.float64)
        self.networth = (np.full_like(np.arange(numberOfFirms), 10)).astype(np.float)
        self.profit = np.zeros(numberOfFirms, dtype=np.float64)
        self.interestRate = np.zeros(numberOfFirms).astype(np.float64) # firm interestRate on loans
        self.leverage = np.ones(numberOfFirms).astype(np.float64)
        self.totalCapital = np.zeros(numberOfFirms).astype(np.float64)
        self.output = np.zeros(numberOfFirms).astype(np.float64)
        self.lgdf = np.zeros(numberOfFirms).astype(np.float64) # loss-given default ratio
        self.default = np.zeros(numberOfFirms).astype(np.float64) # (True=defaulted, False=surviving)

    def isDefaulted(self, i):
        return np.bool(self.default[i])

class Banks:

    def __init__(self,
                numberOfBanks):
        self.numberOfBanks = numberOfBanks
        self.interestRate = np.zeros(numberOfBanks).astype(np.float64) 
        self.networth = (np.full_like(np.arange(numberOfBanks), 10)).astype(np.float64)
        self.deposit = np.zeros(numberOfBanks).astype(np.float64) 
        self.badDebt = np.zeros(numberOfBanks).astype(np.float64) 
        self.profit = np.zeros(numberOfBanks).astype(np.float64) 
        self.creditLinkDegree = np.zeros(numberOfBanks).astype(np.float64) 
        self.nonPerformingLoans = np.zeros(numberOfBanks).astype(np.float64) 
        self.default = np.zeros(numberOfBanks).astype(np.float64) 

    def isDefaulted(self):
        return np.bool(self.default[i])
