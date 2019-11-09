
class Firm:

    def __init__(self,
                alpha, # mean of firms price
                varpf # variance of firms price
                ):
        self.price = np.random.normal(alpha, varpf, numberOfFirms)
        self.debt = 0
        self.networth = 10
        self.profit = 0
        self.interestRate = 0 # firm interestRate on loans
        self.leverage = 1
        self.totalCapital = 0
        self.output = 0
        self.lgdf = 0 # loss-given default ratio
        self.default = False # (True=defaulted, False=surviving)

    def isDefaulted(self):
        return self.default

class Bank:

    def __init__(self):
        self.interestRate = 0
        self.networth = 10
        self.deposit = 0
        self.badDebt = 0
        self.profit = 0
        self.creditLinkDegree = 0
        self.nonPerformingLoans = 0
        self.default = False

    def isDefaulted(self):
        return self.default
