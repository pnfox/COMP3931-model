
class Firm:

    def __init__(self):
        self.price = 0
        self.debt = 0
        self.networth = 0
        self.leverage = 0
        self.default = False # (True=defaulted, False=surviving)

    def isDefaulted(self):
        return self.default

class Bank:

    def __init__(self):
        self.interestRate = 0
        self.networth = 0
        self.deposit = 0
        self.profit = 0
        self.creditLinkDegree = 0
        self.nonPerformingLoans = 0
        self.default = False

    def isDefaulted(self):
        return self.default
