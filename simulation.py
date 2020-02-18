import numpy as np
import os
import agents
import sys

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
            mode=None, # mode to run simulation
            seed=None,
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

        self.mode = mode
        self.continueUntilTime = 0

        if seed == None:
            self.seed = np.random.randint(9000)
        else:
            self.seed = seed

        np.random.seed(self.seed)

        self.firms = agents.Firms(numberOfFirms, self.alpha, self.varpf)

        self.banks = agents.Banks(numberOfBanks)

        # firms-banks credit matching adjacency matrix
        self.link_fb = np.zeros((numberOfFirms, numberOfBanks))
        banksWithFirms = np.ceil(np.random.uniform(0, self.numberOfBanks-1, self.numberOfFirms))
        for i in range(self.numberOfFirms):
            self.link_fb[i][int(banksWithFirms[i])] = 1


        # contains banks that firms use as lookup for interestRates
        self.bankPools = np.zeros((self.numberOfFirms, self.chi))

        self.resultFolder = "results/" + str(self.seed) + "/"

        # Output variables
        self.changeFB = np.array([0]*self.time, dtype=float)
        self.firmOutputReport = np.array([0]*self.time, dtype=float)
        self.firmCapitalReport = np.array([0]*self.time, dtype=float)
        self.firmWealthReport = np.array([0]*self.time, dtype=float)
        self.firmDebtReport = np.array([0]*self.time, dtype=float)
        self.firmProfitReport = np.array([0]*self.time, dtype=float)
        self.firmAvgPrice = np.array([0]*self.time, dtype=float)
        self.firmDefaultReport = np.array([0]*self.time, dtype=float)

        # array to store price, wealth, capital,... from a single firm
        self.individualFirm = np.array([[0,0,0,0,0,0,0,0]], dtype=float)

        self.bankWealthReport = np.array([0]*self.time, dtype=float)
        self.bankDebtReport = np.array([0]*self.time, dtype=float)
        self.bankProfitReport = np.array([0]*self.time, dtype=float)
        self.bankDefaultReport = np.array([0]*self.time, dtype=float)

        self.GDP = np.array([0]*self.time, dtype=float)
        self.avgInterest = np.array([0]*self.time, dtype=float)

    def findBestBank(self, potentialPartners):
        bestInterest = np.inf
        best = np.nan
        for partner in potentialPartners:
            if self.banks.interestRate[int(partner)] < bestInterest:
                bestInterest = self.banks.interestRate[int(partner)]
                best = int(partner)

        return best

    # Find bank-firm links that form credit network
    def findMatchings(self, time):
        self.bankPools = np.ceil(np.random.uniform(0, self.numberOfBanks-1, \
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
                oldInterest = np.inf
            else:
                oldInterest = self.banks.interestRate[currentBank[0][0]]

            #compare old and new
            if (newInterest < oldInterest):
                #switch
                self.changeFB[time] = self.changeFB[time] + 1

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
            self.banks.deposit[bank] = np.sum(self.firms.debt[bankCustomers]) - self.banks.networth[bank]

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

        if np.any(np.where(self.firms.default == 1)):
            raise Exception("Error: Defaulted firms not removed")
        if np.any(self.firms.networth <= 0):
            raise Exception("Firm negative networth", np.where(self.firms.networth <= 0))
        if np.any(self.banks.networth <= 0):
            raise Exception("Banks negative networth at ", np.where(self.banks.networth <= 0))
        if np.any(self.firms.capital < 0):
            raise Exception("Firms with negative total capital", len(np.where(self.firms.capital <= 0)[0]))

    def updateInterestRates(self):
        self.banks.interestRate = self.gamma * np.float_power(self.banks.networth, -self.gamma)

    def updateFrimDebt(self):
        self.firms.debt = self.firms.leverage * self.firms.networth

    def updateFirmCapital(self):
        self.firms.capital = self.firms.networth + self.firms.debt

    def updateFirmOutput(self):
        self.firms.output = self.phi * np.float_power(self.firms.capital, self.beta)

    def updateFirmPrice(self):
        self.firms.price = np.random.normal(self.alpha, self.varpf**2, size=self.numberOfFirms)

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

    def reportResults(self, time):
        totalCapital = np.sum(self.firms.capital)
        totalOutput = np.sum(self.firms.output)
        self.firmOutputReport[time] = totalOutput
        self.firmCapitalReport[time] = totalCapital
        self.firmAvgPrice[time] = np.mean(self.firms.price)
        self.firmWealthReport[time] = np.sum(self.firms.networth)
        self.firmDebtReport[time] = np.sum(self.firms.debt)
        self.firmProfitReport[time] = np.sum(self.firms.profit)
        self.firmDefaultReport[time] = np.count_nonzero(self.firms.default)

        # Gather the results of the last agent
        firmsResults = []
        for i in [self.firms.output, self.firms.capital, self.firms.price,
                    self.firms.networth, self.firms.debt, self.firms.profit,
                    self.firms.default, self.firms.interestRate]:
                firmsResults.append(i[-1])
        self.individualFirm = np.concatenate((self.individualFirm, np.array([firmsResults])))

        # Gather aggregate bank results
        self.bankWealthReport[time] = np.sum(self.banks.networth)
        self.bankDebtReport[time] = np.sum(self.banks.badDebt)
        self.bankProfitReport[time] = np.sum(self.banks.profit)
        self.bankDefaultReport[time] = np.count_nonzero(self.banks.default)

        self.GDP[time] = totalOutput
        self.avgInterest[time] = np.mean(self.banks.interestRate)

    def saveResults(self):

        try:
            os.mkdir(self.resultFolder)
        except FileExistsError:
            print("Simulation with this seed exists")
            override = input("Overwrite results? [Y/n]: ")
            if "N" in override.upper() or ("N" in override.upper() and "Y" in override.upper()):
                exit()

        print("Writing simulation results with seed " + str(self.seed))

        f = open(self.resultFolder + "aggregateResults.csv", "w+")
        output = np.stack((self.firmOutputReport,
                                self.firmCapitalReport,
                                self.firmAvgPrice,
                                self.firmWealthReport,
                                self.firmDebtReport,
                                self.firmProfitReport,
                                self.firmDefaultReport,
                                self.bankWealthReport,
                                self.bankDebtReport,
                                self.bankProfitReport,
                                self.bankDefaultReport,
                                self.GDP,
                                self.avgInterest))
        np.savetxt(f, output.transpose(), delimiter=",")
        f.close()

        # Write results for special firm
        f = open(self.resultFolder + "individualFirmResults.csv", "w+")
        self.individualFirm = self.individualFirm[1:] # remove first row which is just zeros
        np.savetxt(f, self.individualFirm, delimiter=",")
        f.close()

    def interactiveShell(self):

        def continueCmd(self, args):
            if (not args) or (args[0] == ""):
                self.continueUntilTime = self.currentStep + 1
            else:
                self.continueUntilTime = int(args[0])
            if self.continueUntilTime <= self.currentStep:
                raise ValueError

        def help():
            print("List of commands:\n")
            print("{0:20} -- {1}".format("continue","Step simulation forward"))
            print("{0:20} -- {1}".format("continue [step]","Step simulation forward to particular step"))
            print("{0:20} -- {1}".format("exit/quit","Quit simulation"))
            print("{0:20} -- {1}".format("help","Show this list of commands"))
            print("{0:20} -- {1}".format("list","List variables of simulation"))
            print("{0:20} -- {1}".format("print", "Print simulation variable"))

        def listVariables(self):
            print("\nVariables: {0:5}, {1:5}\n".format("firms", "banks"))
            print("Firms attributes:")
            print("\t{0:20} {1:20} {2:20}".format("numberOfFirms", "price", "debt"))
            print("\t{0:20} {1:20} {2:20}".format("networth", "profit", "interestRate"))
            print("\t{0:20} {1:20} {2:20}".format("leverage", "capital", "output"))
            print("\t{0:20} {1:20}".format("lgdf", "default", "debt"))
            print("\n")
            print("Banks attributes:")
            print("\t{0:20} {1:20} {2:20}".format("numberOfFirms", "price", "badDebt"))
            print("\t{0:20} {1:20} {2:20}".format("networth", "profit", "interestRate"))
            print("\t{0:20} {1:20} {2:20}".format("deposit", "creditLinkDegree", "nonPerformingLoans"))
            print("\t{0:20}".format("default"))

        def printVar(self, args):
            try:
                if (not args[0].startswith("firms.")) and (not args[0].startswith("banks.")):
                    raise ValueError
                exec("print(self." + args[0] + ")")
            except (ValueError, IndexError):
                print("Invalid use of command: print")
                print("Usage:")
                print("\tprint [variable].[attribute]")
                print("Use command list to see valid variables and attributes")
            except SyntaxError:
                print("Invalid use of command: print")
                print(str(args[0][:6]) + " has no attribute " + \
                        str(args[0][6:]))

        while(True):
            try:
                shellCommand = str(input(">>> "))
                originalCommand = shellCommand
                shellCommand = shellCommand.lower().split(" ")
                cmd = shellCommand[0]
                args = shellCommand[1:]
                if ("exit" == cmd) or ("quit" == cmd):
                    print("Do you wish to save results?")
                    answer = input("[Y/N] ").lower()
                    if ("y" == answer) or ("yes" == answer):
                        self.saveResults()
                    sys.exit()
                elif ("continue" == cmd) or ("c" == cmd):
                    try:
                        continueCmd(self, args)
                    except (ValueError, IndexError):
                        print("Invalid use of command: continue")
                        print("\tUsage: continue [time to continue to]")
                        continue
                    break
                elif ("help" == cmd) or ("h" == cmd):
                    help()
                elif ("list" == cmd) or ("l" == cmd):
                    listVariables(self)
                elif ("print" == cmd) or ("p" == cmd):
                    printVar(self, args)
            except EOFError:
                sys.exit()
            except AttributeError as e:
                print(e)

    def run(self):
        print("Running Simulation...")
        for t in range(self.time):
            self.currentStep = t
            # replace defaulted firms and banks
            try:
                self.replaceDefaults()
            except Exception as e:
                print("Problem with replacing defaulted firms")
                print(e)

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

            self.reportResults(t)

            if (self.mode == "Interactive") and \
                    (self.continueUntilTime == self.currentStep):
                print("Time: ", t)
                self.interactiveShell()

        self.saveResults()

