import numpy as np
import os
import agents
import sys
import time
from scipy import stats
import matplotlib.pyplot as plt

class Simulation:

    def __init__(self,
            time, # time simulation is ran for
            numberOfFirms,
            numberOfBanks,
            alpha=0.1, # mean of firms price
            varpf=0.4, # variance of firms price
            gamma=0.02, # interest rate parameter
            chi=5, # number of potential partners on credit market
            lambd=4, # intensity of choice
            adj=0.1, # leverage adjustment
            phi=3, # production function parameter
            beta=0.7, # production function parameter
            rCB=0.02, # central bank interest rate
            cB=0.01, # banks costs
            mode=None, # mode to run simulation
            growth=False, # if enabled add agents through every step
            seed=None,
            outputFolder=None
            ):
        self.time = int(time)
        self.numberOfFirms = int(numberOfFirms)
        self.numberOfBanks = int(numberOfBanks)
        self.alpha = float(alpha)
        self.varpf = float(varpf)
        self.gamma = float(gamma)
        self.chi = int(chi)
        self.lambd = float(lambd)
        self.adj = float(adj)
        self.phi = float(phi)
        self.beta = float(beta)
        self.rCB = float(rCB)
        self.cB = float(cB)
        self.bestFirm = 0

        self.mode = mode
        self.growthEnabled = growth
        self.continueUntilTime = 0

        if seed == None:
            self.seed = np.random.randint(900000)
        else:
            self.seed = int(seed)

        np.random.seed(self.seed)

        self.firms = agents.Firms(self.numberOfFirms, self.alpha, self.varpf)
        self.banks = agents.Banks(self.numberOfBanks)
        self.economy = agents.Economy(self.time)

        # firms-banks credit matching adjacency matrix
        # firms borrow from 1 bank but banks have multiple clients
        self.link_fb = np.zeros((self.numberOfFirms, self.numberOfBanks))
        banksWithFirms = np.ceil(np.random.uniform(0, self.numberOfBanks-1, self.numberOfFirms))
        for i in range(self.numberOfFirms):
            self.link_fb[i][int(banksWithFirms[i])] = 1


        # contains banks that firms use as lookup for interestRates
        self.bankPools = np.zeros((self.numberOfFirms, self.chi))

        if not outputFolder:
            self.outputFolder = "results/" + str(self.seed) + "/"
        else:
            if os.path.lexists(outputFolder):
                raise ValueError("Output folder already exists")
            self.outputFolder = outputFolder
        if self.outputFolder[-1] != "/" and os.name == "posix":
            self.outputFolder += "/"
        elif self.outputFolder[-1] != "\\" and os.name == "nt":
            self.outputFolder += "\\"


        # Output variables
        self.changeFB = np.array([0]*self.time, dtype=float) # monitors how many firms change bank
        self.firmOutputReport = np.array([0]*self.time, dtype=float)
        self.firmCapitalReport = np.array([0]*self.time, dtype=float)
        self.firmWealthReport = np.array([0]*self.time, dtype=float)
        self.firmDebtReport = np.array([0]*self.time, dtype=float)
        self.firmProfitReport = np.array([0]*self.time, dtype=float)
        self.firmAvgPrice = np.array([0]*self.time, dtype=float)
        self.firmDefaultReport = np.array([0]*self.time, dtype=float)

        # array to store price, wealth, capital,... from a single firm
        self.individualFirm = np.array([[0,0,0,0,0,0,0,0,0]], dtype=float)

        self.bankWealthReport = np.array([0]*self.time, dtype=float)
        self.bankDebtReport = np.array([0]*self.time, dtype=float)
        self.bankProfitReport = np.array([0]*self.time, dtype=float)
        self.bankDefaultReport = np.array([0]*self.time, dtype=float)

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
            # select potential partners
            potentialPartners = self.bankPools[f]

            # select best bank out of potentia partners
            bestBankIndex = self.findBestBank(potentialPartners)
            newInterest = self.banks.interestRate[bestBankIndex]

            # pick up interest of old partner
            currentBank = np.nonzero(self.link_fb[f])[0]
            if not currentBank:
                oldInterest = np.inf
            else:
                oldInterest = self.banks.interestRate[currentBank[0]]

            # compare old bank with new
            if ( newInterest < oldInterest ):
                # log change in firm-bank relationship
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

        # create new firms
        self.firms.networth[defaulted] = np.random.uniform(2, size=len(defaulted))
        self.firms.leverage[defaulted] = 1
        self.firms.price[defaulted] = np.random.normal(self.alpha, np.sqrt(self.varpf), size=len(defaulted))
        self.firms.interestRate[defaulted] = self.rCB + self.banks.interestRate[ \
                np.nonzero(self.link_fb[defaulted])[1]] + self.gamma * \
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
        self.firms.price = np.random.normal(self.alpha, np.sqrt(self.varpf), size=self.numberOfFirms)

    def updateFirmInterestRate(self):
        bestFirmWorth = self.maxFirmWealth()
        banksOfFirms = np.nonzero(self.link_fb)[1]
        self.firms.interestRate = self.rCB + self.banks.interestRate[banksOfFirms] + \
                             self.gamma*(self.firms.leverage) / \
                             ((1+self.firms.networth/bestFirmWorth))
        
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

        firmsPriceGreaterInterest = np.where(self.firms.price > self.firms.interestRate)
        firmsPriceLessInterest = np.where(self.firms.price <= self.firms.interestRate)

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

    def addAgents(self, t):
        if t % 20 == 0:
            self.banks.addBank()
            self.numberOfBanks += 1

            self.firms.addFirm()
            self.numberOfFirms += 1

            banksWithNewFirm = np.ceil(np.random.uniform(0, self.numberOfBanks-1, self.numberOfFirms))
            self.link_fb = np.column_stack((self.link_fb, np.zeros(self.numberOfFirms-1)))

            self.link_fb = np.vstack((self.link_fb, np.zeros(self.numberOfBanks)))
            self.link_fb[-1][int(banksWithNewFirm[0])] = 1
        else:
            self.firms.addFirm()
            self.numberOfFirms += 1
            banksWithNewFirm = np.ceil(np.random.uniform(0, self.numberOfBanks-1, self.numberOfFirms))
            self.link_fb = np.vstack((self.link_fb, np.zeros(self.numberOfBanks)))
            self.link_fb[-1][int(banksWithNewFirm[0])] = 1
        

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
                    self.firms.default, self.firms.interestRate, self.firms.leverage]:
                firmsResults.append(i[-1])
        self.individualFirm = np.concatenate((self.individualFirm, np.array([firmsResults])))

        # Gather aggregate bank results
        self.bankWealthReport[time] = np.sum(self.banks.networth)
        self.bankDebtReport[time] = np.sum(self.banks.badDebt)
        self.bankProfitReport[time] = np.sum(self.banks.profit)
        self.bankDefaultReport[time] = np.count_nonzero(self.banks.default)

        self.economy.GDP[time] = totalOutput
        self.economy.badDebtAsGDP[time] = np.mean(self.banks.badDebt*100 / totalOutput)
        self.economy.avgInterest[time] = np.mean(self.banks.interestRate)
        self.economy.leverage[time] = self.firmDebtReport[time] / self.firmWealthReport[time]

    def saveResults(self):

        try:
            os.mkdir(self.outputFolder)
        except FileExistsError:
            print("Simulation with this seed exists")
            override = input("Overwrite results? [Y/n]: ")
            if "N" in override.upper() or ("N" in override.upper() and "Y" in override.upper()):
                exit()

        infoFile = open(self.outputFolder + "INFO", "+w")
        infoFile.write("### Simulation Configuration ###\n")
        infoFile.write("\t{0:35} = {1:5}\n".format("Seed", self.seed))
        date = time.strftime("%d %b %Y: %H:%M", time.gmtime())
        infoFile.write("\t{0:35} = {1:5}\n".format("Date ran", date))
        infoFile.write("\t{0:35} = {1:5}\n".format("Total Steps", self.time))
        infoFile.write("\t{0:35} = {1:5}\n".format("Number of Firms", self.numberOfFirms))
        infoFile.write("\t{0:35} = {1:5}\n".format("Number of Banks", self.numberOfBanks))
        infoFile.write("\t{0:35} = {1:5}\n".format("Price Mean (alpha)", self.alpha))
        infoFile.write("\t{0:35} = {1:5}\n".format("Price Variance", self.varpf))
        infoFile.write("\t{0:35} = {1:5}\n".format("Interest rate param (gamma)", self.gamma))
        infoFile.write("\t{0:35} = {1:5}\n".format("Number of potential parters (chi)", self.chi))
        infoFile.write("\t{0:35} = {1:5}\n".format("Intensity of choice (lambd)", self.lambd))
        infoFile.write("\t{0:35} = {1:5}\n".format("Leverage adjustment (adj)", self.adj))
        infoFile.write("\t{0:35} = {1:5}\n".format("Production function param (phi)", self.phi))
        infoFile.write("\t{0:35} = {1:5}\n".format("Production function param (beta)", self.beta))
        infoFile.write("\t{0:35} = {1:5}\n".format("Central bank interest rate (rCB)", self.rCB))
        infoFile.write("\t{0:35} = {1:5}\n\n".format("Bank costs (cB)", self.cB))

        infoFile.write("### Quick Analysis ###\n")
        infoFile.write("{0:20} = {1:5}\n".format("Mean leverage", np.mean(self.economy.leverage[300:])))
        infoFile.write("{0:20} = {1:5}\n".format("Mean firm default", np.mean(self.firmDefaultReport)))
        infoFile.write("{0:20} = {1:5}\n".format("Mean banks default", np.mean(self.bankDefaultReport)))
        infoFile.write("{0:20} = {1:5}\n".format("Mean percentage GDP as non-performing loans", \
                np.mean(self.economy.badDebtAsGDP[300:])))

        # Report simulation fails
        if np.all(self.firms.price == 0):
            infoFile.write("Warning: Firm price error\n")
        if np.all(self.economy.leverage == 0):
            infoFile.write("Warning: Economy leverage error\n")
        if np.mean(self.economy.badDebtAsGDP[300:]) > 3:
            infoFile.write("Warning: High Bad Debt\n")

        infoFile.close()

        print("Writing simulation results with seed " + str(self.seed))

        columnNames = ["Firms Aggregate Output", "Firms Aggregate Capital", \
                "AVG Firm Price", "Firms Aggregate Wealth", "Firms Aggregate Debt", \
                "Firms Aggregate Profit", "Total defaulted firms", \
                "Banks Aggregate Wealth", "Banks Aggregate Debt", \
                "Banks Aggregate Profit", "Total defaulted banks", \
                "Economy GDP", "Average interest rate", "Economy Debt-Equity Ratio"]

        f = open(self.outputFolder + "aggregateResults.csv", "w+")

        for name in columnNames:
            f.write(name+", ")
        f.write('\n')

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
                                self.economy.GDP,
                                self.economy.badDebtAsGDP,
                                self.economy.avgInterest,
                                self.economy.leverage))
        np.savetxt(f, output.transpose(), delimiter=",")
        f.close()

        columnNames = ["Firm Output", "Firm Capital", "Firm Price", \
                "Firm Networth", "Firm Debt", "Firm Profit", \
                "Defaulted", "Firm Interest Rate", "Firm Leverage"]

        # Write results for special firm
        f = open(self.outputFolder + "individualFirmResults.csv", "w+")
        for name in columnNames:
            f.write(name+", ")
        f.write('\n')
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
            print("Continuing until time ", self.continueUntilTime)

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
                if (not args[0].startswith("firms.")) and (not args[0].startswith("banks."))\
                        and (not args[0].startswith("economy.")):
                    raise ValueError
                exec("print(self." + args[0] + ")")
            except (ValueError, IndexError):
                print("Invalid use of command: print")
                print("Usage:")
                print("\tprint [variable].[attribute]")
                print("Use command list to see valid variables and attributes")
                print("")
                print("Examples:")
                print("\tprint firms.price")
                print("")
                print("\tprint firms.networth[3]")
            except SyntaxError:
                print("Invalid use of command: print")
                print(str(args[0][:6]) + " has no attribute " + \
                        str(args[0][6:]))

        while(True):
            try:
                shellCommand = str(input(">>> "))
                originalCommand = shellCommand
                shellCommand = shellCommand.split(" ")
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
                print("Exiting simulation, results will not be saved")
                sys.exit()
            except AttributeError as e:
                print(e)

    def run(self):
        print("Running Simulation...")
        for t in range(self.time):
            self.currentStep = t

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

            # replace defaulted firms and banks
            try:
                self.replaceDefaults()
            except Exception as e:
                print("Problem with replacing defaulted firms")
                print(e)

            if self.growthEnabled:
                self.addAgents(t)

        self.saveResults()

