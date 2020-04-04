import sys
import traceback
import glob
import csv
import re
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import agents
import analyse
from scipy.interpolate import splev, splrep
from scipy import stats

resultNames = {0: "Output", 1: "Capital",
            2 : "Price", 3 : "Wealth",
            4 : "Debt", 5 : "Profit",
            6 : "Default", 7: "Interest"}

def plot(data, data2=None, data3=None, data4=None, title=""):

    maxValue = -np.inf
    minValue = 0
    fig, ax = plt.subplots()
    try:
        for i in [data, data2, data3, data4]:
            if i is None:
                continue
            if (type(i) is not list) and (type(i) is not np.ndarray):
                raise ValueError("Error plot: data must be list")
            if np.amax(i) > maxValue:
                maxValue = np.amax(i)
            if np.amin(i) < minValue:
                minValue = np.amin(i)
            ax.plot(i)
        if title != "":
            ax.set_title(title)
        plt.ylim(minValue, maxValue)
        fig.show()
    except NameError:
        print("Error plot: data must be passed to function")
        return

def pearCoeffs(x,y, stepsize):
    index = 0
    pearson = []
    domain = []
    for i in range(len(y)-stepsize):
        p = stats.pearsonr(x[i:stepsize+i], y[i:stepsize+i])
        if p[1] < 0.2:
            pearson.append(p[0])
            domain.append(index)
        index += 1

    return domain, pearson

def tempAnalysis():
    y = firms.output
    time = np.linspace(0, len(y), num=len(y))

    spline = splrep(time, y, k=3, s=25)
    x = np.linspace(0, len(y), int(len(y)/6))
    y2 = splev(x, spline)

    plt.plot(y, label='Firm output')
    plt.plot(x, y2, label='splines')
    plt.legend()
    plt.grid(True)
    plt.show()


    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(x, y2, label='splines')
    ax[0].legend()
    ax[0].grid(True)
    x, pearson = pearCoeffs(economy.badDebtAsGDP, economy.leverage, 15)
    ax[1].scatter(x,pearson)
    ax[1].set_ylim(-1,1)
    ax[1].grid(True)
    ax[1].set_title("Pearson correlation")
    plt.show()

def tempAnalysis2():

    NL = []; LN = []; DL = []; LD = [];
    for st in range(0,20):
        NL.append(stats.pearsonr(firms.networth[0:100], economy.leverage[0+st:100+st]))
        LN.append(stats.pearsonr(economy.leverage[0:100], firms.networth[0+st:100+st]))
        DL.append(stats.pearsonr(economy.badDebtAsGDP[0:100], economy.leverage[0+st:100+st]))
        LD.append(stats.pearsonr(economy.leverage[0:100], economy.badDebtAsGDP[0+st:100+st]))

    plt.plot(NL, label="networth - leverage+")
    plt.show()

    plt.plot(LN, label="leverage - networth+")
    plt.show()

    plt.plot(DL, label="debt - leverage+")
    plt.show()

def classify(key):

    if not key or type(key) != str:
        raise ValueError

    simulationRuns = np.array([])
    Y = np.array([])
    for folder in glob.glob("results/*/"):
        firms, banks, individualfirm, paramters = openSimulationFiles(folder)

        time = np.linspace(0, len(firms.output), num=len(firms.output)-1)
        p = np.stack((time, firms.output[1:]), axis=-1)
        interpolatedPoints = analyse.splineData(p)
        ddy = np.gradient(np.gradient(interpolatedPoints[:,1]))
        stationaryPoints = analyse.findStationaryPoints(ddy)
        change = analyse.outputVolatility(firms)

        Y = np.append(Y, float(paramters.get(key)))

        # Store imporant simulation features
        features = np.stack((len(stationaryPoints),
                            np.mean(change),
                            np.var(change),
                            np.sum(firms.default),
                            np.sum(banks.default),
                            np.mean(firms.output),
                            np.mean(banks.profit),
                            np.mean(banks.interestRate)))
        for param in paramters.values():
            try:
                features = np.append(features, float(param))
            except ValueError:
                continue

        if not np.any(simulationRuns):
            simulationRuns = features
        else:
            simulationRuns = np.vstack((simulationRuns, features))

    #classifier = PCA().fit(simulationRuns)
    #X = classifier.transform(simulationRuns)
    #print(classifier.n_components_)
    #print(classifier.explained_variance_ratio_)

    classifier = SVC()
    encoder = preprocessing.LabelEncoder()
    Yclass = encoder.fit_transform(Y)
    try:
        classifier.fit(simulationRuns, Yclass)
    except ValueError:
        print("Simulations do not differ by parameter {0:5}".format(key))
        return

    fig, ax = plt.subplots()
    numOfClasses = np.amax(Yclass)+1
    for i in range(numOfClasses):
        classRuns = np.where(Yclass == i)[0]
        classLabel = key+"="+str(Y[classRuns][0])
        ax.scatter(simulationRuns[classRuns,13], \
            simulationRuns[classRuns,2], \
            label=classLabel)

    plt.legend()
    plt.show()

    return

#
# Returns the simulation with a parameter (key) = value
#
def findSimulations(key, value):

    sims = np.array([])
    for folder in glob.glob("results/*/"):
        firms, banks, individuafirm, economy, parameters = openSimulationFiles(folder)

        if parameters.get(key) == value:
            sims = np.append(sims, folder)

    if len(sims) == 0:
        print("No simulation found with " + key + "=" + value)

    return sims

def openSimulationFiles(folder):

    firmsKeys = ['Output', 'Capital',
            'Price', 'Wealth',
            'Debt', 'Profit',
            'Default']
    individualFirmKeys = ['Output', 'Capital',
            'Price', 'Wealth',
            'Debt', 'Profit',
            'Default','Interest',
            'Leverage']
    banksKeys = ['Wealth', 'Debt',
            'Profit', 'Default']

    parameters = {"seed": [], "date": [], "steps": [], "firms": [], "banks": [], \
                "mean": [], "variance": [], "gamma": [], "lambd": [], \
                "adj": [], "phi": [], "beta": [], "rcb": [], "cb": []}

    firms = agents.Firms()
    banks = agents.Banks()
    individualfirm = agents.IndividualFirm()
    economy = agents.Economy(0)

    try:
        with open(folder + "aggregateResults.csv", "r") as f:
            reader = csv.reader(f)
            lines = np.array(list(reader)[300:], dtype=float) # ignore first 300 lines
            firms.output = lines.transpose()[0] # sum of total firm output at each step
            firms.capital = lines.transpose()[1] # sum of total firm capital at each step
            firms.price = lines.transpose()[2] # average firm price at each step
            firms.networth = lines.transpose()[3] # sum of total networth at each step
            firms.debt = lines.transpose()[4]
            firms.profit = lines.transpose()[5]
            firms.default = lines.transpose()[6] # sum of total firm defaults at each step

            banks.networth = lines.transpose()[7]
            banks.badDebt = lines.transpose()[8]
            banks.profit = lines.transpose()[9]
            banks.default = lines.transpose()[10]
            economy.GDP = lines.transpose()[11]
            economy.badDebtAsGDP = lines.transpose()[12]
            economy.avgInterest = lines.transpose()[13]
            economy.leverage = lines.transpose()[14]
        with open(folder + "individualFirmResults.csv", "r") as f:
            reader = csv.reader(f)
            lines = np.array(list(reader)[300:], dtype=float)
            individualfirm.output = lines.transpose()[0]
            individualfirm.capital = lines.transpose()[1]
            individualfirm.price = lines.transpose()[2]
            individualfirm.networth = lines.transpose()[3]
            individualfirm.debt = lines.transpose()[4]
            individualfirm.profit = lines.transpose()[5]
            individualfirm.default = lines.transpose()[6]
            individualfirm.interest = lines.transpose()[7]
            individualfirm.leverage = lines.transpose()[8]
        with open(folder + "INFO", "r") as f:
            lines = list(f)
            index = 0
            for line in lines:
                line = line.lower().split()
                if line:
                    value = line[-1]
                for key in parameters.keys():
                    for word in line:
                        if re.match('.*'+key+'.*', word):
                            parameters.update({key: value})

    except FileNotFoundError:
        print("No file found")
        exit()

    return firms, banks, individualfirm, economy, parameters

def selectResults(files):
    choice = 0
    if len(files) == 0:
        print("No result files to read")
        print("Please run simulator first")
        exit()
    if len(files) == 1:
        choice = 0
    if len(files) > 1:
        print("Choose a simulation run to analyse")
        index = 0
        for i in files:
            print("[" + str(index) + "]: " + i)
            index += 1
        try:
            choice = int(input(">>> "))
            if choice >= len(files) or choice < 0:
                print("Please give valid choice")
                return -1
        except ValueError:
            print("Invalid Input")
            return -1
    return choice

def executeCommand(cmd):
    cmd = cmd.split(" ")

    args = ""
    if cmd[0] == "classify":
        try:
            classify(cmd[1])
        except (IndexError, TypeError):
            print("Usage: classify [simulation attribute]")
    if cmd[0] == "exec":
        exec(cmd[1])
    if cmd[0] == "exit" or cmd[0] == "quit":
        raise EOFError
    if cmd[0] == "find":
        print(findSimulations(cmd[1], cmd[2]))
    if cmd[0] == "test":
        tempAnalysis()
    if cmd[0] == "test2":
        tempAnalysis2()
    if cmd[0] == "open":
        choice = selectResults(folders)
        if choice > len(folders) or choice < 0:
            return
        print("Opening results from " + folders[choice])
        global firms, banks, individualfirm, economy, parameters
        firms, banks, individualfirm, economy, parameters = openSimulationFiles(folders[choice])

    if cmd[0] == "plot":

        for i in cmd[1:]:
            if (not i.startswith("firms")) and (not i.startswith("banks")) \
                    and (not i.startswith("individual")) and (not i.startswith("economy")):
                print("Invalid argument: ", i)
                continue
            else:
                args += i+","
        if args == "":
            return
        # if we get here cmd is valid
        exec(cmd[0] + "(" + args + ")")

    if cmd[0] == "printparams" or cmd[0] == "params":
        for x in parameters:
            print ("{0:20}: {1:10}".format(x,parameters[x]))
        
    if cmd[0] == "help":
        print("List of commands:\n")
        print("{0:20} -- {1}".format("exec [python code]", "USE WITH CARE"))
        print("{0:20} -- {1}".format("exit/quit", "Quit simulation reader"))
        print("{0:20} -- {1}".format("find [parameter] [value]", "Finds simulations with parameter=value"))
        print("{0:20} -- {1}".format("open", "Open simulation files for analysis"))
        print("{0:20} -- {1}".format("plot [data list]", "Plots data as line graph"))
        print("{0:20} -- {1}".format("plot [data list1] [data list2]", "Plots data as line graph"))
        print("{0:20} -- {1}".format("params/printparams", "Print simulation parameters"))
        print("{0:20} -- {1}".format("help", "Shows this list of commands"))
    if cmd[0] == "list":
        print("\nVariables: {0:5}, {1:5}, {1:5}, {1:5}\n".format("firms", "individualfirm", "banks", "economy"))
        print("Firms attributes:")
        print("\t{0:20} {1:20}".format("price", "debt"))
        print("\t{0:20} {1:20}".format("networth", "profit"))
        print("\t{0:20} {1:20}".format("capital", "output"))
        print("\t{0:20}".format("default"))
        print("\n")
        print("Individual Firm attributes:")
        print("\t{0:20} {1:20}".format("price", "debt"))
        print("\t{0:20} {1:20}".format("networth", "profit"))
        print("\t{0:20} {1:20}".format("capital", "output"))
        print("\t{0:20} {1:20}".format("default", "interest"))
        print("\t{0:20}".format("leverage"))
        print("\n")
        print("Banks attributes:")
        print("\t{0:20} {1:20}".format("networth", "badDebt"))
        print("\t{0:20} {1:20}".format("profit", "default"))
        print("Economy attributes:")
        print("\t{0:20} {1:20}".format("GDP", "avgInterest"))
        print("\t{0:20}".format("leverage"))


if __name__=="__main__":
    folders = glob.glob("results/*/")
    choice = 0
    while(True):
        try:
            choice = selectResults(folders)
            if choice != -1:
                break
        except EOFError:
            print("Exiting reader")
            sys.exit()

    print("Opening results from " + folders[choice])
    global firms, banks, individualfirm, economy, parameters
    firms, banks, individualfirm, economy, parameters = openSimulationFiles(folders[choice])

    while(True):
        try:
            shellCommand = str(input(">>> "))
            executeCommand(shellCommand)
        except EOFError:
            print("Exiting reader")
            sys.exit()
        except AttributeError as e:
            print(e)

    sys.exit()



