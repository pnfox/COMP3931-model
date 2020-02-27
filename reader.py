import sys
import glob
import csv
import re
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import preprocessing
import agents
import analyse

resultNames = {0: "Output", 1: "Capital",
            2 : "Price", 3 : "Wealth",
            4 : "Debt", 5 : "Profit",
            6 : "Default", 7: "Interest"}

def plot(data, data2=None, data3=None, data4=None, title=""):
    fig, ax = plt.subplots()
    try:
        for i in [data, data2, data3, data4]:
            if i is None:
                continue
            if (type(i) is not list) and (type(i) is not np.ndarray):
                raise ValueError("Error plot: data must be list")
            ax.plot(i)
        if title != "":
            ax.set_title(title)
        fig.show()
    except NameError:
        print("Error plot: data must be passed to function")
        return

def classify(key):

    if not key or type(key) != str:
        raise ValueError

    simulationRuns = np.array([])
    Y = np.array([])
    for folder in glob.glob("results/*/"):
        firms, banks, individualFirm, paramters = openSimulationFiles(folder)

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
        print("Only one class found for {0:5}".format(key))
        return

    print(classifier.dual_coef_)
    plt.scatter(simulationRuns[:,1], simulationRuns[:,2], c=Y)
    plt.show()

    return

def openSimulationFiles(folder):

    firmsKeys = ['Output', 'Capital',
            'Price', 'Wealth',
            'Debt', 'Profit',
            'Default']
    individualFirmKeys = ['Output', 'Capital',
            'Price', 'Wealth',
            'Debt', 'Profit',
            'Default','Interest']
    banksKeys = ['Wealth', 'Debt',
            'Profit', 'Default']

    economy = {"GDP": [], "Avg interest":[]}
    parameters = {"seed": [], "date": [], "steps": [], "firms": [], "banks": [], \
                "mean": [], "variance": [], "gamma": [], "lambd": [], \
                "adj": [], "phi": [], "beta": [], "rcb": [], "cb": []}

    firms = agents.Firms()
    banks = agents.Banks()
    individualFirm = agents.IndividualFirm()

    try:
        with open(folder + "aggregateResults.csv", "r") as f:
            reader = csv.reader(f)
            lines = np.array(list(reader)[300:], dtype=float) # ignore first 300 lines
            firms.output = lines.transpose()[0]
            firms.capital = lines.transpose()[1]
            firms.price = lines.transpose()[2]
            firms.networth = lines.transpose()[3]
            firms.debt = lines.transpose()[4]
            firms.profit = lines.transpose()[5]
            firms.default = lines.transpose()[6]

            banks.networth = lines.transpose()[7]
            banks.badDebt = lines.transpose()[8]
            banks.profit = lines.transpose()[9]
            banks.default = lines.transpose()[10]
            economy.get("GDP").append(lines.transpose()[11])
            economy.get("Avg interest").append(lines.transpose()[12])
        with open(folder + "individualFirmResults.csv", "r") as f:
            reader = csv.reader(f)
            lines = np.array(list(reader)[300:], dtype=float)
            individualFirm.output = lines.transpose()[0]
            individualFirm.capital = lines.transpose()[1]
            individualFirm.price = lines.transpose()[2]
            individualFirm.networth = lines.transpose()[3]
            individualFirm.debt = lines.transpose()[4]
            individualFirm.profit = lines.transpose()[5]
            individualFirm.default = lines.transpose()[6]
            individualFirm.interest = lines.transpose()[7]
        with open(folder + "INFO", "r") as f:
            lines = list(f)
            index = 0
            for line in lines:
                line = line.lower().split()
                value = line[-1]
                for key in parameters.keys():
                    for word in line:
                        if re.match('.*'+key+'.*', word):
                            parameters.update({key: value})

    except FileNotFoundError:
        print("No file found")
        exit()

    return firms, banks, individualFirm, parameters

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
    cmd = cmd.lower()
    cmd = cmd.split(" ")

    args = ""
    if cmd[0] == "exec":
        exec(cmd[1])
    if cmd[0] == "exit" or cmd[0] == "quit":
        raise EOFError
    if cmd[0] == "classify":
        try:
            classify(cmd[1])
        except (IndexError, TypeError):
            print("Usage: classify [simulation attribute]")
    if cmd[0] == "plot":

        for i in cmd[1:]:
            if (not i.startswith("firms")) and (not i.startswith("banks")) \
                    and (not i.startswith("individual")):
                print("Invalid argument: ", i)
                continue
            else:
                args += i+","
        if args == "":
            return
        # if we get here cmd is valid
        exec(cmd[0] + "(" + args + ")")
    if cmd[0] == "help":
        print("List of commands:\n")
        print("{0:20} -- {1}".format("exec [python code]", "USE WITH CARE"))
        print("{0:20} -- {1}".format("exit/quit", "Quit simulation reader")) 
        print("{0:20} -- {1}".format("plot [data list]", "Plots data as line graph"))
        print("{0:20} -- {1}".format("plot [data list1] [data list2]", "Plots data as line graph"))
        print("{0:20} -- {1}".format("help", "Shows this list of commands"))
    if cmd[0] == "list":
        print("\nVariables: {0:5}, {1:5}, {1:5}\n".format("firms", "individualFirm", "banks"))
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
        print("\n")
        print("Banks attributes:")
        print("\t{0:20} {1:20}".format("networth", "badDebt"))
        print("\t{0:20} {1:20}".format("profit", "default"))


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
    firms, banks, individualFirm, parameters = openSimulationFiles(folders[choice])

    while(True):
        try:
            shellCommand = str(input(">>> ")).lower()
            executeCommand(shellCommand)
        except EOFError:
            print("Exiting reader")
            sys.exit()
        except AttributeError as e:
            print(e)

    sys.exit()



