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
import validation
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

def multiplot(firms, banks):
    plt.plot(firms.output, linewidth=2)
    plt.savefig("../COMP3931-report/images/firmoutput.png", bbox_inches = 'tight',
    pad_inches = 0)
    plt.clf()
    plt.plot(firms.debt, linewidth=2)
    plt.savefig("../COMP3931-report/images/firmdebt.png", bbox_inches = 'tight',
    pad_inches = 0)
    plt.clf()
    plt.plot(firms.price)
    #plt.hist(firms.price, bins=40)
    plt.savefig("../COMP3931-report/images/firmprice.png", bbox_inches = 'tight',
    pad_inches = 0)
    plt.clf()
    plt.plot(banks.networth, linewidth=2)
    plt.savefig("../COMP3931-report/images/banknetworth.png", bbox_inches = 'tight',
    pad_inches = 0)
    plt.clf()

def pearCoeffs(x,y, stepsize, a):
    index = a
    pearson = []
    domain = []
    for i in range(len(y)-stepsize):
        p = stats.pearsonr(x[i:stepsize+i], y[i:stepsize+i])
        if p[1] < 0.05:
            pearson.append(p[0])
            domain.append(index)
        index += a

    return domain, pearson

def spline(data, smooth=2, scale=1):
    x = np.linspace(0, len(data), num=len(data))
    spline = splrep(x, data, k=3, s=smooth)
    x = np.linspace(0, len(data), int(len(data)*scale)) # if this x has more than len(data) lots of pearson oscillation
    return x, splev(x, spline)

def normalize(data):
    data = (data - np.mean(data)) / np.std(data)
    return data

def tempAnalysis():

    # magic algorithm that transforms data into more useful stuff
    # hopefully this makes cycles more clear
    x, smoothLeverage = spline(economy.leverage, \
            len(economy.leverage)*np.var(economy.leverage)*0.5, 3)
    #smoothLeverage = np.gradient(smoothLeverage)

    # plot algorithm output
    fig,ax = plt.subplots(3,1,sharex=True)
    ax[0].plot(banks.profit)
    ax[0].set_title("Economy output")
    ax[0].grid(True)
    ax[1].plot(economy.leverage)
    ax[1].set_title("Economy leverage")
    ax[1].grid(True)
    extend=[-0.5, 800.5, 0, 1]
    ax[2].imshow(smoothLeverage[np.newaxis,:], cmap="plasma", aspect="auto", extent=extend)
    ax[2].set_title("magic")
    ax[2].grid(True)
    plt.show()

    print("Plotting normalized leverage and normalized smooth leverage")
    normalizedLeverage = normalize(economy.leverage)
    normalizedChange = normalize(smoothLeverage)

    plt.plot(normalizedLeverage)
    plt.plot(x, normalizedChange)
    plt.show()

    x2, smoothNetworth = spline(firms.output, \
        len(firms.output)*np.var(firms.output)*0.1, 3)
    normalizedNetworth = normalize(smoothNetworth)

    print("Plotting normalized output and normalized smooth output")
    nw = normalize(firms.output)
    sp, spTypes = findStationaryPoints(smoothNetworth)
    plt.plot(np.linspace(200,1000, len(nw)), nw)
    x2 = x2 + 200
    plt.plot(x2, normalizedNetworth)
    plt.scatter(x2[sp], normalizedNetworth[sp], c='r', zorder=3)
    plt.xticks(fontsize=14)
    plt.yticks([])
    plt.show()

    # calculate plt.xcorr
    corr = np.correlate(normalizedNetworth, normalizedChange, "full")
    corr /= np.sqrt(np.dot(normalizedNetworth, normalizedNetworth) * \
            np.dot(normalizedChange, normalizedChange))
    plt.plot(corr)
    plt.show()
    maxlags = 100
    Nx = len(normalizedNetworth)
    lags = np.arange(-maxlags, maxlags + 1)
    correls = corr[Nx - 1 - maxlags:Nx + maxlags]
    xspace = np.linspace(-maxlags/2, maxlags/2, len(correls))
    plt.plot(xspace, correls)
    plt.show()

    return

def tempAnalysis2():

    window = 10
    window2 = 50
    NL = []; NL2 = []
    for lag in range(0,100):
        NL.append(stats.pearsonr(firms.networth[0+lag:window+lag], economy.leverage[0+lag:window+lag]))
        NL2.append(stats.pearsonr(firms.networth[0:window2], economy.leverage[0+lag:window2+lag]))

    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(NL)
    ax[0].set_title(label="networth - leverage+ window=30")
    ax[0].grid(True)
    ax[1].plot(NL2)
    ax[1].set_title(label="networth - leverage+ window=50")
    ax[1].grid(True)
    plt.show()

# For a given simulation finds correlations of x with features
def findCorrelations(firms, x):
    features = [firms.output, firms.networth, firms.debt, \
            firms.profit, firms.capital, banks.networth, \
            banks.badDebt, banks.profit]

    # Use splines to smooth randomness of data
    smoothX = spline(x, \
        len(x)*np.var(x)*0.6, 3)[1]
    smoothX = normalize(smoothX)

    # Calculate correlations of leverage with other features
    corrArray = np.zeros((8,4799))
    j = 0
    for data in features:
        smoothData = spline(data, \
            len(data)*np.var(data)*0.17, 3)[1]
        smoothData = normalize(smoothData)
        corr = np.correlate(smoothData, smoothX, "full")
        corr /= np.sqrt(np.dot(smoothData, smoothData) * \
            np.dot(smoothX, smoothX))
        #corr = stats.pearsonr(smoothData, smoothX)[1]
        corrArray[j] = corr
        j += 1

    return corrArray

#
# Returns the indices where stationaryPoints occur in data and
# the type stationary point, -1 for maximum, 1 for minimum
#
def findStationaryPoints(data):
    ddata = np.gradient(data)
    stationaryPoints = np.array([], dtype=int)
    pointType = np.array([], dtype=int)
    diff = np.gradient(ddata) # data[i] - data[i-1]
    index = 0
    for i in ddata:
        if index == len(ddata)-1:
            continue
        if ddata[index]*ddata[index+1] < 0:
            pT = -1 if diff[index]<0 else 1
            stationaryPoints = np.append(stationaryPoints, index)
            pointType = np.append(pointType, pT)
        index += 1

    # remove saddle points
    # by seeing how close they are
    index = 0
    for p in stationaryPoints:
        if index != 0: # skip first
            pL = ((data[p] - data[previous]) / data[previous] ) * 100
            if pL > -0.1 and pL < 0.1: # reject small stationary Points less than 1 percent change
                 stationaryPoints = np.delete(stationaryPoints, index)
                 pointType = np.delete(pointType, index)
                 index -= 1
                 continue
        previous = p
        index += 1
    

    return stationaryPoints, pointType

def montecarlo():

    simFolders = "results/*_var015/"
    simulations = glob.glob(simFolders)
    if not simulations:
        print("No result files to analyze")
        return

    print("Analysing " + str(len(simulations)) + " results from " + str(simFolders))
    print("===========================")

    # Collect data of many simulations
    aggregateCorrelations = np.zeros((len(simulations), 8, 4799)) # correlation vectors of leverage vs 8 features
    aggregateCrises = np.zeros((len(simulations), 4)) # for each simulation stores number of crises and there size
    aggregateOutput = np.zeros((len(simulations), 800))
    change = np.zeros((len(simulations), 49))
    allCrisesLoss = np.array([])
    i = 0
    for folder in simulations:

        firms, banks, individualfirm, economy, parameters  = openSimulationFiles(folder)

        aggregateOutput[i] = firms.output

        quarterlyOutput = firms.output[np.arange(0, len(firms.output), len(firms.output)/50, dtype=int)]
        change[i] = (np.diff(quarterlyOutput) / quarterlyOutput[:-1]) * 100

        # Store correlation values in array
        aggregateCorrelations[i] = findCorrelations(firms, economy.leverage)

        # Find boom and busts of economy
        x, smoothOutput = spline(firms.output, \
                len(firms.output)*np.var(firms.output)*0.1, 3)

        sp, spType = findStationaryPoints(smoothOutput)
        crisesSize = np.zeros(len(sp))
        percentLoss = np.zeros(len(sp))
        index = 0
        for p in sp:
            if index != 0: # skip first
                cS = np.fabs(smoothOutput[p] - smoothOutput[previous])
                pL = ((smoothOutput[p] - smoothOutput[previous]) / smoothOutput[previous] ) * 100
                crisesSize[index] = cS
                percentLoss[index] = pL
            previous = p
            index += 1
        crisesSize = crisesSize[1:]
        percentLoss = percentLoss[1:]

        allCrisesLoss = np.append(allCrisesLoss, percentLoss[percentLoss < 0])
        if spType[0] < 0: # if first stationary point was maximum
            meanBoom = np.mean(crisesSize[::1])
            meanBust = np.mean(crisesSize[::2])
        else: # if first stationary point was minimum
            meanBoom = np.mean(crisesSize[::2])
            meanBust = np.mean(crisesSize[::1])
        aggregateCrises[i][0] = len(percentLoss[percentLoss < 0])
        aggregateCrises[i][1] = meanBust
        aggregateCrises[i][2] = np.mean(percentLoss[percentLoss < 0])
        aggregateCrises[i][3] = np.std(percentLoss)
        i += 1

    plt.hist(allCrisesLoss, bins=200) # shows the size of crises our findStationaryPoints is capturing
    plt.ylabel("Frequency", fontsize=14)
    plt.xlabel("% GDP change", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    meanCorr = np.mean(aggregateCorrelations, axis=0)

    meanOutput = np.median(aggregateOutput, axis=0)
    stdOutput = np.std(aggregateOutput, axis=0)
    plt.plot(meanOutput, c='b')
    plt.plot(meanOutput + stdOutput, dashes=[2,2], c='b')
    plt.plot(meanOutput - stdOutput, dashes=[2,2], c='b')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

    plt.hist(aggregateCrises[:,2], bins=40)
    plt.title("Distribution of Average of Crises in Simulation")
    plt.show()

    print("Max and Min correlations with leverage vs features")
    for i in meanCorr:
        print(np.amax(i), np.amin(i))
    print("")

    print("Average crises (busts) size")
    print(np.mean(aggregateCrises[:,0])) # average of simulations bust size
    print(np.std(aggregateCrises[:,0]))
    print("Average percentage GDP loss during crisis")
    print("Use this values to check against 2007Q4 and 2008Q1")
    print(np.mean(allCrisesLoss))
    print(np.std(allCrisesLoss))
    print(np.mean(aggregateCrises[:,2]))
    print(np.std(aggregateCrises[:,2])) # if this is low then our simulations are consistent in variation of crises
    plt.scatter(aggregateCrises[:,2], aggregateCrises[:,3])
    plt.show()
    print(np.where(aggregateCrises[:,2] < 1) and np.where(aggregateCrises[:,2] > 0.5))
    print("Average percentage GDP change")
    plt.hist(np.mean(change, axis=0), bins=40)
    plt.title("Distribution of average quarterly % change")
    plt.show()
    allChanges = np.reshape(change, (500*49))
    plt.hist(allChanges, bins=200)
    plt.title("Distribution of quarterly all % change")
    plt.show()
    print(np.mean(change) == np.mean(np.mean(change, axis=0)))
    print(np.mean(change) == np.mean(allChanges))
    print(np.std(change))
    print(np.amax(change), np.amin(change))

    # see if quarterly change of countries is comparable to each simulation
    oecd = validation.getAllOECD()
    testResults = 0
    n = len(oecd[0])
    m = len(change[0])
    criticalValue = 1.63*np.sqrt((n+m)/(n*m))
    for dataset in oecd:
        tests = []
        d = []
        for i in change:
            t = stats.ks_2samp(dataset, i) # uk quarterly %change compare with first sim
            if t[1] < 0.05 and t[0] > criticalValue:
                tests.append(t[0])
        testResults += len(tests)
        d.append(tests)
    print("KS-tests with OECD: ", testResults)
    print(np.mean(d))
   

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
        dy = np.gradient(interpolatedPoints[:,1])
        stationaryPoints = analyse.findStationaryPoints(dy)
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
            lines = np.array(list(reader)[201:], dtype=float) # ignore first 300 lines
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

    global firms, banks, individualfirm, economy, parameters
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
    if cmd[0] == "monte":
        montecarlo()
    if cmd[0] == "open":
        choice = selectResults(folders)
        if choice > len(folders) or choice < 0:
            return
        print("Opening results from " + folders[choice])
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
    if cmd[0] == "multiplot":
        multiplot(firms, banks)

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
    global firms, banks, individualfirm, economy, parameters
    folders = glob.glob("results/*/")
    folder = ""
    choice = 0
    while(True):
        try:
            if len(sys.argv) == 2:
                folder = sys.argv[1]
                if folder != "/":
                    folder += "/"
                break
            else:
                choice = selectResults(folders)
                if choice != -1:
                    break
        except EOFError:
            print("Exiting reader")
            sys.exit()

    if choice != 0:
        print("Opening results from " + folders[choice])
        print("Type 'help' for command options")
        firms, banks, individualfirm, economy, parameters = openSimulationFiles(folders[choice])
    else:
        print("Type 'help' for command options")
        firms, banks, individualfirm, economy, parameters = openSimulationFiles(folder)

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



