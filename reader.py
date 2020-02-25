import sys
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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

def classify(data, firms):
    
    # Data classes:
    #   class 0: decreasing value
    #   class 1: increasing value
    #   class 2: little change
    classifiedData = np.array([])
    tol = 600

    previousValue = data[0]
    for i in data:
        if i == data[0]:
            continue
        if i < previousValue - tol:
            classifiedData = np.append(classifiedData, 0)
        elif i > previousValue + tol:
            classifiedData = np.append(classifiedData, 1)
        else:
            classifiedData = np.append(classifiedData, 2)
        previousValue = i

    classifier = SVC(kernel='linear')

    X = np.stack((firms.capital[1:], firms.price[1:], firms.networth[1:], \
            firms.debt[1:], firms.profit[1:])).transpose()

    #classifier.fit(X, classifiedData)
    time = np.linspace(1, len(classifiedData)+1, num=len(classifiedData))

    points = np.stack((time, firms.output[1:]), axis=-1)
    interpolatedPoints = analyse.splineData(points)
    dy = np.gradient(interpolatedPoints[:,1])
    stationaryPoints = analyse.findStationaryPoints(dy)

    # calculate information about stationaryPoints
    distances = np.array([])
    angles = np.array([])
    j = stationaryPoints[0]
    for i in stationaryPoints[1:]:
        x1 = interpolatedPoints[j,0]
        x2 = interpolatedPoints[i,0]
        y1 = interpolatedPoints[j,1]
        y2 = interpolatedPoints[i,1]
        dist = np.sqrt((x2-x1)**2+(y2-y1)**2)
        distances = np.append(distances, dist)
        angles = np.append(angles, np.arccos((x2-x1)/dist))
        j = i

    avgAngle = np.mean(angles)
    change = np.ceil(distances*(angles/avgAngle))
    change = np.append(change, 0)

    plt.hist(change)
    plt.show()

    # plot stationaryPoints
    plt.plot(time, firms.output[1:])
    plt.scatter(interpolatedPoints[stationaryPoints,0], \
            interpolatedPoints[stationaryPoints,1], c=change)
    plt.show()

    return classifiedData

def checkChange(data, data2):
    for i in range(1, len(data)):
        changeInData1 = data[i] - data[i-1]
        changeInData2 = data2[i] - data2[i-1]
        if changeInData1 * changeInData2 < 0:
            print("Change not off the same sign at " + str(i))
            print("Change in data1: " + str(changeInData1))
            print("Change in data2: " + str(changeInData2))

def openSimulationFile(folder):

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
    except FileNotFoundError:
        print("No file found")
        exit()

    return firms, banks, individualFirm

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
    if cmd[0] == "plot" or cmd[0] == "classify":

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
    firms, banks, individualFirm = openSimulationFile(folders[choice])

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



