import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
import agents

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

def checkChange(data, data2):
    for i in range(1, len(data)):
        changeInData1 = data[i] - data[i-1]
        changeInData2 = data2[i] - data2[i-1]
        if changeInData1 * changeInData2 < 0:
            print("Change not off the same sign at " + str(i))
            print("Change in data1: " + str(changeInData1))
            print("Change in data2: " + str(changeInData2))

def getObject(A, time):
    if not type(A) == dict:
        print("Error printObject: expected type(A) as dict")
        return
    if not type(time) == int:
        print("Error printObject: expected type(time) as int")
        return

    print("Printing object at time ", time)
    index = 0
    for k in A.keys():
        print(k + ": " + str(A.get(k)[time]))
        index += 1

def getObjectValue(A, time, value):
    if not type(A) == dict:
        print("Error printObject: expected type(A) as dict")
        return
    if not type(time) == int:
        print("Error printObject: expected type(time) as int")
        return

    print(A.get(value)[time])

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
            lines = np.array(list(reader), dtype=float)
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
            lines = np.array(list(reader), dtype=float)
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

folders = glob.glob("results/*/")
choice = 0
if len(folders) == 0:
    print("No result files to read")
    print("Please run simulator first")
    exit()
if len(folders) == 1:
    choice = 0
if len(folders) > 1:
    print("Please choose a simulation run to analyse")
    index = 0
    for i in folders:
        print("[" + str(index) + "]: " + i)
        index += 1
    try:
        choice = int(input(">>> "))
        if choice >= len(folders):
            print("Please give valid choice")
    except ValueError:
        print("Invalid Input")
        exit()

print("Opening results from " + folders[choice])
firms, banks, individualFirm = openSimulationFile(folders[choice])

print("Where individualFirm went bankrupt")
print(np.where(individualFirm.default == 1)[0])
plot(firms.output)
plot(individualFirm.output)
