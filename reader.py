import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

resultNames = {0: "Output", 1: "Capital",
            2 : "Price", 3 : "Wealth",
            4 : "Debt", 5 : "Profit",
            6 : "Default"}

def plot(data, data2=None, data3=None, data4=None):
    fig, ax = plt.subplots()
    try:
        for i in [data, data2, data3, data4]:
            if i == None:
                continue
            if (len(i) != 2) or (type(i[0]) is not dict) \
                    or (type(i[1]) is not str):
                raise ValueError("Error plot: data must be tuple of the form (dict, str)")
            ax.plot(i[0].get(i[1]), label=i[1])
        ax.legend()
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

def openFiles(folder):

    try:
        with open(folder + "aggregateResults.csv", "r") as f:
            reader = csv.reader(f)
            lines = list(reader)
    except FileNotFoundError:
        print("No file found")
        exit()

    firms = {'Output':[], 'Capital':[],
            'Price':[], 'Wealth':[],
            'Debt':[], 'Profit':[],
            'Default':[]}
    banks = {'Wealth':[], 'Debt':[],
            'Profit':[], 'Default':[]}

    for l in lines:
        l = np.asarray(l, dtype=float)
        for i in range(7):
            keyword = resultNames.get(i)
            firms.get(keyword).append(float(l[i]))
        for i in range(7, 11):
            keyword = resultNames.get(i-4)
            banks.get(keyword).append(float(l[i]))

    return firms, banks

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
        choice = int(input())
    except ValueError:
        print("Invalid Input")
        exit()

print("Opening results from " + folders[choice])
firms, banks = openFiles(folders[choice])
