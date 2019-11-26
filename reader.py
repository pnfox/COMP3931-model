import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

def plot(data, data2=None, data3=None, data4=None):
    fig, ax = plt.subplots()
    try:
        for i in [data, data2, data3, data4]:
            if i == None:
                continue
            if (len(i) != 2) or (type(i[0]) is not np.ndarray) \
                    or (type(i[1]) is not str):
                raise ValueError("Error plot: data must be tuple of the form (array, str)")
        fig1 = ax.plot(data[0], label=data[1])
        if not data2 == None:
            fig2 = ax.plot(data2[0], label=data2[1])
        if not data3 == None:
            fig3 = ax.plot(data3[0], label=data3[1])
        if not data4 == None:
            fig4 = ax.plot(data4[0], label=data4[1])
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

files = glob.glob("results/*.csv")
choice = 0
if len(files) == 0:
    print("No result files to read")
    print("Please run simulator first")
    exit()
if len(files) == 1:
    choice = 0
if len(files) > 1:
    print("Please choose a simulation run to analyse")
    index = 0
    for i in files:
        print("[" + str(index) + "]: " + i)
        index += 1
    try:
        choice = int(input())
    except ValueError:
        print("Invalid Input")
        exit()
try:
    with open(files[choice], "r") as f:
        reader = csv.reader(f)
        lines = list(reader)
except FileNotFoundError:
    print("No file found")
    exit()

firmOutput = np.array([], dtype=float)
firmCapital = np.array([], dtype=float)
firmWealth = np.array([], dtype=float)
firmDebt = np.array([], dtype=float)
firmProfit = np.array([], dtype=float)
firmPrice = np.array([], dtype=float)
firmDefault = np.array([], dtype=float)
bankWealth = np.array([], dtype=float)
bankProfit = np.array([], dtype=float)
bankDefault = np.array([], dtype=float)
for l in lines:
    l = np.asarray(l, dtype=float)
    firmOutput = np.append(firmOutput, l[0])
    firmCapital = np.append(firmCapital, l[1])
    firmWealth = np.append(firmWealth, l[2])
    firmDebt = np.append(firmDebt, l[3])
    firmProfit = np.append(firmProfit, l[4])
    firmPrice = np.append(firmPrice, l[5])
    firmDefault = np.append(firmDefault, l[6])
    bankWealth = np.append(bankWealth, l[7])
    bankProfit = np.append(bankProfit, l[7])
    bankDefault = np.append(bankDefault, l[9])

