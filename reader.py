import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

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

