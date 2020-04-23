import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

def getGDPValidationData(fileName):
    try:
        data = open(fileName)
        reader = csv.reader(data)
        lines = list(reader)
    except FileNotFoundError:
        print("File not found")
        return

    ukGDP = []; spainGDP = []; franceGDP = []
    for l in lines:
        if ("Current" in l[2]) \
                and ("euro" in l[2]) and ("Unadjusted" in l[3]):
            if l[-2] == ":" or l[-2] == "":
                continue
            numbers = l[-2].split(",")
            gdp = ""
            for i in numbers:
                gdp += i
            if "United Kingdom" in l[1]:
                ukGDP.append(float(gdp))
            elif "France" in l[1]:
                franceGDP.append(float(gdp))
            elif "Spain" in l[1]:
                spainGDP.append(float(gdp))

    time = open("validation/GDP/namq_10_gdp_Label.csv")
    lines = list(time)

    timeStamps = []
    timeStampsFound = False
    for l in lines:
        if "TIME" in l:
            timeStampsFound = True
            continue
        if timeStampsFound:
            ts = l.split("\"")
            if len(ts) < 2:
                timeStampsFound = False
            else:
                timeStamps.append(ts[1])

    return timeStamps, ukGDP, franceGDP, spainGDP

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Please provide filename")
        sys.exit(0)

    fileName = sys.argv[1]
    time, uk, france, spain = getGDPValidationData(fileName)
    plt.plot(time, france)
    plt.xticks(rotation=90)
    plt.title("France GDP")
    plt.show()
    
    ukChange = []; spainChange = []; franceChange = []
    # measure average GDP ukChange
    for i in range(len(uk)):
        if i == 0:
            continue

        ukC = uk[i]/uk[i-1]
        spainC = spain[i]/spain[i-1]
        franceC = france[i]/france[i-1]
        ukChange.append(ukC)
        spainChange.append(spainC)
        franceChange.append(franceC)

    i = 0
    for change in [ukChange, spainChange, franceChange]:
        print("")
        if i == 0:
            print("======== UK ========")
        if i == 1:
            print("======== Spain =========")
        if i == 2:
            print("======== France =========")
        print("Mean and standard deviation GDP change")
        print(np.mean(change) - 1)
        print(np.std(change))
        print("Max and min GDP change")
        print(np.amax(change), time[np.where(change == np.amax(change))[0][0]])
        print(np.amin(change), time[np.where(change == np.amin(change))[0][0]])
        i += 1
