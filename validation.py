import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

def getEuroStatData():
    try:
        data = open("validation/GDP/Eurostat/namq_10_gdp_1_Data.csv")
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

    time = open("validation/GDP/Eurostat/namq_10_gdp_Label.csv")
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

def getOECDData():
    try:
        data = open("validation/GDP/OECD/DP_LIVE_23042020223727726.csv")
        reader = csv.reader(data)
        lines = list(reader)
    except FileNotFoundError:
        print("File not found")
        return

    timeStamps = []
    ukChange = []; spainChange = []; usaChange = []; germanyChange = []
    for l in lines:
        if "GBR" == l[0]:
            time = l[-3]
            timeStamps.append(time)
            c = float(l[-2])
            ukChange.append(c)
        if "ESP" == l[0]:
            c = float(l[-2])
            spainChange.append(c)
        if "USA" == l[0]:
            c = float(l[-2])
            usaChange.append(c)
        if "DEU" == l[0]: # germany
            c = float(l[-2])
            germanyChange.append(c)

    return timeStamps, ukChange, spainChange, usaChange, germanyChange

if __name__=="__main__":

    time, uk, france, spain = getEuroStatData()
    ukChange = []; spainChange = []; franceChange = []; usaChange = []; germanyChange = []
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

    time, ukChange, spainChange, usaChange, germanyChange = getOECDData()
    print("")
    print("===== OECD Dataset =====")
    print("Uk change: ", np.mean(ukChange))
    print("Spain change: ", np.mean(spainChange))
    print("USA change: ", np.mean(usaChange))
