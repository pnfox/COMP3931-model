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
