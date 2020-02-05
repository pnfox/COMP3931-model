import csv

def getGDPValidationData(fileName):
    try:
        data = open(fileName)
        reader = csv.reader(data)
        lines = list(reader)
    except IOError:
        raise Exception("Please give a valid validation csv file")

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

    return ukGDP, franceGDP, spainGDP
