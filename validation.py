import csv

def getGDPValidationData():
    data = open("/home/pnfox/Downloads/tmp/namq_10_gdp_1_Data.csv")
    reader = csv.reader(data)
    lines = list(reader)

    ukGDP = []
    for l in lines:
        if ("United Kingdom" in l[1]) and ("Current" in l[2]) \
                and ("euro" in l[2]) and ("Unadjusted" in l[3]):
            if l[-2] == ":" or l[-2] == "":
                continue
            numbers = l[-2].split(",")
            gdp = ""
            for i in numbers:
                gdp += i
            ukGDP.append(float(gdp))

    return ukGDP
