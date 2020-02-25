import numpy as np
import matplotlib.pyplot as plt

#
# Returns the interpolation of data
#
def splineData(data):

    if len(data.shape) > 2:
        print("Spline Data: data must be 1d array")
        return None
   
    splineX = np.array([])
    splineY = np.array([])
    x = data[:,0]
    y = data[:,1]

    # maybe try data.reshape((int(len(data)/3, 3))
    d = 4 if len(y)%2 == 0 else 5
    pointsY = y.reshape((int(len(y)/d), d))
    pointsX = x.reshape((int(len(x)/d), d))
    for i in pointsY:
        splineY = np.append(splineY, np.mean(i))
    for i in pointsX:
        splineX = np.append(splineX, np.mean(i))
   
    spline = np.stack((splineX, splineY), axis=-1)

    return spline
#
# Returns the indices where stationaryPoints occur in data
#
def findStationaryPoints(data):
    stationaryPoints = np.array([], dtype=int)
    diff = np.fabs(np.diff(data))
    avgDiff = np.mean(diff)
    variance = np.var(diff)
    print("Average difference: ", avgDiff)
    print("Difference variance: ", variance)
    index = 0
    for i in data:
        if index == len(data)-1:
            continue
        diff = np.fabs(data[index] - data[index+1])
        #if data[index] < avgDiff and data[index] > -avgDiff:
        if data[index]*data[index+1] < 0:
        #if diff < 1.4*np.sqrt(variance)*avgDiff:
            stationaryPoints = np.append(stationaryPoints, index)
        index += 1

    return stationaryPoints

