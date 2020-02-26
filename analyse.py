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
    index = 0
    for i in data:
        if index == len(data)-1:
            continue
        if data[index]*data[index+1] < 0:
            stationaryPoints = np.append(stationaryPoints, index)
        index += 1

    return stationaryPoints

#
# Returns a list of points where the GDP increased/decreased significantly
# These points are integer values calculated by
#           Euclidian distance to next point * (angle / mean angle)
# Large positive integers denote a large increase in GDP while large negative
# values denote a large decrease in GDP
#
def outputVolatility(firms):

    time = np.linspace(start=0, stop=len(firms.output), num=len(firms.output)-1)

    points = np.stack((time, firms.output[1:]), axis=-1)
    interpolatedPoints = splineData(points)
    dy = np.gradient(np.gradient(interpolatedPoints[:,1]))
    stationaryPoints = findStationaryPoints(dy)

    # calculate information about stationaryPoints
    distances = np.array([])
    angles = np.array([])
    j = stationaryPoints[0]
    for i in stationaryPoints[1:]:
        x1 = interpolatedPoints[j,0]
        x2 = interpolatedPoints[i,0]
        y1 = interpolatedPoints[j,1]
        y2 = interpolatedPoints[i,1]
        dist = np.sqrt((x2-x1)**2+(y2-y1)**2)
        distances = np.append(distances, dist)
        angles = np.append(angles, np.arcsin((y2-y1)/dist))
        j = i

    avgAngle = np.fabs(np.mean(angles))
    change = np.ceil(distances*(angles/avgAngle))
    change = np.append(change, 0)

    return change
