import numpy as np

file = open("./img/same_plane/list_of_points.txt", "r")
tab_lines = []

line = file.readline()
while line :
    tab_lines.append(line)
    line = file.readline()

file.close()

points1 = []
points2 = []
for line in tab_lines :
    line2 = line.split("/")
    point1 = line2[0].split(" ")
    point2 = line2[1].split(" ")
    points1.append([int(point1[1]), int(point1[3])])
    points2.append([int(point2[1]), int(point2[3])])

print("Number of points : " + str(len(points1)))

print(points1)
print(points2)

A = np.zeros((len(points1), 9))
for i in range(len(points1)):
    x1, y1 = points1[i]
    x2, y2 = points2[i]
    A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

print(A)


