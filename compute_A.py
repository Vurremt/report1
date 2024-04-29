import numpy as np

# Data recovery
file = open("./img/same_plane/list_of_points.txt", "r")
tab_lines = []

line = file.readline()
while line :
    tab_lines.append(line)
    line = file.readline()

file.close()

# Points separation
points1 = []
points2 = []
for line in tab_lines :
    line2 = line.split("/")
    point1 = line2[0].split(" ")
    point2 = line2[1].split(" ")
    points1.append([int(point1[1]), int(point1[3])])
    points2.append([int(point2[1]), int(point2[3])])

print("Number of points : ", len(points1))

print(points1)
print(points2)

# Construction of A
A = np.zeros((len(points1), 9))
for i in range(len(points1)):
    x1, y1 = points1[i]
    x2, y2 = points2[i]
    A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

print(A)

# U, Σ, Vt
U, Σ, Vt = np.linalg.svd(A)
print("U = ", U)
print("Σ = ", Σ)
print("V^T = ", Vt)


# F1 is the last column of V (last line of Vt)
F1 = Vt[-1].reshape(3, 3)


U1, Σ1, V1t = np.linalg.svd(F1)

# Force rank 2 on F1 with minimum value of Σ1 set to 0
Σ1[-1] = 0
Σ1_diag = np.diag(Σ1)

# Reconstruction of F
F = np.dot(U1, np.dot(Σ1_diag, V1t))

print("F = ", F)


# Epipolar Lines
#In image 1 :
print(" --- Epipolar line in image 1 associated with (x', y') in image 2 ---")
for i in range(len(points2)):
    a = F[0][0]*points2[i][0] + F[0][1]*points2[i][1] + F[0][2]
    b = F[1][0]*points2[i][1] + F[1][1]*points2[i][1] + F[1][2]
    c = F[2][0]*points2[i][1] + F[2][1]*points2[i][1] + F[2][2]
    print("Point ", i, " : ", a, "x + ", b, "y + ", c, " = 0")

#In image 2 :
print(" --- Epipolar line in image 2 associated with (x, y) in image 1 ---")
for i in range(len(points2)):
    a = F[0][0]*points1[i][0] + F[1][0]*points1[i][1] + F[2][0]
    b = F[0][1]*points1[i][1] + F[1][1]*points1[i][1] + F[2][1]
    c = F[0][2]*points1[i][1] + F[1][2]*points1[i][1] + F[2][2]
    print("Point ", i, " : ", a, "x + ", b, "y + ", c, " = 0")


