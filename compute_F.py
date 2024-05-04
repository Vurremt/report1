import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
colors = []
for line in tab_lines :
    line2 = line.split("/")
    point1 = line2[0].split(" ")
    point2 = line2[1].split(" ")
    points1.append([int(point1[1]), int(point1[3])])
    points2.append([int(point2[1]), int(point2[3])])
    colors.append("#"+line2[2])

if len(points1) != len(points2) :
    print("Error : not the same number of points for img 1 and img 2, exit...")
    exit()
print("Number of points : ", len(points1))

print(points1)
print(points2)

# Construction of A
A = np.zeros((len(points1), 9))
for i in range(len(points1)):
    x1, y1 = points1[i]
    x2, y2 = points2[i]
    A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

print("\nPrint A ?")
var = input("[y/n] : ")
if var == "y" : print(A)

# U, Σ, Vt
U, Σ, Vt = np.linalg.svd(A)

print("\nPrint U, Σ and V^T ?")
var = input("[y/n] : ")
if var == "y" :
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

print("\nF = ", F)

# Eigenvalues of F
eigenvalues = np.linalg.eigvals(F)
print("\n----------\nThe eigenvalues of F are : ", eigenvalues)

# Rank of F
rank = np.linalg.matrix_rank(F)
print("The rank of F is : ", rank, "\n----------\n")


# Open the image 1 and 2
img1 = Image.open("./img/same_plane/img1_circle.JPG")
img2 = Image.open("./img/same_plane/img2_circle.JPG")
width1, height1 = img1.size
width2, height2 = img2.size

fig = plt.figure(figsize=(10, 5))

# First image
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(img1)
ax1.set_xlim([0, width1])
ax1.set_ylim([0, height1])
ax1.invert_yaxis()

# Second image
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(img2)
ax2.set_xlim([0, width2])
ax2.set_ylim([0, height2])
ax2.invert_yaxis()


# Epipolar Lines

print("\nPrint epipolar lines equations ?")
var = input("[y/n] : ")
display = False
if var == "y" : display = True

print("\n --- Epipolar lines in image 1 and image 2 ---")
for i in range(len(points1)):
    a1 = F[0][0]*points2[i][0] + F[0][1]*points2[i][1] + F[0][2]
    b1 = F[1][0]*points2[i][0] + F[1][1]*points2[i][1] + F[1][2]
    c1 = F[2][0]*points2[i][0] + F[2][1]*points2[i][1] + F[2][2]
    if display : print("Img 1 : Point ", i, " : ", a1, "x + ", b1, "y + ", c1, " = 0")

    a2 = F[0][0]*points1[i][0] + F[1][0]*points1[i][1] + F[2][0]
    b2 = F[0][1]*points1[i][0] + F[1][1]*points1[i][1] + F[2][1]
    c2 = F[0][2]*points1[i][0] + F[1][2]*points1[i][1] + F[2][2]
    if display : print("Img 2 : Point ", i, " : ", a2, "x + ", b2, "y + ", c2, " = 0\n\n")
    
    x1 = np.linspace(0, width1, 4000)
    y1 = (-a1*x1 - c1) / b1
    ax1.plot(x1, y1, color=colors[i])

    x2 = np.linspace(0, width2, 4000)
    y2 = (-a2*x2 - c2) / b2
    ax2.plot(x2, y2, color=colors[i])

# Afficher les images
plt.show()
