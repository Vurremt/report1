import numpy as np

# Load lists of points in images 1 and 2 and return them in arrays with their associated colors
def load_points(path, list_of_point):
    # Data recovery
    file = open(path + list_of_point, "r")
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
    print("Number of points loaded : ", len(points1))

    print(points1)
    print(points2)
    return points1, points2, colors

# Compute A, then U, Σ, Vt, F' and finally reconstruct F with lists of points 1 and 2
def construct_F(points1, points2, asking):
    # Construction of A
    A = np.zeros((len(points1), 9))
    for i in range(len(points1)):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

    if asking :
        print("\nPrint A ?")
        var = input("[y/n] : ")
        if var == "y" : print(A)

    # U, Σ, Vt
    U, Σ, Vt = np.linalg.svd(A)

    if asking :
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
    return F

# Verification of the rank of F
def verif_F(F):
    # Eigenvalues of F
    eigenvalues = np.linalg.eigvals(F)
    print("\n----------\nThe eigenvalues of F are : ", eigenvalues)

    # Rank of F
    rank = np.linalg.matrix_rank(F)
    print("The rank of F is : ", rank, "\n----------\n")

