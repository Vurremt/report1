import matplotlib.pyplot as plt
import numpy as np

# Creation of the setup of the images
def creation_of_img(img1, img2):
    fig = plt.figure(figsize=(10, 5))

    # First image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img1)
    ax1.set_xlim([0, img1.size[0]])
    ax1.set_ylim([0, img1.size[1]])
    ax1.invert_yaxis()

    # Second image
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img2)
    ax2.set_xlim([0, img2.size[0]])
    ax2.set_ylim([0, img2.size[1]])
    ax2.invert_yaxis()

    return ax1, ax2

# Calculate the equations of epipolar lines associated at point i between images 1 and 2
# It return two equations with the values a1, b1 c1, a2, b2, c2 like :
# In image 1 : the equation of the line is a1 x + b1 y + c1 = 0
# In image 2 : the equation of the line is a2 x + b2 y + c2 = 0
def calculate_1_epipolar_line(F, points1, points2, i, display):
    a1 = F[0][0]*points2[i][0] + F[0][1]*points2[i][1] + F[0][2]
    b1 = F[1][0]*points2[i][0] + F[1][1]*points2[i][1] + F[1][2]
    c1 = F[2][0]*points2[i][0] + F[2][1]*points2[i][1] + F[2][2]
    if display : print("Img 1 : Point ", i, " : ", a1, "x + ", b1, "y + ", c1, " = 0")

    a2 = F[0][0]*points1[i][0] + F[1][0]*points1[i][1] + F[2][0]
    b2 = F[0][1]*points1[i][0] + F[1][1]*points1[i][1] + F[2][1]
    c2 = F[0][2]*points1[i][0] + F[1][2]*points1[i][1] + F[2][2]
    if display : print("Img 2 : Point ", i, " : ", a2, "x + ", b2, "y + ", c2, " = 0\n\n")

    return a1, b1, c1, a2, b2, c2

# Draw epipolar lines in their associated image
def draw_1_epipolar_line(a1, b1, c1, a2, b2, c2, current_color, img1, img2, ax1, ax2):
    x1 = np.linspace(0, img1.size[0], img1.size[1])
    y1 = (-a1*x1 - c1) / b1
    ax1.plot(x1, y1, color=current_color)

    x2 = np.linspace(0, img2.size[0], img2.size[1])
    y2 = (-a2*x2 - c2) / b2
    ax2.plot(x2, y2, color=current_color)


# Each epipolar lines according to their associated points in images 1 and 2
def compute_epipolar_lines(F, points1, points2, colors, img1, img2, ax1, ax2):
    
    # Choose to show equations or not
    print("\nWould you want to print epipolar lines equations ?")
    var = input("[y/n] : ")
    display = False
    if var == "y" :
        display = True
        print("\n --- Epipolar lines in image 1 and image 2 ---")

    # Compute and draw all epipolar lines, one by one
    for i in range(len(points1)):

        # We obtain equations of both epipolar lines
        a1, b1, c1, a2, b2, c2 = calculate_1_epipolar_line(F, points1, points2, i, display)

        draw_1_epipolar_line(a1, b1, c1, a2, b2, c2, colors[i], img1, img2, ax1, ax2)

        
    
