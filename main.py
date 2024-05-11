"""

©Copyright Code :
Chloé BRICE, INSA Lyon
Evahn LE GAL, ISIMA Clermont INP

Code co-written with Chloé BRICE, with the agreement of the professor, the report and the images will nevertheless be different for each student
Only the code was written and used as a team

Python 3.11.6

"""


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Name of current folder
folder_name = "./img/not_same_plane/"



### Question A-2 : Compute F ###

from A2_compute_F import load_points, construct_F, verif_F

# Name of list of points file
list_of_point_F = "list_of_points.txt"

print("\n - - - - - Question A-2 : Compute F - - - - -\n")

# Loading of points in images 1 and 2
points1, points2, colors = load_points(folder_name, list_of_point_F)

# Compute F with lists of points
F = construct_F(points1, points2, True)



### Question A-3 : Verification of the rank of F ###

print("\n - - - - - Question A-3 : Verification of F - - - - -")

verif_F(F)



### Question A-2 : Draw lines between corresponding points

from A2_linked_points import open_2_images, concate_img_1_2, save_and_show

#Name of image 1 loaded with circle
img1_circle = "img1_circle.jpg"
#Name of image 2 loaded with circle
img2_circle = "img2_circle.jpg"
#Name of linked image for 1 and 2 saved
linked_img = "linked_img.jpg"

print("\n - - - - - Question A-2 : Draw Corresponding Lines - - - - -\n")

# Open images
img1, img2 = open_2_images(folder_name, img1_circle, img2_circle)

# Generate an image with img1 and img2 and their linked lines
concate_img_1_2(img1, img2, points1, points2, colors)
# Show the image (and save the final result)
save_and_show(folder_name + "output/", linked_img)



### Question A-4 : Draw epipolar lines ###

from A4_epipolar_lines import creation_of_img, compute_epipolar_lines

#Name of epipolar lines image saved
epipolar_lines = "epipolar_lines.jpg"

print("\n - - - - - Question A-4 : Draw Epipolar Lines - - - - -\n")

# We create the setup for epipolar lines
ax1, ax2 = creation_of_img(img1, img2)

# We compute and draw each epipolar lines
compute_epipolar_lines(F, points1, points2, colors, img1, img2, ax1, ax2)
# Show the image (and save the final result)
save_and_show(folder_name + "output/", epipolar_lines)



### Question A-5 : Epipolar lines of points not computed in F ###

from A5_not_computed_points import compute_existing_epipolar_lines, not_computed_epipolar_lines

#Name of the list of not computed points
list_not_computed_points = "not_computed_points.txt"
#Name of image 1 loaded with features points not computed in the F matrix
img1_not_computed_points = "img1_not_computed_points.jpg"
#Name of image 2 loaded with features points not computed in the F matrix
img2_not_computed_points = "img2_not_computed_points.jpg"
#Name of image saved of epipolar lines of points not computed
img_epipoar_lines_not_computed = "epipoar_lines_not_computed.jpg"

print("\n - - - - - Question A-5 : Epipolar Lines of Not Computed Points - - - - -\n")

# Choose to calculate and draw not computed points
print("Would you like to take others feature points that are not included in the calculation of the F matrix ?")
var = input("[y/n] : ")
if var == "y" :

    img_not_computed_1 = Image.open(folder_name + "not_computed/" + img1_not_computed_points)
    img_not_computed_2 = Image.open(folder_name + "not_computed/" + img2_not_computed_points)

    # We create the setup for images with not computed points
    ax1_not_computed, ax2_not_computed = creation_of_img(img_not_computed_1, img_not_computed_2)

    # Draw lines of the 8 points used in the calculation of F (same color)
    compute_existing_epipolar_lines(F, points1, points2, "#1f00ff", img_not_computed_1, img_not_computed_2, ax1_not_computed, ax2_not_computed)


    # Load not computed points
    points_not_computed1, points_not_computed2, colors_not_computed = load_points(folder_name + "not_computed/", list_not_computed_points)

    # Draw lines of the points not used for F
    not_computed_epipolar_lines(F, points_not_computed1, points_not_computed2, colors_not_computed, img_not_computed_1, img_not_computed_2, ax1_not_computed, ax2_not_computed)    
    # Show the image (and save the final result)
    save_and_show(folder_name + "output/", img_epipoar_lines_not_computed)



### Question B-7 : Find intrinsic camera parameters with Kruppa equations ###

from B7_intrinsic_parameters import epipoles

print("\n - - - - - Question B-7 : Intrinsic Camera Parameters - - - - -\n")

print("Would you like to find intrinsic camera parameters with Kruppa equations ?")
var = input("[y/n] : ")
if var == "y" :

    # Find epipoles
    e1x, e2x = epipoles(F)

## We have not been able to find the rest of the calculations, which means that we cannot obtain K and the parameters
    


### Question A-6 : Algorithm with more than 8 points ###

print("\n - - - - - Question A-6 : Compute f with more than 8 points - - - - -\n")

# Choose to calculate and draw extra points
print("Would you like to take more than 8 points in the calculation of the F matrix ?\n(Make sure you already run the C++ program for generate in the folder extra_points/ images and associated .txt)")
var = input("[y/n] : ")
if var == "y" :

    # Name of list of points file
    list_extra_points = "extra_points.txt"
    #Name of image 1 loaded with extra points
    img1_extra_points = "img1_extra_points.jpg"
    #Name of image 2 loaded with extra points
    img2_extra_points = "img2_extra_points.jpg"
    #Name of epipolar lines image saved of extra points data
    extra_points = "extra_points.jpg"

    extra_points1, extra_points2, extra_colors = load_points(folder_name + "extra_points/", list_extra_points)
    F2 = construct_F(extra_points1, extra_points2, False)

    img1_extra = Image.open(folder_name + "extra_points/" + img1_extra_points)
    img2_extra = Image.open(folder_name + "extra_points/" + img2_extra_points)
    ax1_extra, ax2_extra = creation_of_img(img1_extra, img2_extra)

    compute_epipolar_lines(F2, extra_points1, extra_points2, extra_colors, img1_extra, img2_extra, ax1_extra, ax2_extra)
    save_and_show(folder_name + "output/", extra_points)


 
        
    
