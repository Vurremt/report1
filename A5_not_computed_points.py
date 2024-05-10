import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


from A4_epipolar_lines import calculate_1_epipolar_line, draw_1_epipolar_line

# Draw epipolar lines with the same color
def compute_existing_epipolar_lines(F, points1, points2, color, img1, img2, ax1, ax2):

    # Compute and draw all epipolar lines, one by one
    for i in range(len(points1)):
        a1, b1, c1, a2, b2, c2 = calculate_1_epipolar_line(F, points1, points2, i, False)
        draw_1_epipolar_line(a1, b1, c1, a2, b2, c2, color, img1, img2, ax1, ax2)


# Compute and draw each epipolar lines of points not used for compute F
def not_computed_epipolar_lines(F, points1, points2, colors, img1, img2, ax1, ax2):
    
    # Compute and draw all epipolar lines, one by one
    for i in range(len(points1)):
        a1, b1, c1, a2, b2, c2 = calculate_1_epipolar_line(F, points1, points2, i, False)
        draw_1_epipolar_line(a1, b1, c1, a2, b2, c2, colors[i], img1, img2, ax1, ax2)
    
    
