/*** ** *

©Copyright Code :
Chloé BRICE, INSA Lyon
Evahn LE GAL, ISIMA Clermont INP

Code co-written with Chloé BRICE, with the agreement of the professor, the report and the images will nevertheless be different for each student
Only the code was written and used as a team

VS Code with OpenCV 4.9.0 and CMake 3.29.2

* ** ***/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

int main() {
    srand (time(NULL));
    // Load Images
    cv::Mat img1 = cv::imread("..\\..\\img\\not_same_plane\\img1.jpg", cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread("..\\..\\img\\not_same_plane\\img2.jpg", cv::IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error : impossible to load image." << std::endl;
        return 1;
    }

    // Ask number of points
    int numPoints;
    std::cout << "How many points do you want to extract ?\nAnswer : ";
    std::cin >> numPoints;

    int choose_algo;
    std::cout << "What algorithm do you want to use ?\n[1] ORB\n[2] AKAZE\nAnswer : ";
    std::cin >> choose_algo;

    // Extract features points
    cv::Ptr<cv::ORB> orb;
    cv::Ptr<cv::AKAZE> akaze;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    switch(choose_algo){
        case 1 :
            orb = cv::ORB::create(numPoints);
            orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
            orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
            break;
        case 2 :
            akaze = cv::AKAZE::create();
            akaze->setMaxPoints(numPoints);
            akaze->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
            akaze->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
            break;
        default :
            std::cerr << "Error : algorithm doesn't exist." << std::endl;
            return 1;
    }

    // Create a BFMatcher object with Hamming's distance
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    std::sort(matches.begin(), matches.end());

    // Filtering matches by distance
    float maxDistance = 100.0;
    std::cout << "Choose the threshold you want for your points (around 50 and 100 for a image 1200x2000)\nAnswer : ";
    std::cin >> maxDistance;
    std::vector<cv::DMatch> filteredMatches;
    for (const cv::DMatch& match : matches) {
        if (match.distance < maxDistance) {
            filteredMatches.push_back(match);
        }
    }
    std::cout << "Number of points remaining : " << filteredMatches.size() << std::endl;

    // Write features points in the file
    std::ofstream outFile("..\\..\\img\\not_same_plane\\extra_points\\extra_points.txt");
    for (const cv::DMatch& match : filteredMatches) {
        cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
        cv::Point2f pt2 = keypoints2[match.trainIdx].pt;
        cv::Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256); // Random Color
        std::stringstream colorHex;
        colorHex << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(color[2])
                 << std::setw(2) << static_cast<int>(color[1]) << std::setw(2) << static_cast<int>(color[0]);
        outFile << "img1[ " << (int)(pt1.x + 0.5) << " ; " << (int)(pt1.y + 0.5) << " ]/img2[ " << (int)(pt2.x + 0.5) << " ; " << (int)(pt2.y + 0.5) << " ]/";
        outFile << colorHex.str() << "/";
        outFile << std::endl;

        // Draw circle of associated features points
        cv::circle(img1, pt1, 30, color, 5);
        cv::circle(img2, pt2, 30, color, 5);
    }
    outFile.close();

    // Save images with highlighted points
    cv::imwrite("..\\..\\img\\not_same_plane\\extra_points\\img1_extra_points.jpg", img1);
    cv::imwrite("..\\..\\img\\not_same_plane\\extra_points\\img2_extra_points.jpg", img2);

    return 0;
}