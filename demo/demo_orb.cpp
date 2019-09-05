/**
 * File: demo_brief.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DLoopDetector
 */

#include <iostream>
#include <vector>
#include <string>
#include <iterator>
// DLoopDetector and DBoW2
#include "DBoW2.h" // defines BriefVocabulary
#include "DLoopDetector.h" // defines BriefLoopDetector
#include "DVision.h" // Brief 

#include <opencv/cv.h>
// #include <opencv/highgui.h>
// OpenCV
// #include <opencv/cv.h>
// #include <opencv/highgui.h>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/features2d/features2d.hpp>
// #include <opencv2/imgproc/imgproc.hpp>

//opencv3.1.0
//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/features2d.hpp>
//#include <opencv2/imgproc.hpp>

#include <opencv2/opencv.hpp>

#include "demoDetector_orb.h"

using namespace DLoopDetector;
using namespace DBoW2;
using namespace DVision;
using namespace std;

bool loadParams(std::string filename, string &DIR, string &POSE, int &IMAGE_W, int &IMAGE_H)
{
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  if(!fs.isOpened())
  {
    fprintf(stderr, "loadParams falied. check file please.\n");
        return false;
  }
  fs["IMAGE_DIR"] >> DIR;
  fs["POSE_FILE"] >> POSE;
  fs["IMAGE_W"] >> IMAGE_W;
  fs["IMAGE_H"] >> IMAGE_H;
  return true;
}

// ----------------------------------------------------------------------------

static const char *VOC_FILE = "./resources/ORBvoc.txt";
string IMAGE_DIR = "/media/kadn/DATA2/thunder/dataset/sequences/00/image_0";
string POSE_FILE = "/media/kadn/DATA2/thunder/dataset/poses/00.txt";
static int IMAGE_W = 1241; // image size
static int IMAGE_H = 376;
// static const char *BRIEF_PATTERN_FILE = "./resources/brief_pattern.yml";

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

int main()
{
  string paramsfile = "./config.yml";
  if(!loadParams(paramsfile, IMAGE_DIR, POSE_FILE, IMAGE_W, IMAGE_H))
  {
    return -1;
  }
  // prepares the demo
  demoDetector<ORBVocabulary, ORBLoopDetector, FORB::TDescriptor> 
    demo(VOC_FILE, IMAGE_DIR, POSE_FILE, IMAGE_W, IMAGE_H);
  
  try 
  {
    // run the demo with the given functor to extract features
    ORBextractor extractor(500, 1.2f, 8, 20, 7);
    // vector<cv::KeyPoint> keypoints;
    // cv::Mat descriptors;
    // cv::Mat img = cv::imread("./resources/images/SVS_L_1235603737.305831.png");
    // extractor(img,keypoints, descriptors);
    demo.run("ORB", extractor);
  }
  catch(const std::string &ex)
  {
    cout << "Error: " << ex << endl;
  }

  return 0;
}
