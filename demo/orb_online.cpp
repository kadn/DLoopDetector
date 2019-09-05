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

#include "demoDetector_online.h"

using namespace DLoopDetector;
using namespace DBoW2;
using namespace DVision;
using namespace std;

// ----------------------------------------------------------------------------

static const char *VOC_FILE = "./resources/ORBvoc.txt";
static const char *IMAGE_DIR = "/media/kadn/DATA2/thunder/dataset/sequences/00/image_0";
static const char *POSE_FILE = "/media/kadn/DATA2/thunder/dataset/poses/00.txt";
static const int IMAGE_W = 1241; // image size
static const int IMAGE_H = 376;
// static const char *BRIEF_PATTERN_FILE = "./resources/brief_pattern.yml";

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

int main()
{
  // prepares the demo
  demoDetector<ORBVocabulary, ORBLoopDetector, FORB::TDescriptor> 
    demo(VOC_FILE, IMAGE_DIR, POSE_FILE, IMAGE_W, IMAGE_H);
  
  try 
  {
    // run the demo with the given functor to extract features
    ORBextractor extractor(500, 1.2f, 8, 20, 7);
    demo.run("ORB", extractor);
  }
  catch(const std::string &ex)
  {
    cout << "Error: " << ex << endl;
  }

  return 0;
}
