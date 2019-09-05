/**
 * File: demoDetector.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DLoopDetector
 * License: see the LICENSE.txt file
 */

#ifndef __DEMO_DETECTOR__
#define __DEMO_DETECTOR__

#include <iostream>
#include <vector>
#include <string>
#include <time.h>
#include <opencv/cv.h>
// #include <opencv/highgui.h>
// OpenCV
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>

// //opencv3.1.0
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h>
#include "DLoopDetector.h"
#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h>
#include <DVision/DVision.h>
#include "ORBextractor.h"


using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

/// @param TVocabulary vocabulary class (e.g: BriefVocabulary)
/// @param TDetector detector class (e.g: BriefLoopDetector)
/// @param TDescriptor descriptor class (e.g: bitset for Brief)
template<class TVocabulary, class TDetector, class TDescriptor>
/// Class to run the demo 
class demoDetector
{
public:

  /**
   * @param vocfile vocabulary file to load
   * @param imagedir directory to read images from
   * @param posefile pose file
   * @param width image width
   * @param height image height
   */
  demoDetector(const std::string &vocfile, const std::string &imagedir,
    const std::string &posefile, int width, int height);
    
  ~demoDetector(){}

  /**
   * Runs the demo
   * @param name demo name
   * @param extractor functor to extract features
   */
  void run(const std::string &name, 
    ORBextractor extractor);

protected:

  /**
   * Reads the robot poses from a file
   * @param filename file
   * @param xs
   * @param ys
   */
  void readPoseFile(const char *filename, std::vector<double> &xs, 
    std::vector<double> &ys) const;

protected:

  std::string m_vocfile;
  std::string m_imagedir;
  std::string m_posefile;
  int m_width;
  int m_height;
  int mnlastloop;
  int mnloop;
  int mnthreshold;
};

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
demoDetector<TVocabulary, TDetector, TDescriptor>::demoDetector
  (const std::string &vocfile, const std::string &imagedir,
  const std::string &posefile, int width, int height)
  : m_vocfile(vocfile), m_imagedir(imagedir), m_posefile(posefile),
    m_width(width), m_height(height), mnlastloop(0), mnloop(0), mnthreshold(25)
{
}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
void demoDetector<TVocabulary, TDetector, TDescriptor>::run
  (const std::string &name, ORBextractor extractor)
{
  cout << "DLoopDetector Demo" << endl 
    << "Dorian Galvez-Lopez" << endl
    << "http://doriangalvez.com" << endl << endl;
  
  // Set loop detector parameters
  typename TDetector::Parameters params(m_height, m_width);
  
  // Parameters given by default are:
  // use nss = true
  // alpha = 0.3
  // k = 3
  // geom checking = GEOM_DI
  // di levels = 0
  
  // We are going to change these values individually:
  params.use_nss = true; // use normalized similarity score instead of raw score
  params.alpha = 0.3; // nss threshold
  params.k = 1; // a loop must be consistent with 1 previous matches
  params.geom_check = GEOM_DI; // use direct index for geometrical checking
  params.di_levels = 2; // use two direct index levels
  
  // To verify loops you can select one of the next geometrical checkings:
  // GEOM_EXHAUSTIVE: correspondence points are computed by comparing all
  //    the features between the two images.
  // GEOM_FLANN: as above, but the comparisons are done with a Flann structure,
  //    which makes them faster. However, creating the flann structure may
  //    be slow.
  // GEOM_DI: the direct index is used to select correspondence points between
  //    those features whose vocabulary node at a certain level is the same.
  //    The level at which the comparison is done is set by the parameter
  //    di_levels:
  //      di_levels = 0 -> features must belong to the same leaf (word).
  //         This is the fastest configuration and the most restrictive one.
  //      di_levels = l (l < L) -> node at level l starting from the leaves.
  //         The higher l, the slower the geometrical checking, but higher
  //         recall as well.
  //         Here, L stands for the depth levels of the vocabulary tree.
  //      di_levels = L -> the same as the exhaustive technique.
  // GEOM_NONE: no geometrical checking is done.
  //
  // In general, with a 10^6 vocabulary, GEOM_DI with 2 <= di_levels <= 4 
  // yields the best results in recall/time.
  // Check the T-RO paper for more information.
  //
  
  // Load the vocabulary to use
  cout << "Loading " << name << " vocabulary..." << endl;

  // clock_t tStart = clock();
  TVocabulary voc;
  

  bool res = voc.loadFromTextFile(m_vocfile);
  // TVocabulary voc(m_vocfile);
  cout << "loaded " << endl;
  // Initiate loop detector with the vocabulary 
  cout << "Processing sequence..." << endl;
  TDetector detector(voc, params);
  
  cout << "TDetector detector" << endl;
  // Process images
  vector<cv::KeyPoint> keys;
  // vector<TDescriptor> descriptors;
  cv::Mat descriptors;

  // load image filenames  
  vector<string> filenames = 
    DUtils::FileFunctions::Dir(m_imagedir.c_str(), ".png", true);
  
  // load robot poses
  vector<double> xs, ys;
  readPoseFile(m_posefile.c_str(), xs, ys);
  
  // we can allocate memory for the expected number of images
  //detector.allocate(filenames.size());
  detector.allocate(8000);
  
  // prepare visualization windows
  DUtilsCV::GUI::tWinHandler win = "Current image";
  DUtilsCV::GUI::tWinHandler winplot = "Trajectory";
  
  // cv::RNG rng(time(0));
  
  cv::Scalar randcolor = cv::Scalar(50,255,50);
  DUtilsCV::Drawing::Plot::Style normal_style(randcolor,4);
  //DUtilsCV::Drawing::Plot::Style normal_style(2); // thickness
  DUtilsCV::Drawing::Plot::Style loop_style('r', 20); // color, thickness
  DUtilsCV::Drawing::Plot::Style another('b', 3);
  
  DUtilsCV::Drawing::Plot implot(480, 640,
    - *std::max_element(xs.begin(), xs.end()),
    - *std::min_element(xs.begin(), xs.end()),
    *std::min_element(ys.begin(), ys.end()),
    *std::max_element(ys.begin(), ys.end()), 20);
  
  // prepare profiler to measure times
  DUtils::Profiler profiler;
  
  int count = 0;
  map<DBoW2::EntryId, double> entry2Time;

  cout << "started " << endl;
  // go
  for(unsigned int i = 0; i < filenames.size(); ++i)
  {
    // get image
    cv::Mat im = cv::imread(filenames[i].c_str(), 0); // grey scale
    
    // cout << "got image============== " << endl;

    // show image
    DUtilsCV::GUI::showImage(im, true, &win, 10);
    
    // cout << "show image============== " << endl;
    // get features
    profiler.profile("features");

    extractor(im, keys, descriptors);

    vector<cv::Mat> mdescriptors;
    for(int i = 0;i<descriptors.rows;i++)
    {
      mdescriptors.push_back(descriptors.row(i));
    }

    profiler.stop();
        
    // add image to the collection and check if there is some loop
    DetectionResult result;
    
    profiler.profile("detection");
    // detector.detectLoop(keys, descriptors, result);
    //double timestamp = this->mImageConverter.cv_ptr->header.stamp.toSec();
    double timestamp = time(NULL);
    detector.detectLoop(keys, mdescriptors, result, timestamp, entry2Time);
    profiler.stop();

    //--------------------single threshold---------------------
//    if(result.detection())
//    {
//      cout << "This is " << i << " image ..- Loop found with image " << result.match << "!" << endl;
//      ++count;
//    }
//
////  show trajectory
//    if(i > 0)
//    {
//      if(result.detection())
//        implot.line(-xs[i-1], ys[i-1], -xs[i], ys[i], loop_style);
//      else
//        implot.line(-xs[i-1], ys[i-1], -xs[i], ys[i], normal_style);
//
//      DUtilsCV::GUI::showImage(implot.getImage(), true, &winplot, 10);
//    }
      //--------------------single threshold end---------------------

  // --------------------double threshold---------------------
    if(result.detection())
    {
        mnloop = i;
        if(mnloop - mnlastloop > mnthreshold && entry2Time[result.query] - entry2Time[result.match] > 5)
        {
            cout << "This is " << i << " image ..- Loop found with image " << result.match << "!" << endl;
            implot.line(-xs[i-1], ys[i-1], -xs[i], ys[i], loop_style);
        }
        else
        {
            //implot.line(-xs[i-1], ys[i-1], -xs[i], ys[i], another);
            implot.line(-xs[i-1], ys[i-1], -xs[i], ys[i], normal_style);
        }
        mnlastloop = mnloop;
    } else if(i>0){
            implot.line(-xs[i-1], ys[i-1], -xs[i], ys[i], normal_style);
    }
      // --------------------double threshold end---------------------

    DUtilsCV::GUI::showImage(implot.getImage(), true, &winplot, 10);
  }




  //------------------------after loop------------------------
  if(count == 0)
  {
    cout << "No loops found in this image sequence" << endl;
  }
  else
  {
    cout << count << " loops found in this image sequence!" << endl;
  } 

  cout << endl << "Execution time:" << endl
    << " - Feature computation: " << profiler.getMeanTime("features") * 1e3
    << " ms/image" << endl
    << " - Loop detection: " << profiler.getMeanTime("detection") * 1e3
    << " ms/image" << endl;

  cout << endl << "Press a key to finish..." << endl;
  DUtilsCV::GUI::showImage(implot.getImage(), true, &winplot, 0);
}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
void demoDetector<TVocabulary, TDetector, TDescriptor>::readPoseFile
  (const char *filename, std::vector<double> &xs, std::vector<double> &ys)
  const
{
  xs.clear();
  ys.clear();
  
  fstream f(filename, ios::in);
  
  string s;
  double x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12;
  while(!f.eof())
  {
    getline(f, s);
    if(!f.eof() && !s.empty())
    {
      sscanf(s.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &x1, &x2, &x3, &x4, &x5, &x6, &x7, &x8, &x9, &x10, &x11, &x12);
      // xs.push_back(x4/10.0+30.0);
      // ys.push_back(x12/10.0+10.0);
      xs.push_back(x4/5.0+15.0);
      ys.push_back(x12/5.0+5.0);
    }
  }
  cout << "xs----size"<< xs.size() << endl;
  
  f.close();
}

// ---------------------------------------------------------------------------

#endif

