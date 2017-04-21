#include <caffe/caffe.hpp>
#include "opencv2/opencv.hpp"
#include "caffe/util/panorama_util.hpp"

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
                      
using namespace cv;
using namespace std;

cv::Mat computeRotatMat(double yaw, double pitch, double roll)
{
  cv::Mat rotatMat;
  Mat rotatMatPitch = cv::Mat(3, 3, CV_64F);
  rotatMatPitch.at<double>(0, 0) = 1;
  rotatMatPitch.at<double>(0, 1) = 0;
  rotatMatPitch.at<double>(0, 2) = 0;
  rotatMatPitch.at<double>(1, 0) = 0;
  rotatMatPitch.at<double>(1, 1) = cos(-pitch);
  rotatMatPitch.at<double>(1, 2) = -sin(-pitch);
  rotatMatPitch.at<double>(2, 0) = 0;
  rotatMatPitch.at<double>(2, 1) = sin(-pitch);
  rotatMatPitch.at<double>(2, 2) = cos(-pitch);

  Mat rotatMatYaw = cv::Mat(3, 3, CV_64F);
  rotatMatYaw.at<double>(0, 0) = cos(yaw);
  rotatMatYaw.at<double>(0, 1) = 0;
  rotatMatYaw.at<double>(0, 2) = sin(yaw);
  rotatMatYaw.at<double>(1, 0) = 0;
  rotatMatYaw.at<double>(1, 1) = 1;
  rotatMatYaw.at<double>(1, 2) = 0;
  rotatMatYaw.at<double>(2, 0) = -sin(yaw);
  rotatMatYaw.at<double>(2, 1) = 0;
  rotatMatYaw.at<double>(2, 2) = cos(yaw);

  Mat rotatMatRoll = cv::Mat(3, 3, CV_64F);
  rotatMatRoll.at<double>(0, 0) = cos(roll);
  rotatMatRoll.at<double>(0, 1) = -sin(roll);
  rotatMatRoll.at<double>(0, 2) = 0;
  rotatMatRoll.at<double>(1, 0) = sin(roll);
  rotatMatRoll.at<double>(1, 1) = cos(roll);
  rotatMatRoll.at<double>(1, 2) = 0;
  rotatMatRoll.at<double>(2, 0) = 0;
  rotatMatRoll.at<double>(2, 1) = 0;
  rotatMatRoll.at<double>(2, 2) = 1;

  rotatMat = rotatMatYaw*rotatMatPitch*rotatMatRoll;
  
  return rotatMat;
}


std::vector<cv::Mat> makeRotatMatList(std::vector<double>& params)
{
  std::vector<cv::Mat> rotatMats;
  for(size_t i = 0; i < params.size(); i += 3)
  {
	cv::Mat rotatMat = computeRotatMat(params[i], params[i+1], params[i+2]);
    rotatMats.push_back(rotatMat);
  }
  return rotatMats;
}


int PanoImg2Warp(cv::Mat& srcPanoImg, cv::Mat& dstWarpImg, cv::Mat& rotatMat, double hFOV, int width, int height)
{
  int srcWidth  = srcPanoImg.cols;
  int srcHeight = srcPanoImg.rows;

  if (srcWidth != srcHeight*2)
  {
    return 2;
  } 

  if (width <= 0 || height <= 0)
  {
    return 4;
  }

  double F  = (width/2)/tan(hFOV/2.f);
  double du = srcWidth/2/CV_PI;
  double dv = srcHeight/CV_PI;

  Mat tmpCoor = Mat::zeros(3, 1, CV_64F);
  Mat warpMap = cv::Mat(height, width, CV_32FC2);

  for(int j = 0; j < height; j++)
  {
    for(int i = 0; i < width; i++)
    {
      tmpCoor.at<double>(0) = (double)(i-width/2);
      tmpCoor.at<double>(1) = (double)(height/2-j);
      tmpCoor.at<double>(2) = F;

      tmpCoor = rotatMat*tmpCoor;
      normalize(tmpCoor, tmpCoor);

      double latitude  = asin(tmpCoor.at<double>(1));
      double longitude = atan2(tmpCoor.at<double>(0), tmpCoor.at<double>(2));

      float u = longitude*du + srcWidth/2;
      float v = srcHeight/2 - latitude*dv;

      warpMap.at<Point2f>(j, i) = Point2f(u, v);
    }
  }

  dstWarpImg = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  remap(srcPanoImg, dstWarpImg, warpMap, cv::Mat(), CV_INTER_CUBIC, BORDER_WRAP);

  return 0;
}

void makeWarpImgList(cv::Mat& srcPanoImg, std::vector<cv::Mat>& warpImgList, std::vector<cv::Mat>& rotatMats, const std::vector<int> size, double hFOV)
{
  int width = size[0];
  int height = size[1];
  size_t index = 0;
  cv::Mat dst;
  while(index < rotatMats.size()){
    PanoImg2Warp(srcPanoImg, dst, rotatMats[index++], hFOV, width, height);
    warpImgList.push_back(dst);
  }
}

std::vector<cv::Mat> convertRotatMat2Inv(std::vector<cv::Mat> rotatMats)
{
  std::vector<cv::Mat> rotatMatsInv;
  for(size_t i = 0; i < rotatMats.size(); i++)
  {
    rotatMatsInv.push_back(rotatMats[i].inv());
  }
  return rotatMatsInv;
}

std::pair<double, double> map2PanoCoord(double cols, double rows, std::vector<int> size, double F, double du, double dv, cv::Mat& rotatMat)
{
  int warpWidth = size[0];
  int warpHeight = size[1];
  int panoWidth = size[2];
  int panoHeight = size[3];
  Mat tmpCoor = Mat::zeros(3, 1, CV_64F);

  tmpCoor.at<double>(0) = static_cast<double>(rows - warpWidth/2);
  tmpCoor.at<double>(1) = static_cast<double>(warpHeight/2 - cols);
  tmpCoor.at<double>(2) = F;

  tmpCoor = rotatMat*tmpCoor;
  normalize(tmpCoor, tmpCoor);

  double latitude  = asin(tmpCoor.at<double>(1));
  double longitude = atan2(tmpCoor.at<double>(0), tmpCoor.at<double>(2));

  double pCols = longitude*du + panoWidth/2;
  double pRows = panoHeight/2 - latitude*dv;
  std::pair<double, double> coord = std::make_pair(pRows, pCols);
  //std::pair<double, double> coord = std::make_pair(pCols, pRows);

  return coord;
}


bool convertWarpCoord2Pano(std::vector<std::pair<double, double> >& boxesCoord, std::vector<std::vector<float> >& detections,
		const std::vector<int> size, float confidence_threshold, cv::Mat& rotatMat, double hFOV)
{
  int warpWidth = size[0];
  int warpHeight = size[1];
  int panoWidth = size[2];
  int panoHeight = size[3];
  double F  = (warpWidth/2)/tan(hFOV/2.f);
  double du = panoWidth/2/CV_PI;
  double dv = panoHeight/CV_PI;
  for(size_t i = 0; i < detections.size(); ++i){
    const vector<float>& detection = detections[i];  
    const float score = detection[2]; 
    if (score >= confidence_threshold){
      double xmin = static_cast<double>(detection[3] * warpWidth);
      double ymin = static_cast<double>(detection[4] * warpHeight); 
      double xmax = static_cast<double>(detection[5] * warpWidth); 
      double ymax = static_cast<double>(detection[6] * warpHeight); 
      //[xmin, ymin, xmax, ymax]
      for(double rows = xmin; rows < xmax; rows += 1){
        double cols = ymin;
        std::pair<double, double> coord1 = map2PanoCoord(cols, rows, size, F, du, dv, rotatMat);
        boxesCoord.push_back(coord1);

        cols = ymax;
        std::pair<double, double> coord2 = map2PanoCoord(cols, rows, size, F, du, dv, rotatMat);
        boxesCoord.push_back(coord2);
      }
      for(double cols = ymin; cols < ymax; cols += 1){
        double rows = xmin;
        std::pair<double, double> coord1 = map2PanoCoord(cols, rows, size, F, du, dv, rotatMat);
        boxesCoord.push_back(coord1);

        rows = xmax;
        std::pair<double, double> coord2 = map2PanoCoord(cols, rows, size, F, du, dv, rotatMat);
        boxesCoord.push_back(coord2);
      }
    }
  }
  return true;
}

bool drawCoordInPanoImg(cv::Mat& panoImg, std::vector<std::pair<double, double> > boxesCoords)
{
  std::cout << "size is " << boxesCoords.size() << std::endl;
  for(int index = 0; index < boxesCoords.size(); index++)
  {
    int row = static_cast<int>(boxesCoords[index].first);
    int col = static_cast<int>(boxesCoords[index].second);
    cv::Point point;
    point.x = col;
    point.y = row;
    cv::circle(panoImg, point, 1, cv::Scalar(0,0,255));
  }

  return true;
}
