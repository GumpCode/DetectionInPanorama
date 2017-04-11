#include "opencv2/opencv.hpp"
#include "caffe/util/panorama_util.hpp"

using namespace cv;

int PanoImg2Warp(Mat& srcPanoImg, Mat& dstWarpImg, int width, int height, double hFOV, double yaw, double pitch, double roll)
{
  if (fabs(pitch)>CV_PI/2)
  {
    return 1;
  }

  int srcWidth  = srcPanoImg.cols;
  int srcHeight = srcPanoImg.rows;

  if (srcWidth != srcHeight*2)
  {
    return 2;
  } 

  if (hFOV <= 0 || hFOV >= CV_PI)
  {
    return 3;
  }

  if (width <= 0 || width <= 0)
  {
    return 4;
  }

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

  Mat rotatMat = rotatMatYaw*rotatMatPitch*rotatMatRoll;

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

int WarpCoord2Pano(Mat& PanoImg, Mat& WarpImg, double yaw, double pitch, std::vector<double>& coords, std::vector<double>& result)
{
  if (fabs(pitch)>CV_PI/2)
  {
    return 1;
  }

  int panoWidth  = PanoImg.cols;
  int panoHeight = PanoImg.rows;
  int warpWidth = WarpImg.cols;
  int warpHeight = WarpImg.rows;

  if (panoWidth != panoHeight*2)
  {
    return 2;
  } 

  double panoWidthOffset = yaw/(2*CV_PI) * panoWidth;
  double panoHeightOffset = (CV_PI-pitch)/CV_PI * panoHeight;
  for(std::vector<double>::iterator coord = coords.begin(); coord != coords.end(); coord++){
    double coord_w = panoWidthOffset - warpWidth/2 + *coord;
    coord++;
    double coord_h = panoHeightOffset - warpHeight/2 + *coord;
    result.push_back(coord_w);
    result.push_back(coord_h);
  }
  
  
  return 0;
}

std::vector<std::pair<double, double> > MakeWarpImgList(cv::Mat& srcPanoImg, std::vector<cv::Mat>& warpImgList, int width, int height)
{
  cv::Mat dst;
  std::pair<double, double> param;
  std::vector<std::pair<double, double> > result;
  PanoImg2Warp(srcPanoImg, dst, width, height, CV_PI/2, 0, 0, 0);
  warpImgList.push_back(dst);
  param = std::make_pair(0, 0);
  result.push_back(param);
  PanoImg2Warp(srcPanoImg, dst, width, height, CV_PI/2, CV_PI/2, 0, 0);
  warpImgList.push_back(dst);
  param = std::make_pair(CV_PI/2, 0);
  result.push_back(param);
  PanoImg2Warp(srcPanoImg, dst, width, height, CV_PI/2, CV_PI, 0, 0);
  warpImgList.push_back(dst);
  param = std::make_pair(CV_PI, 0);
  result.push_back(param);
  PanoImg2Warp(srcPanoImg, dst, width, height, CV_PI/2, CV_PI*3/2, 0, 0);
  warpImgList.push_back(dst);
  param = std::make_pair(CV_PI*3/2, 0);
  result.push_back(param);
  PanoImg2Warp(srcPanoImg, dst, width, height, CV_PI/2, 0, CV_PI/2, 0);
  warpImgList.push_back(dst);
  param = std::make_pair(0, CV_PI/2);
  result.push_back(param);
  PanoImg2Warp(srcPanoImg, dst, width, height, CV_PI/2, 0, -(CV_PI/2), 0);
  warpImgList.push_back(dst);
  param = std::make_pair(0, -(CV_PI/2));
  result.push_back(param);

  return result;
}
