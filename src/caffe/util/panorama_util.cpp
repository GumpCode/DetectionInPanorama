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
using namespace caffe;

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

int BoxesWarp(vector<double>& selectedBoxs, int width, int height, double hFOV, double yaw, double pitch, double roll, cv::Mat& panoImg, cv::Mat& partImg)
{
    int panoWidth = panoImg.cols;
    int panoHeight = panoImg.rows;
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

    ////////////////////////////////////////////////////////////

    Mat boxImg = cv::Mat::zeros(height, width, CV_8UC3);

    for (size_t i=0; i<selectedBoxs.size(); i+=4)
    {
        rectangle(boxImg, Point(selectedBoxs[i], selectedBoxs[i+1]),
                  Point(selectedBoxs[i+2], selectedBoxs[i+3]), Scalar(255, 0, 255), 2);
    }

    cv::Mat a = partImg.clone();
    for (size_t i=0; i<selectedBoxs.size(); i+=4)
    {
        rectangle(a, Point(selectedBoxs[i], selectedBoxs[i+1]),
                  Point(selectedBoxs[i+2], selectedBoxs[i+3]), Scalar(255, 0, 255), 2);
    }
    cv::imwrite("aaaaaa.jpg", a);



    double F  = (width/2)/tan(hFOV/2.f);
    double du = panoWidth/2/CV_PI;
    double dv = panoHeight/CV_PI;

    Mat tmpCoor = Mat::zeros(3, 1, CV_64F);
    Mat warpMap = cv::Mat(panoHeight, panoWidth, CV_32FC2);

    for(int j = 0; j < panoHeight; j++)
    {
        for(int i = 0; i < panoWidth; i++)
        {
            double latitude  = (panoHeight/2 - j)/dv;
            double longitude = (i - panoWidth/2)/du;

            tmpCoor.at<double>(0) = cos(latitude)*sin(longitude);
            tmpCoor.at<double>(1) = sin(latitude);
            tmpCoor.at<double>(2) = cos(latitude)*cos(longitude);

            tmpCoor = rotatMat.inv()*tmpCoor;

            if (tmpCoor.at<double>(2) < DBL_EPSILON)
            {
                warpMap.at<Point2f>(j, i) = Point2f(-1.f, -1.f);
            }
            else
            {
                float u = tmpCoor.at<double>(0)*F/tmpCoor.at<double>(2);
                float v = tmpCoor.at<double>(1)*F/tmpCoor.at<double>(2);

                warpMap.at<Point2f>(j, i) = Point2f(u+width/2, height/2-v);
            }
        }
    }

    cv::Mat dstPanoBox;
    dstPanoBox = cv::Mat(panoHeight, panoWidth, CV_8UC3, cv::Scalar(1, 1, 1));
    remap(boxImg, dstPanoBox, warpMap, cv::Mat(), CV_INTER_CUBIC);
    //addWeighted(dstPanoBox, 0.5, panoImg, 0.4, 0.0, panoImg);
    add(dstPanoBox, panoImg, panoImg);
    cv::imwrite("b.jpg", panoImg);
    cv::waitKey(0);

    return 0;
}

int convertWarpCoord2Pano(cv::Mat& panoImg, std::vector<std::vector<float> >& detections, std::vector<double>& param, int warpWidth, int warpHeight, float confidence_threshold, cv::Mat& im)
{
  cv::Mat result;
  std::vector<double> boxes;
  for(size_t i = 0; i < detections.size(); ++i){
    const vector<float>& detection = detections[i];  
    const float score = detection[2]; 
    if (score >= confidence_threshold){
      boxes.push_back(static_cast<double>(detection[3] * warpWidth)); 
      boxes.push_back(static_cast<double>(detection[4] * warpHeight)); 
      boxes.push_back(static_cast<double>(detection[5] * warpWidth)); 
      boxes.push_back(static_cast<double>(detection[6] * warpHeight)); 
    }
  }
  BoxesWarp(boxes, warpWidth, warpHeight, param[0], param[1], param[2], param[3], panoImg, im);
  return 0;
}

void MakeWarpImgList(cv::Mat& srcPanoImg, std::vector<cv::Mat>& warpImgList, std::vector<double>& params, int width, int height)
{
  size_t index = 0;
  std::string s;
  cv::Mat dst;
  while(index < params.size()){
    PanoImg2Warp(srcPanoImg, dst, width, height, params[index], params[index+1], params[index+2], params[index+3]);
    warpImgList.push_back(dst);
    index += 4;
  }
}
