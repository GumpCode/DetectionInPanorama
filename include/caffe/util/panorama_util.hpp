#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int PanoImg2Warp(cv::Mat& srcPanoImg, cv::Mat& dstWarpImg, int width, int height, double hFOV, double yaw, double pitch, double roll);
void MakeWarpImgList(cv::Mat& srcPanoImg, std::vector<cv::Mat>& warpImgList, std::vector<double>& params, int width, int height);
int BoxesWarp(vector<double>& selectedBoxs, int width, int height, double hFOV, double yaw, double pitch, double roll, cv::Mat& panoImg);
int convertWarpCoord2Pano(cv::Mat& panoImg, std::vector<std::vector<float> >& detections, std::vector<double>& params, int warpWidth, int warpHeight, float confidence_threshold);
