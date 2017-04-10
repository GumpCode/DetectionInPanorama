#include "opencv2/opencv.hpp"

int PanoImg2Warp(cv::Mat& srcPanoImg, cv::Mat& dstWarpImg, int width, int height, double hFOV, double yaw, double pitch, double roll);
int WarpCoord2Pano(cv::Mat& srcPanoImg, cv::Mat& dstWarpImg, int width, int height, double hFOV, double yaw, double pitch, double roll);
