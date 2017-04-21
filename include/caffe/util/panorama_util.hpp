#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

cv::Mat computeRotaMat(double yaw, double pitch, double roll);

std::vector<cv::Mat> makeRotaMatList(std::vector<double>& params);

int PanoImg2Warp(cv::Mat& srcPanoImg, cv::Mat& dstWarpImg, cv::Mat& rotaMat, double hFOV, int width, int height);

void makeWarpImgList(cv::Mat& srcPanoImg, std::vector<cv::Mat>& warpImgList, std::vector<cv::Mat>& rotaMats, const std::vector<int> size, double hFOV);

std::vector<cv::Mat> convertRotaMat2Inv(std::vector<cv::Mat> rotaMats);

std::pair<double, double> map2PanoCoord(double cols, double rows, std::vector<int> size, double F, double du, double dv, cv::Mat& rotaMatInv);

bool convertWarpCoord2Pano(std::vector<std::pair<double, double> >& boxesCoord, std::vector<std::vector<float> >& detections,
		const std::vector<int> size, float confidence_threshold, cv::Mat& rotaMat, double hFOV);

bool drawCoordInPanoImg(cv::Mat& panoImg, std::vector<std::pair<double, double> > boxesCoords);

bool drawCoordInWarpImg(cv::Mat& warpImg, std::vector<std::vector<float> >& detections, double confidence_threshold);
bool fixPointRange(int& x, int& y, int width, int height);
