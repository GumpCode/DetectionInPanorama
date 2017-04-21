#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

cv::Mat computeRotatMat(double yaw, double pitch, double roll);

std::vector<cv::Mat> makeRotatMatList(std::vector<double>& params);

int PanoImg2Warp(cv::Mat& srcPanoImg, cv::Mat& dstWarpImg, cv::Mat& rotatMat, double hFOV, int width, int height);

void makeWarpImgList(cv::Mat& srcPanoImg, std::vector<cv::Mat>& warpImgList, std::vector<cv::Mat>& rotatMats, const std::vector<int> size, double hFOV);

std::vector<cv::Mat> convertRotatMat2Inv(std::vector<cv::Mat> rotatMats);

std::pair<double, double> map2PanoCoord(double cols, double rows, std::vector<int> size, double F, double du, double dv, cv::Mat& rotatMatInv);

bool convertWarpCoord2Pano(std::vector<std::pair<double, double> >& boxesCoord, std::vector<std::vector<float> >& detections,
		const std::vector<int> size, float confidence_threshold, cv::Mat& rotatMat, double hFOV);

bool drawCoordInPanoImg(cv::Mat& panoImg, std::vector<std::pair<double, double> > boxesCoords);
