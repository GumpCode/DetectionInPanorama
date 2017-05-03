#include "opencv2/opencv.hpp"
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cstdio>  
#include <iostream>  
#include <cstring>  
#include <cmath>  

int sig(double d);

struct Point{  
    double x,y; Point(){}  
    Point(double x,double y):x(x),y(y){}  
    bool operator==(const Point&p)const{  
        return sig(x-p.x)==0&&sig(y-p.y)==0;  
    }  
};  

double cross(Point o,Point a,Point b);

double area(Point* ps,int n);

int lineCross(Point a,Point b,Point c,Point d,Point&p);

void polygon_cut(Point*p,int&n,Point a,Point b);

double intersectArea(Point a,Point b,Point c,Point d);

double intersectArea(Point*ps1,int n1,Point*ps2,int n2); 

cv::Mat computeRotaMat(double yaw, double pitch, double roll);

std::vector<cv::Mat> makeRotaMatList(std::vector<double>& params);

int panoImg2Warp(cv::Mat& srcPanoImg, cv::Mat& dstWarpImg, cv::Mat& rotaMat, double hFOV, int width, int height);

void makeWarpImgList(cv::Mat& srcPanoImg, std::vector<cv::Mat>& warpImgList, std::vector<cv::Mat>& rotaMats, const std::vector<int> size, double hFOV);

std::vector<cv::Mat> convertRotaMat2Inv(std::vector<cv::Mat> rotaMats);

std::pair<double, double> map2PanoCoord(double cols, double rows, std::vector<int> size, double F, double du, double dv, cv::Mat& rotaMat);

bool convertWarpCoord2Pano(std::vector<std::pair<double, double> >& boxesCoord, std::vector<std::vector<float> >& detections,
		const std::vector<int> size, float confidence_threshold, cv::Mat& rotaMat, double hFOV);

bool convertWarpCoord2Pano2(std::vector<std::pair<double, double> >& boxesCoord, std::vector<std::vector<float> >& detections, std::vector<int>& filtedIndexs, int& current, 
		const std::vector<int> size, cv::Mat& rotaMat, double hFOV);

bool drawCoordInPanoImg(cv::Mat& panoImg, std::vector<std::pair<double, double> > boxesCoords);

bool drawCoordInWarpImg(cv::Mat& warpImg, std::vector<std::vector<float> >& detections, double confidence_threshold);

bool fixPointRange(int& x, int& y, int width, int height);

bool setCoord(std::vector<float>& d, Point* p);

float computeOverlap(std::vector<float> d1, std::vector<float> d2);

bool compareFunc(std::vector<float> a, std::vector<float> b);

bool applyNMS(std::vector<std::vector<float> >& mapDetections, std::vector<int>& filtedIndexs);

//std::vector<int> applyNMS4Detections(std::vector<std::vector<std::vector<float> > >& allDetections, std::vector<int> size, float confidence_threshold, std::vector<cv::Mat> rotaMats, double hFOV);
bool applyNMS4Detections(std::vector<std::vector<std::vector<float> > >& allDetections, std::vector<int>& filtedIndexs, std::vector<int> size, float confidence_threshold, std::vector<cv::Mat> rotaMats, double hFOV);

//void flip(std::vector<float>& coord, std::vector<int> size);
//
//bool map2OtherView(std::vector<std::vector<float > >&mapCoords, std::vector<std::vector<float> >& coords, std::vector<cv::Mat>& rotaMats, int srcNum, int dstNum, float confidence_threshold, std::vector<int> size, double hFOV);
//
//float computCrossArea(std::vector<float>& mapCoord1, std::vector<float>& mapCoord2);
//
//bool applyNMS(std::vector<std::vector<float> >& mapCoords, std::vector<std::vector<float> >& detections);
//
//bool applyNMS4Detections(std::vector<std::vector<std::vector<float> > >& allDetections, std::vector<int> size, float confidence_threshold, std::vector<cv::Mat >& rotaMats, double hFOV, double angleOffset);
