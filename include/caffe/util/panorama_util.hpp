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

int dcmp(double x);

struct Point{  
  double x,y; 
  Point(){}  
  Point(double x,double y):x(x),y(y){}  
  bool operator==(Point&p){  
    return sig(x-p.x)==0&&sig(y-p.y)==0;  
  }  
};  

struct Detection{
  int imgId, id, label;
  float score;
  Point point[4];
  Detection(){}
  //Detection(int imgId, int id, int label, float score, Point p1,
  //  Point p2, Point p3, Point p4):
  //    imgId(imgId), id(id),label(label),score(score),point[0](p1.x,p1.y),
  //    p2(p2.x,p2.y),p3(p3.x,p3.y),point[1](p4.x,p4.y){}
};
      
struct aParams{
  int warpWidth, warpHeight, panoWidth, panoHeight;
  double F, du, dv;
  aParams(int warpWidth, int warpHeight, int panoWidth, int panoHeight,
    double F, double du, double dv):
      warpWidth(warpWidth), warpHeight(warpHeight), panoWidth(panoWidth),
      panoHeight(panoHeight), F(F), du(du), dv(dv){}
};

typedef Point Vector;

Vector operator-(Point A, Point B);

double Dot(Vector A, Vector B);

double cross(Vector A, Vector B);

double cross(Point o,Point a,Point b);

bool InSegment(Point P,Point a1,Point a2);

bool SegmentIntersection(Point a1,Point a2,Point b1,Point b2);

double area(Point* ps,int n);

int lineCross(Point a,Point b,Point c,Point d,Point&p);

void polygon_cut(Point*p,int&n,Point a,Point b);

double intersectArea(Point a,Point b,Point c,Point d);

double intersectArea(Point*ps1,int n1,Point*ps2,int n2); 

cv::Mat computeRotaMat(double yaw, double pitch, double roll);

std::vector<cv::Mat> makeRotaMatList(std::vector<double>& params);

int panoImg2Warp(cv::Mat& srcPanoImg, cv::Mat& dstWarpImg, cv::Mat& rotaMat, aParams params);

void makeWarpImgList(cv::Mat& srcPanoImg, std::vector<cv::Mat>& warpImgList, std::vector<cv::Mat>& rotaMats, aParams params);

std::pair<double, double> map2PanoCoord(double cols, double rows, aParams params, cv::Mat& rotaMat);

std::vector<cv::Mat> convertRotaMat2Inv(std::vector<cv::Mat> rotaMats);

bool convertWarpCoord2Pano(std::vector<std::pair<double, double> >& boxesCoord, std::vector<Detection>& detections, std::vector<int>& filtedIndexs, int& current, aParams params, cv::Mat& rotaMat);

bool drawCoordInPanoImg(cv::Mat& panoImg, std::vector<std::pair<double, double> > boxesCoords);

bool drawCoordInWarpImg(cv::Mat& warpImg, std::vector<Detection>& detections, double confidence_threshold);

std::pair<int, int> fixPointRange(int x, int y, int width, int height);

bool setCoord(Detection& d, Point* p, aParams params, std::vector<cv::Mat> rotaMat);

float computeOverlap(Detection d1, Detection d2, aParams& params, std::vector<cv::Mat>& rotaMat);

bool sortByScore(Detection a, Detection b);

bool applyNMS(std::vector<Detection >& mapDetections, std::vector<int>& filtedIndexs, aParams& params, std::vector<cv::Mat> rotaMat);

bool applyNMS4Detections(std::vector<std::vector<Detection > >& allDetections, std::vector<int>& filtedIndexs, aParams& params, float confidence_threshold, std::vector<cv::Mat> rotaMats);
