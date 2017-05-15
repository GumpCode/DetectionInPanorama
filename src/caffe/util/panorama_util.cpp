#include "caffe/util/panorama_util.hpp"

#define maxn 510  
const double eps=1E-8;  
                      

/* 
    类型：多边形相交面积模板 
*/  
int sig(double d){  
  return(d>eps)-(d<-eps);
}

int dcmp(double x)
{
  if(fabs(x) < eps) return 0;
  return x < 0 ? -1:1;
}

Vector operator-(Point A, Point B)
{
  return Vector(A.x - B.x, A.y - B.y);
}

double Dot(Vector A, Vector B)
{
  return A.x*B.x + A.y*B.y;
}

double cross(Vector A, Vector B)
{
  return A.x*B.y - A.y*B.x;
}

double cross(Point o,Point a,Point b){  
  return(a.x-o.x)*(b.y-o.y)-(b.x-o.x)*(a.y-o.y);  
}

bool InSegment(Point P,Point a1,Point a2)  
{  
  return dcmp(cross(a1-P,a2-P))==0 && dcmp(Dot(a1-P,a2-P))<=0;  
}  

bool SegmentIntersection(Point a1,Point a2,Point b1,Point b2)  
{  
  double c1=cross(a2-a1,b1-a1),c2=cross(a2-a1,b2-a1);  
  double c3=cross(b2-b1,a1-b1),c4=cross(b2-b1,a2-b1);  
  if(dcmp(c1)*dcmp(c2)<0 && dcmp(c3)*dcmp(c4)<0) return true;  
  if(dcmp(c1)==0 && InSegment(b1,a1,a2) ) return true;  
  if(dcmp(c2)==0 && InSegment(b2,a1,a2) ) return true;  
  if(dcmp(c3)==0 && InSegment(a1,b1,b2) ) return true;  
  if(dcmp(c4)==0 && InSegment(a2,b1,b2) ) return true;  
  return false;  
}  

double area(Point* ps,int n){  
  ps[n]=ps[0];  
  double res=0;  
  for(int i=0;i<n;i++){  
      res+=ps[i].x*ps[i+1].y-ps[i].y*ps[i+1].x;  
  }
  return res/2.0;  
}  

int lineCross(Point a,Point b,Point c,Point d,Point&p){  
  double s1,s2;  
  s1=cross(a,b,c);  
  s2=cross(a,b,d);  
  if(sig(s1)==0&&sig(s2)==0) return 2;  
  if(sig(s2-s1)==0) return 0;  
  p.x=(c.x*s2-d.x*s1)/(s2-s1);  
  p.y=(c.y*s2-d.y*s1)/(s2-s1);  
  return 1;  
}  

//多边形切割  
//用直线ab切割多边形p，切割后的在向量(a,b)的左侧，并原地保存切割结果  
//如果退化为一个点，也会返回去,此时n为1  
void polygon_cut(Point*p,int&n,Point a,Point b){  
  static Point pp[maxn];  
  int m=0;p[n]=p[0];  
  for(int i=0;i<n;i++){  
      if(sig(cross(a,b,p[i]))>0) pp[m++]=p[i];  
      if(sig(cross(a,b,p[i]))!=sig(cross(a,b,p[i+1])))  
          lineCross(a,b,p[i],p[i+1],pp[m++]);  
  }  
  n=0;  
  for(int i=0;i<m;i++)  
      if(!i||!(pp[i]==pp[i-1]))  
          p[n++]=pp[i];  
  while(n>1&&p[n-1]==p[0])n--;  
}  

//---------------华丽的分隔线-----------------//  
//返回三角形oab和三角形ocd的有向交面积,o是原点//  
double intersectArea(Point a,Point b,Point c,Point d){  
  Point o(0,0);  
  int s1=sig(cross(o,a,b));  
  int s2=sig(cross(o,c,d));  
  if(s1==0||s2==0)return 0.0;//退化，面积为0  
  if(s1==-1) std::swap(a,b);  
  if(s2==-1) std::swap(c,d);  
  Point p[10]={o,a,b};  
  int n=3;
  polygon_cut(p,n,o,c);  
  polygon_cut(p,n,c,d);  
  polygon_cut(p,n,d,o);  
  double res=fabs(area(p,n));  
  if(s1*s2==-1) res=-res;return res;  
}  

//求两多边形的交面积  
double intersectArea(Point*ps1, int n1, Point*ps2, int n2){  
  if(area(ps1,n1)<0) std::reverse(ps1,ps1+n1);
  if(area(ps2,n2)<0) std::reverse(ps2,ps2+n2);
  ps1[n1]=ps1[0];
  ps2[n2]=ps2[0];
  double res=0;
  for(int i=0;i<n1;i++){
    for(int j=0;j<n2;j++){
      res += intersectArea(ps1[i],ps1[i+1],ps2[j],ps2[j+1]);  
    }  
  }
  return res;//assumeresispositive!  
}

cv::Mat computeRotaMat(double yaw, double pitch, double roll)
{
  cv::Mat rotaMat;
  cv::Mat rotaMatPitch = cv::Mat(3, 3, CV_64F);
  rotaMatPitch.at<double>(0, 0) = 1;
  rotaMatPitch.at<double>(0, 1) = 0;
  rotaMatPitch.at<double>(0, 2) = 0;
  rotaMatPitch.at<double>(1, 0) = 0;
  rotaMatPitch.at<double>(1, 1) = cos(-pitch);
  rotaMatPitch.at<double>(1, 2) = -sin(-pitch);
  rotaMatPitch.at<double>(2, 0) = 0;
  rotaMatPitch.at<double>(2, 1) = sin(-pitch);
  rotaMatPitch.at<double>(2, 2) = cos(-pitch);

  cv::Mat rotaMatYaw = cv::Mat(3, 3, CV_64F);
  rotaMatYaw.at<double>(0, 0) = cos(yaw);
  rotaMatYaw.at<double>(0, 1) = 0;
  rotaMatYaw.at<double>(0, 2) = sin(yaw);
  rotaMatYaw.at<double>(1, 0) = 0;
  rotaMatYaw.at<double>(1, 1) = 1;
  rotaMatYaw.at<double>(1, 2) = 0;
  rotaMatYaw.at<double>(2, 0) = -sin(yaw);
  rotaMatYaw.at<double>(2, 1) = 0;
  rotaMatYaw.at<double>(2, 2) = cos(yaw);

  cv::Mat rotaMatRoll = cv::Mat(3, 3, CV_64F);
  rotaMatRoll.at<double>(0, 0) = cos(roll);
  rotaMatRoll.at<double>(0, 1) = -sin(roll);
  rotaMatRoll.at<double>(0, 2) = 0;
  rotaMatRoll.at<double>(1, 0) = sin(roll);
  rotaMatRoll.at<double>(1, 1) = cos(roll);
  rotaMatRoll.at<double>(1, 2) = 0;
  rotaMatRoll.at<double>(2, 0) = 0;
  rotaMatRoll.at<double>(2, 1) = 0;
  rotaMatRoll.at<double>(2, 2) = 1;

  rotaMat = rotaMatYaw*rotaMatPitch*rotaMatRoll;
  
  return rotaMat;
}


std::vector<cv::Mat> makeRotaMatList(std::vector<double>& params)
{
  std::vector<cv::Mat> rotaMats;
  for(size_t i = 0; i < params.size(); i += 3)
  {
	  cv::Mat rotaMat = computeRotaMat(params[i], params[i+1], params[i+2]);
    rotaMats.push_back(rotaMat);
  }
  return rotaMats;
}


int panoImg2Warp(cv::Mat& srcPanoImg, cv::Mat& dstWarpImg, cv::Mat& rotaMat, aParams params)
{
  if (params.panoWidth != params.panoHeight*2)
  {
    return 2;
  } 

  if (params.warpWidth <= 0 || params.warpHeight <= 0)
  {
    return 4;
  }

  cv::Mat tmpCoor = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat warpMap = cv::Mat(params.warpHeight, params.warpWidth, CV_32FC2);

  for(int j = 0; j < params.warpHeight; j++)
  {
    for(int i = 0; i < params.warpWidth; i++)
    {
      tmpCoor.at<double>(0) = (double)(i-params.warpWidth/2);
      tmpCoor.at<double>(1) = (double)(params.warpHeight/2-j);
      tmpCoor.at<double>(2) = params.F;

      tmpCoor = rotaMat*tmpCoor;
      normalize(tmpCoor, tmpCoor);

      double latitude  = asin(tmpCoor.at<double>(1));
      double longitude = atan2(tmpCoor.at<double>(0), tmpCoor.at<double>(2));

      float u = longitude*params.du + params.panoWidth/2;
      float v = params.panoHeight/2 - latitude*params.dv;

      warpMap.at<cv::Point2f>(j, i) = cv::Point2f(u, v);
    }
  }

  dstWarpImg = cv::Mat(params.warpHeight,  params.warpWidth, CV_8UC3, cv::Scalar(0, 0, 0));
  remap(srcPanoImg, dstWarpImg, warpMap, cv::Mat(), CV_INTER_CUBIC, cv::BORDER_WRAP);

  return 0;
}

void makeWarpImgList(cv::Mat& srcPanoImg, std::vector<cv::Mat>& warpImgList, std::vector<cv::Mat>& rotaMats, aParams params)
{
  int index = 0;
  cv::Mat dst;
  while(index < rotaMats.size()){
    panoImg2Warp(srcPanoImg, dst, rotaMats[index++], params);
    warpImgList.push_back(dst);
  }
}

std::vector<cv::Mat> convertRotaMat2Inv(std::vector<cv::Mat> rotaMats)
{
  std::vector<cv::Mat> rotaMatsInv;
  for(size_t i = 0; i < rotaMats.size(); i++)
  {
    rotaMatsInv.push_back(rotaMats[i].inv());
  }
  return rotaMatsInv;
}

std::pair<double, double> map2PanoCoord(double cols, double rows, aParams params, cv::Mat& rotaMat)
{
  cv::Mat tmpCoor = cv::Mat::zeros(3, 1, CV_64F);

  tmpCoor.at<double>(0) = static_cast<double>(cols - params.warpWidth/2);
  tmpCoor.at<double>(1) = static_cast<double>(params.warpHeight/2 - rows);
  tmpCoor.at<double>(2) = params.F;

  tmpCoor = rotaMat*tmpCoor;
  normalize(tmpCoor, tmpCoor);

  double latitude  = asin(tmpCoor.at<double>(1));
  double longitude = atan2(tmpCoor.at<double>(0), tmpCoor.at<double>(2));

  double pCols = longitude*params.du + params.panoWidth/2;
  double pRows = params.panoHeight/2 - latitude*params.dv;
  std::pair<double, double> coord = std::make_pair(pCols, pRows);

  return coord;
}


bool convertWarpCoord2Pano(std::vector<std::pair<double, double> >& boxesCoord, std::vector<Detection>& detections, std::vector<int>& filtedIndexs, int& current, aParams params, cv::Mat& rotaMat)
{
  for(size_t i = 0; i < detections.size(); ++i){
    Detection detection = detections[i];  
    int index = static_cast<int>(detection.id);
    int currentInx = filtedIndexs[current];
    if (currentInx == index){
      current++;
      double xmin = static_cast<double>(detection.point[0].x);
      double ymin = static_cast<double>(detection.point[0].y); 
      double xmax = static_cast<double>(detection.point[2].x); 
      double ymax = static_cast<double>(detection.point[2].y); 
      //[xmin, ymin, xmax, ymax]
      std::pair<double, double> coord;
      for(double cols = xmin; cols < xmax; cols += 1){
        double rows = ymin;
        coord = map2PanoCoord(cols, rows, params, rotaMat);
        boxesCoord.push_back(coord);

        rows = ymax;
        coord = map2PanoCoord(cols, rows, params, rotaMat);
        boxesCoord.push_back(coord);
      }
      for(double rows = ymin; rows < ymax; rows += 1){
        double cols = xmin;
        coord= map2PanoCoord(cols, rows, params, rotaMat);
        boxesCoord.push_back(coord);

        cols = xmax;
        coord= map2PanoCoord(cols, rows, params, rotaMat);
        boxesCoord.push_back(coord);
      }
    }
  }
  return true;
}

bool drawCoordInPanoImg(cv::Mat& panoImg, std::vector<std::pair<double, double> > boxesCoords)
{
  for(int index = 0; index < boxesCoords.size(); index++)
  {
    int col = static_cast<int>(boxesCoords[index].first);
    int row = static_cast<int>(boxesCoords[index].second);
    cv::Point point;
    point.x = col;
    point.y = row;
    cv::circle(panoImg, point, 1, cv::Scalar(0,0,255));
  }

  return true;
}

std::pair<int, int> fixPointRange(int x, int y, int width, int height)
{
  if(x <= 0)
  {
    x = 1;
  } else if(x > width){
    x = width;
  } else if(y <= 0){
    y = 1;
  } else if(y > height){
    y = height;
  }
  std::pair<int, int> p = std::make_pair(x, y);

  return p;
}

bool drawCoordInWarpImg(cv::Mat& warpImg, std::vector<Detection>& detections, double confidence_threshold)
{
  //Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
  for(size_t i = 0; i < detections.size(); ++i)
  {
    Detection& detection = detections[i];
    const float score = detection.score; 
    if (score >= confidence_threshold)
    {
      int xmin = static_cast<double>(detection.point[0].x);
      int ymin = static_cast<double>(detection.point[0].y); 
      int xmax = static_cast<double>(detection.point[2].x); 
      int ymax = static_cast<double>(detection.point[2].y); 
      std::pair<int, int> pMin = fixPointRange(xmin, ymin, warpImg.cols, warpImg.rows);
      std::pair<int, int> pMax = fixPointRange(xmax, ymax, warpImg.cols, warpImg.rows);
      cv::rectangle(warpImg, cv::Point(pMin.first, pMin.second), cv::Point(pMax.first, pMax.second), 
          cv::Scalar(255, 0, 255), 2);
    }
  }
  return true;
}

bool setCoord(Detection& d, Point* p, aParams params, std::vector<cv::Mat> rotaMats)
{
  std::pair<double, double> coord;
  coord = map2PanoCoord(d.point[0].x, d.point[0].y, params, rotaMats[d.imgId]);
  p[0].x = coord.first;
  p[0].y = coord.second;
  coord = map2PanoCoord(d.point[1].x, d.point[1].y, params, rotaMats[d.imgId]);
  p[1].x = coord.first;
  p[1].y = coord.second;
  coord = map2PanoCoord(d.point[2].x, d.point[2].y, params, rotaMats[d.imgId]);
  p[2].x = coord.first;
  p[2].y = coord.second;
  coord = map2PanoCoord(d.point[3].x, d.point[3].y, params, rotaMats[d.imgId]);
  p[3].x = coord.first;
  p[3].y = coord.second;
  if((fabs(p[0].x - p[2].x) > params.panoWidth/2) || 
        (fabs(p[0].y - p[2].y) > params.panoHeight/2))
  {
    std::cout << "num " << d.imgId << std::endl;
    std::cout << fabs(p[0].x - p[2].x) <<  "   " << 
        fabs(p[0].y - p[2].y) << std::endl;
    for(int i = 0; i < 4; i++)
    {
      std::cout << "a  " << p[i].x << "  " << p[i].y << std::endl;
    }
    p[1].x = params.panoWidth + p[1].x;
    p[2].x = params.panoWidth + p[2].x;
  }

  p[4] = p[0];

  return true;
}

float computeOverlap(Detection d1, Detection d2, aParams& params, std::vector<cv::Mat>& rotaMats)
{
  Point ps1[10], ps2[10];
  setCoord(d1, ps1, params, rotaMats);
  setCoord(d2, ps2, params, rotaMats);
  if(d2.imgId == 0)
  {
    for(int i = 0; i < 4; i++)
    {
      std::cout << "point 1 is " << ps1[i].x <<  "  " << ps1[i].y << std::endl;
    }
    for(int i = 0; i < 4; i++)
    {
      std::cout << "point 2 is " << ps2[i].x <<  "  " << ps2[i].y << std::endl;
    }
  }

  bool flag = false;
  for(int i = 0; i < 4; i++)
  {
    for(int j = 0; j < 4; j++)
    {
      if(SegmentIntersection(ps1[i], ps1[i+1], ps2[j], ps2[j+1]))
      {
        flag = true;
      }
    }
  }

  if((ps1[0].x > ps2[0].x) && (ps1[0].y > ps2[0].y) && (ps1[1].x < ps2[1].x)
      && (ps1[2].y < ps2[2].y)) 
  {
    flag = true;
  } else if((ps1[0].x > ps2[0].x) && (ps1[0].y > ps2[0].y) && (ps1[1].x < ps2[1].x)
      && (ps1[2].y < ps2[2].y)) 
  {
    flag = true;
  }

  if(flag)
  {
    double interArea = intersectArea(ps1, 4, ps2, 4);
    double area1 = fabs(area(ps1, 4));
    double area2 = fabs(area(ps2, 4));
    double unionArea = area1 + area2 - interArea;
    //std::cout << area1 << " 1111 " << area2 << std::endl;
    //std::cout << interArea << " area " << unionArea << std::endl;
    //std::cout << " max " << interArea/unionArea <<  " " << std::max(interArea/area1, interArea/area2)
    //  << std::endl;
    return std::max(interArea/unionArea, std::max(interArea/area1, interArea/area2));
    //return interArea/unionArea;
  } else {
    return 0;
  }
}

bool sortByScore(Detection a, Detection b)
{
  return a.score > b.score;
}

bool applyNMS(std::vector<Detection >& mapDetections, std::vector<int>& filtedIndexs, aParams& params, std::vector<cv::Mat> rotaMats)
{
  sort(mapDetections.begin(), mapDetections.end(), sortByScore);
  int num = 0;
  for(int i = 0; i < mapDetections.size(); i++)
  {
    Detection d1 = mapDetections[i];
    for(int j = mapDetections.size()-1; j > i; j--)
    {
      Detection d2 = mapDetections[j];
      float th = computeOverlap(d1, d2, params, rotaMats);
      if(d2.imgId == 0){
        if(th > 0.3)
        {
          std::cout << th << " th " << std::endl;
        }
      }
      //if(th > 0 && (d2.imgId == 0 || d2.imgId == 1))
      //{
      //  int xmin = static_cast<double>(d1.point[0].x);
      //  int ymin = static_cast<double>(d1.point[0].y); 
      //  int xmax = static_cast<double>(d1.point[2].x); 
      //  int ymax = static_cast<double>(d1.point[2].y); 
      //  std::cout << "th is " << th << std::endl;
      //  std::cout << xmin << " " << xmax << " " << ymin << " " << ymax << std::endl;
      //  xmin = static_cast<double>(d2.point[0].x);
      //  ymin = static_cast<double>(d2.point[0].y); 
      //  xmax = static_cast<double>(d2.point[2].x); 
      //  ymax = static_cast<double>(d2.point[2].y); 
      //  std::cout << xmin << " " << xmax << " " << ymin << " " << ymax << std::endl;
      //  std::cout << std::endl;
      //  num++;
      //}
      if(th > 0.3)
      {
        mapDetections.erase(mapDetections.begin() + j);
      }
    }
  }
  std::cout << num << std::endl;

  for(int i = 0; i < mapDetections.size(); i++)
  {
    Detection d = mapDetections[i];
    filtedIndexs.push_back(d.id);
  }

  return true;
}

bool applyNMS4Detections(std::vector<std::vector<Detection > >& allDetections, std::vector<int>& filtedIndexs, aParams& params, float confidence_threshold, std::vector<cv::Mat> rotaMats)
{
  std::vector<Detection> mapDetections;
  for(int j = 0; j < allDetections.size(); j++)
  {
    std::vector<Detection > detections = allDetections[j];
    cv::Mat rotaMat = rotaMats[j];
    for(int i = 0; i < detections.size(); ++i)
    {
      Detection detection = detections[i];
      const float score = detection.score;
      Detection mapDetection;
      if (score >= confidence_threshold){
        mapDetection.imgId = detection.imgId;
        mapDetection.score = detection.score;
        mapDetection.label = detection.label;
        for(int k = 0; k < 4; k++)
        {
          double x = static_cast<double>(detection.point[k].x);
          double y = static_cast<double>(detection.point[k].y); 
          std::pair<double, double> coord = map2PanoCoord(x, y, params, rotaMat);
          mapDetection.point[k].x = coord.first;
          mapDetection.point[k].y = coord.second;
        }
        
        mapDetections.push_back(detection);
      }
    }
  }
  applyNMS(mapDetections, filtedIndexs, params, rotaMats);
  sort(filtedIndexs.begin(), filtedIndexs.end());
  mapDetections.clear();

  return true;
}
