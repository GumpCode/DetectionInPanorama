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


int panoImg2Warp(cv::Mat& srcPanoImg, cv::Mat& dstWarpImg, cv::Mat& rotaMat, double hFOV, int width, int height)
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

  cv::Mat tmpCoor = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat warpMap = cv::Mat(height, width, CV_32FC2);

  for(int j = 0; j < height; j++)
  {
    for(int i = 0; i < width; i++)
    {
      tmpCoor.at<double>(0) = (double)(i-width/2);
      tmpCoor.at<double>(1) = (double)(height/2-j);
      tmpCoor.at<double>(2) = F;

      tmpCoor = rotaMat*tmpCoor;
      normalize(tmpCoor, tmpCoor);

      double latitude  = asin(tmpCoor.at<double>(1));
      double longitude = atan2(tmpCoor.at<double>(0), tmpCoor.at<double>(2));

      float u = longitude*du + srcWidth/2;
      float v = srcHeight/2 - latitude*dv;

      warpMap.at<cv::Point2f>(j, i) = cv::Point2f(u, v);
    }
  }

  dstWarpImg = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  remap(srcPanoImg, dstWarpImg, warpMap, cv::Mat(), CV_INTER_CUBIC, cv::BORDER_WRAP);

  return 0;
}

void makeWarpImgList(cv::Mat& srcPanoImg, std::vector<cv::Mat>& warpImgList, std::vector<cv::Mat>& rotaMats, const std::vector<int> size, double hFOV)
{
  int width = size[0];
  int height = size[1];
  size_t index = 0;
  cv::Mat dst;
  while(index < rotaMats.size()){
    panoImg2Warp(srcPanoImg, dst, rotaMats[index++], hFOV, width, height);
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

std::pair<double, double> map2PanoCoord(double cols, double rows, std::vector<int> size, double F, double du, double dv, cv::Mat& rotaMat)
{
  int warpWidth = size[0];
  int warpHeight = size[1];
  int panoWidth = size[2];
  int panoHeight = size[3];
  cv::Mat tmpCoor = cv::Mat::zeros(3, 1, CV_64F);

  tmpCoor.at<double>(0) = static_cast<double>(rows - warpWidth/2);
  tmpCoor.at<double>(1) = static_cast<double>(warpHeight/2 - cols);
  tmpCoor.at<double>(2) = F;

  tmpCoor = rotaMat*tmpCoor;
  normalize(tmpCoor, tmpCoor);

  double latitude  = asin(tmpCoor.at<double>(1));
  double longitude = atan2(tmpCoor.at<double>(0), tmpCoor.at<double>(2));

  double pCols = longitude*du + panoWidth/2;
  double pRows = panoHeight/2 - latitude*dv;
  std::pair<double, double> coord = std::make_pair(pRows, pCols);

  return coord;
}


bool convertWarpCoord2Pano(std::vector<std::pair<double, double> >& boxesCoord, std::vector<std::vector<float> >& detections,
		const std::vector<int> size, float confidence_threshold, cv::Mat& rotaMat, double hFOV)
{
  int warpWidth = size[0];
  int warpHeight = size[1];
  int panoWidth = size[2];
  int panoHeight = size[3];
  double F  = (warpWidth/2)/tan(hFOV/2.f);
  double du = panoWidth/2/CV_PI;
  double dv = panoHeight/CV_PI;
  for(size_t i = 0; i < detections.size(); ++i){
    const std::vector<float>& detection = detections[i];  
    const float score = detection[2]; 
    if (score >= confidence_threshold){
      double xmin = static_cast<double>(detection[3] * warpWidth);
      double ymin = static_cast<double>(detection[4] * warpHeight); 
      double xmax = static_cast<double>(detection[5] * warpWidth); 
      double ymax = static_cast<double>(detection[6] * warpHeight); 
      //[xmin, ymin, xmax, ymax]
      for(double rows = xmin; rows < xmax; rows += 1){
        double cols = ymin;
        std::pair<double, double> coord1 = map2PanoCoord(cols, rows, size, F, du, dv, rotaMat);
        boxesCoord.push_back(coord1);
      }

      for(double rows = xmin; rows < xmax; rows += 1){
        double cols = ymax;
        std::pair<double, double> coord2 = map2PanoCoord(cols, rows, size, F, du, dv, rotaMat);
        boxesCoord.push_back(coord2);
      }

      for(double cols = ymin; cols < ymax; cols += 1){
        double rows = xmin;
        std::pair<double, double> coord1 = map2PanoCoord(cols, rows, size, F, du, dv, rotaMat);
        boxesCoord.push_back(coord1);
      }

      for(double cols = ymin; cols < ymax; cols += 1){
        double rows = xmax;
        std::pair<double, double> coord2 = map2PanoCoord(cols, rows, size, F, du, dv, rotaMat);
        boxesCoord.push_back(coord2);
      }
    }
  }
  return true;
}

bool convertWarpCoord2Pano2(std::vector<std::pair<double, double> >& boxesCoord, std::vector<std::vector<float> >& detections, std::vector<int>& filtedIndexs, int& current, const std::vector<int> size, cv::Mat& rotaMat, double hFOV)
{
  int warpWidth = size[0];
  int warpHeight = size[1];
  int panoWidth = size[2];
  int panoHeight = size[3];
  double F  = (warpWidth/2)/tan(hFOV/2.f);
  double du = panoWidth/2/CV_PI;
  double dv = panoHeight/CV_PI;
  for(size_t i = 0; i < detections.size(); ++i){
    const std::vector<float> detection = detections[i];  
    int index = static_cast<int>(detection[0]);
    int currentInx = filtedIndexs[current];
    //std::cout << current << std::endl;
    //std::cout << currentInx << " index " << index << std::endl;
    if (currentInx == index){
      current++;
      double xmin = static_cast<double>(detection[3] * warpWidth);
      double ymin = static_cast<double>(detection[4] * warpHeight); 
      double xmax = static_cast<double>(detection[5] * warpWidth); 
      double ymax = static_cast<double>(detection[6] * warpHeight); 
      //[xmin, ymin, xmax, ymax]
      std::pair<double, double> coord;
      for(double rows = xmin; rows < xmax; rows += 1){
        double cols = ymin;
        coord = map2PanoCoord(cols, rows, size, F, du, dv, rotaMat);
        boxesCoord.push_back(coord);

        cols = ymax;
        coord = map2PanoCoord(cols, rows, size, F, du, dv, rotaMat);
        boxesCoord.push_back(coord);
      }
      for(double cols = ymin; cols < ymax; cols += 1){
        double rows = xmin;
        coord= map2PanoCoord(cols, rows, size, F, du, dv, rotaMat);
        boxesCoord.push_back(coord);

        rows = xmax;
        coord= map2PanoCoord(cols, rows, size, F, du, dv, rotaMat);
        boxesCoord.push_back(coord);
      }
    }
  }
  return true;
}

bool drawCoordInPanoImg(cv::Mat& panoImg, std::vector<std::pair<double, double> > boxesCoords)
{
  for(size_t index = 0; index < boxesCoords.size(); index++)
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

//bool drawCoordInPanoImg(cv::Mat& panoImg, std::vector<std::vector<float> > boxesCoords)
//{
//  for(size_t index = 0; index < boxesCoords.size(); index++)
//  {
//    std::vector<float> detection = boxesCoords[index];
//    int row = static_cast<int>(detection[);
//    int col = static_cast<int>(boxesCoords[index].second);
//    cv::Point point;
//    point.x = col;
//    point.y = row;
//    cv::circle(panoImg, point, 1, cv::Scalar(0,0,255));
//  }
//
//  return true;
//}

bool fixPointRange(int& x, int& y, int width, int height)
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
  return true;
}

bool drawCoordInWarpImg(cv::Mat& warpImg, std::vector<std::vector<float> >& detections, double confidence_threshold)
{
  //Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
  for(size_t i = 0; i < detections.size(); ++i)
  {
    const std::vector<float>& detection = detections[i];  
    const float score = detection[2]; 
    if (score >= confidence_threshold)
    {
      int xmin = static_cast<double>(detection[3] * warpImg.cols);
      int ymin = static_cast<double>(detection[4] * warpImg.rows); 
      int xmax = static_cast<double>(detection[5] * warpImg.cols); 
      int ymax = static_cast<double>(detection[6] * warpImg.rows); 
      fixPointRange(xmin, ymin, warpImg.cols, warpImg.rows);
      fixPointRange(xmax, ymax, warpImg.cols, warpImg.rows);
      cv::rectangle(warpImg, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 0, 255), 2);
    }
  }
  return true;
}


//bool map2OtherView(std::vector<std::vector<float > >&mapCoords, std::vector<std::vector<float> >& coords, std::vector<cv::Mat>& rotaMats, int srcNum, int dstNum, float confidence_threshold, std::vector<int> size, double hFOV, double angleOffset)
//{
//  int num = 0;
//  for(size_t k = 0; k < coords.size(); k++)
//  {
//    const vector<float>& coord = coords[k];
//    float score = coord[2];
//    if(score > confidence_threshold){
//      num++;
//    }
//  }
//  if(!num)
//  {
//    return false;
//  }
//
//  double F  = (size[0]/2)/tan(hFOV/2.f);
//  for(int i = 0; i < coords.size(); i++)
//  {
//    vector<float> coord = coords[i];
//    vector<float> tmp(coord);
//    float score = coord[2];
//    if(score > confidence_threshold)
//    {
//      //flip(tmp, size);
//      for(int j = 3; j < coord.size(); j+=2)
//      {
//        cv::Mat tmpCoor = Mat::zeros(3, 1, CV_64F);
//        //std::cout << "coord" << j << " is " << tmp[j] << std::endl;
//        //std::cout << "coord" << j+1 << " is " << tmp[j+1] << std::endl;
//        tmpCoor.at<double>(0) = static_cast<double>(tmp[j]*size[0] - size[0]/2);
//        tmpCoor.at<double>(1) = static_cast<double>(size[1]/2 - tmp[j+1]*size[1]);
//        tmpCoor.at<double>(2) = F;
//        std::cout << "coord" << j << " is " << tmpCoor.at<double>(0) << std::endl;
//        std::cout << "coord" << j+1 << " is " << tmpCoor.at<double>(1) << std::endl;
//        tmpCoor = rotaMats[dstNum]*rotaMats[srcNum].inv()*tmpCoor;
//        tmp[0] = tmpCoor.at<float>(0) + size[0]/2;
//        tmp[1] = size[1]/2 - tmpCoor.at<float>(1);
//        //std::cout << "tmp" << j << " is " << tmp[0] << std::endl;
//        //std::cout << "tmp" << j+1 << " is " << tmp[1] << std::endl;
//      }
//      tmp.push_back(i);
//      mapCoords.push_back(tmp);
//    }
//  }
//  return true;
//}
//
//float computCrossArea(std::vector<float>& mapCoord1, std::vector<float>& mapCoord2)
//{
//  float area = 0;
//  return area;
//}
//
//bool applyNMS(std::vector<std::vector<float> >& mapCoords, std::vector<std::vector<float> >& detections)
//{
//  float area_threshold = 0.5;
//  for(size_t i = 0; i < mapCoords.size(); i++)
//  {
//    for(size_t j = i; j < mapCoords.size(); j++)
//    {
//      if(computCrossArea(mapCoords[i], mapCoords[j]) > area_threshold)
//      {
//        if(mapCoords[i][2] > mapCoords[j][2])
//        {
//          //to do: remove j by index
//          ;
//        }
//      }
//    }
//  }
//  return true;
//}
//
//bool applyNMS4Detections(std::vector<std::vector<std::vector<float> > >& allDetections, std::vector<int> size, float confidence_threshold, std::vector<cv::Mat >& rotaMats, double hFOV, double angleOffset)
//{
//  for(int i = 0; i < allDetections.size(); i++)
//  {
//    std::vector<std::vector<float> > mapCoords;
//    if(i==0)
//    //if((i>=0) && (i<=7))
//    {
//      map2OtherView(mapCoords, allDetections[i+1], rotaMats, i+1, i, confidence_threshold, size, hFOV, angleOffset);
//      //applyNMS(mapCoords, allDetections[i+1]);
//      //map2OtherView(mapCoords, allDetections[i+8], rotaMats, i+8, i, confidence_threshold, size, hFOV);
//      //applyNMS(allDetections[i+8], mapCoords);
//      //map2OtherView(mapCoords, allDetections[i+9], rotaMats, i+9, i, confidence_threshold, size, hFOV);
//      //applyNMS(allDetections[i+9], mapCoords);
//      //map2OtherView(mapCoords, allDetections[24], rotaMats, 24, i, confidence_threshold, size, hFOV);
//      //applyNMS(allDetections[24], mapCoords);
//    //} else if((i>=8) && (i<=15)) {
//      //map2OtherView(mapCoords, allDetections[i-8], rotaMats, i-8, i, confidence_threshold, size, hFOV);
//      //map2OtherView(mapCoords, allDetections[i-7], rotaMats, i-7, i, confidence_threshold, size, hFOV);
//      //map2OtherView(mapCoords, allDetections[i+1], rotaMats, i+1, i, confidence_threshold, size, hFOV);
//      //map2OtherView(mapCoords, allDetections[i+8], rotaMats, i+8, i, confidence_threshold, size, hFOV);
//      //map2OtherView(mapCoords, allDetections[i+9], rotaMats, i+9, i, confidence_threshold, size, hFOV);
//      //applyNMS(allDetections[i], mapCoords);
//
//    //} else if((i>=16) && (i<=23)) {
//    //  map2OtherView(mapCoords, allDetections[i+1], rotaMats, i+1, i, confidence_threshold, hFOV);
//    //  map2OtherView(mapCoords, allDetections[i+8], rotaMats, i+8, i, confidence_threshold, hFOV);
//    //  map2OtherView(mapCoords, allDetections[i+9], rotaMats, i+9, i, confidence_threshold, hFOV);
//    //  map2OtherView(mapCoords, allDetections[25], rotaMats, 25, i, confidence_threshold, hFOV);
//    //  applyNMS(allDetections[i], mapCoords);
//    }
//  }
//  return true;
//}
//

bool setCoord(std::vector<float>& d, Point* p)
{
  float xmin = d[3];
  float ymin = d[4];
  float xmax = d[5];
  float ymax = d[6];
  p[0].x = static_cast<double>(xmin);
  p[0].y = static_cast<double>(ymin);
  p[1].x = static_cast<double>(xmax);
  p[1].y = static_cast<double>(ymin);
  p[2].x = static_cast<double>(xmax);
  p[2].y = static_cast<double>(ymax);
  p[3].x = static_cast<double>(xmin);
  p[3].y = static_cast<double>(ymax);
  p[4] = p[0];

  return true;
}

float computeOverlap(std::vector<float> d1, std::vector<float> d2)
{
  Point ps1[100], ps2[100];
  setCoord(d1, ps1);
  setCoord(d2, ps2);

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
    //std::cout << interArea << " area " << unionArea << std::endl;
    //std::cout << std::max(interArea/unionArea, std::max(interArea/area1, interArea/area2)) 
    //  << std::endl;
    return std::max(interArea/unionArea, std::max(interArea/area1, interArea/area2));
  } else {
    return 0;
  }
}

bool compareFunc(std::vector<float> a, std::vector<float> b)
{
  return a[2] > b[2];
}

bool applyNMS(std::vector<std::vector<float> >& mapDetections, std::vector<int>& filtedIndexs)
{
  //for(int i = 0; i < mapDetections.size(); i++)
  //{
  //  std::vector<float> d = mapDetections[i];
  //  std::cout << d[0] << " aaa" << std::endl;
  //}
  sort(mapDetections.begin(), mapDetections.end(), compareFunc);
  for(int i = 0; i < mapDetections.size(); i++)
  {
    std::vector<float> d1 = mapDetections[i];
    for(int j = mapDetections.size()-1; j > i; j--)
    {
      std::vector<float> d2 = mapDetections[j];
      float th = computeOverlap(mapDetections[i], mapDetections[j]);
      std::cout << "th is " << th << std::endl;
      //if(computeOverlap(mapDetections[i], mapDetections[j]) > 0.5)
      //if((th > 0.5) && (d1[1] != d2[1]))
      if(th > 0.1)
      {
        std::cout << th << std::endl;
        //std::cout << d2[1] << " " << d2[2] << " " << d2[3] << " " << d2[4] << " "
        //  << d2[5] << " " << d2[6] << std::endl;
        mapDetections.erase(mapDetections.begin() + j);
      }
    }
  }

  for(int i = 0; i < mapDetections.size(); i++)
  {
    std::vector<float> d = mapDetections[i];
    //std::cout << d[0] << " aaa" << std::endl;
    filtedIndexs.push_back(d[0]);
  }

  return true;
}

bool applyNMS4Detections(std::vector<std::vector<std::vector<float> > >& allDetections, std::vector<int>& filtedIndexs, std::vector<int> size, float confidence_threshold, std::vector<cv::Mat> rotaMats, double hFOV)
{
  int warpWidth = size[0];
  int warpHeight = size[1];
  int panoWidth = size[2];
  int panoHeight = size[3];
  double F  = (warpWidth/2)/tan(hFOV/2.f);
  double du = panoWidth/2/CV_PI;
  double dv = panoHeight/CV_PI;
  std::vector<std::vector<float> > mapDetections;
  for(int j = 0; j < allDetections.size(); j++)
  {
    std::vector<std::vector<float> > detections = allDetections[j];
    cv::Mat rotaMat = rotaMats[j];
    for(int i = 0; i < detections.size(); ++i)
    {
      const std::vector<float> detection = detections[i];  
      const float score = detection[2];
      std::vector<float> mapDetection;
      if (score >= confidence_threshold){
        for(int k = 0; k < detection.size(); k++)
        {
          if(k < 3)
          {
            mapDetection.push_back(detection[k]);
          } else {
            double x = static_cast<double>(detection[k++] * warpWidth);
            double y = static_cast<double>(detection[k] * warpHeight); 
            std::pair<double, double> coord = map2PanoCoord(x, y, size, F, du, dv, rotaMat);
            mapDetection.push_back(coord.first);
            mapDetection.push_back(coord.second);
          }
        }
        mapDetections.push_back(mapDetection);
      }
    }
  }
  applyNMS(mapDetections, filtedIndexs);
  sort(filtedIndexs.begin(), filtedIndexs.end());
  mapDetections.clear();

  return true;
}
