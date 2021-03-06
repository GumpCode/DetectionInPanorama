// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <sys/time.h>

#include "caffe/util/panorama_util.hpp"

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)

class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file,
           const string& mean_file,
           const string& mean_value);

  std::vector<vector<float> > Detect(const cv::Mat& img);

 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value) {
  //Caffe::set_mode(Caffe::CPU);
  Caffe::set_mode(Caffe::GPU);

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.01,
    "Only store detections with score higher than the threshold.");
//DEFINE_string(detected_flag, "",
//    "the flag point to detect for full or part.");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using SSD mode.\n"
        "Usage:\n"
        "    ssd_detect [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
    return 1;
  }

  const string& model_file = argv[1];
  const string& weights_file = argv[2];
  const string& mean_file = FLAGS_mean_file;
  const string& mean_value = FLAGS_mean_value;
  //const string& out_file = FLAGS_out_file;
  //const string& flag = FLAGS_detected_flag;
  const float confidence_threshold = FLAGS_confidence_threshold;

  // Initialize the network.
  Detector detector(model_file, weights_file, mean_file, mean_value);

  // Set the output mode.
  //std::streambuf* buf = std::cout.rdbuf();
  //std::ofstream outfile;
  //if (!out_file.empty()) {
  //  outfile.open(out_file.c_str());
  //  if (outfile.good()) {
  //    buf = outfile.rdbuf();
  //  }
  //}
  //std::ostream out(buf);

  //make the stored vector

  // Process image one by one.
  int warpWidth = 300;
  int warpHeight = 300;
  int panoWidth = 3840;
  int panoHeight = 1920;
  //int s[] = {warpWidth, warpHeight, panoWidth, panoHeight};
  //std::vector<int> size(s, s+4);
  //std::vector<double> params;
  //const double angle_47 = 47*CV_PI/180;
  const double angle_45 = 45*CV_PI/180;
  const double hFOV = 47*CV_PI/180;
  const double roll = 0.0;
  //const double angleOffset = 45*CV_PI/180;

  /*准备每个视图的yaw, pitch, roll参数*/
  std::vector<double> rotaParams;
  for(double pitch = -angle_45; pitch < angle_45+0.01; pitch += angle_45)
  {
    int n = 0;
    for(double yaw = 0; yaw < 2*CV_PI; yaw += angle_45)
    {
      n++;
      //if((yaw + hFOV/2) > 2*CV_PI)
      //{
      //  std::cout << yaw + hFOV/2 << std::endl;
      //  std::cout << "error" << std::endl;
      //}
      //[yaw, pitch, roll]
      rotaParams.push_back(yaw);
      rotaParams.push_back(pitch);
      rotaParams.push_back(roll);
    }
  }

  //for top view
  rotaParams.push_back(0.0);
  rotaParams.push_back(CV_PI);
  rotaParams.push_back(roll);
  //for bottom view
  rotaParams.push_back(0.0);
  rotaParams.push_back(-CV_PI/2);
  rotaParams.push_back(roll);

  std::vector<cv::Mat> rotaMats = makeRotaMatList(rotaParams);
  std::vector<cv::Mat> rotaMatsInv = convertRotaMat2Inv(rotaMats);

  std::ifstream infile(argv[3]);
  std::string file;
  std::vector<cv::Mat> warpImgs;

  double F = (warpWidth/2)/tan(hFOV/2.f);
  double du = panoWidth/2/CV_PI;
  double dv = panoHeight/CV_PI;
  aParams params(warpWidth, warpHeight, panoWidth, panoHeight, F, du, dv);

  struct timeval start;
  struct timeval end;
  unsigned long timer;
  gettimeofday(&start,NULL);

  while(infile >> file){
    cv::Mat img = cv::imread(file, -1);
    CHECK(!img.empty()) << "Unable to decode image " << file;
    makeWarpImgList(img, warpImgs, rotaMats, params);
    std::vector<std::vector<Detection> > allDetections;
    int num_ = 0;
    for(int i = 0; i < warpImgs.size(); i++){
      cv::Mat im = warpImgs[i];
      cv::Mat rotaMat = rotaMats[i];
      //Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
      std::vector<std::vector<float> > detections = detector.Detect(im);
      std::vector<Detection> detections_tmp;
      //Point p1, p2, p3, p4;
      for(int j = 0; j < detections.size(); j++)
      {
        Detection detection;
        std::vector<float> d = detections[j];
        detection.imgId = i;
        detection.id = num_++;
        detection.label = d[1];
        detection.score = d[2];
        detection.point[0].x = static_cast<double>(d[3] * params.warpWidth);  //xmin
        detection.point[0].y = static_cast<double>(d[4] * params.warpHeight); //ymin
        detection.point[1].x = static_cast<double>(d[5] * params.warpWidth);  //xmax
        detection.point[1].y = static_cast<double>(d[4] * params.warpHeight); //ymin
        detection.point[2].x = static_cast<double>(d[5] * params.warpWidth);  //xmax
        detection.point[2].y = static_cast<double>(d[6] * params.warpHeight); //ymax
        detection.point[3].x = static_cast<double>(d[3] * params.warpWidth);  //xmin
        detection.point[3].y = static_cast<double>(d[6] * params.warpHeight); //ymax
        for(int k = 0; k < 4; k++)
        {
          std::pair<int, int> p = fixPointRange(detection.point[k].x, detection.point[k].y,
              params.warpWidth, params.warpHeight);
          detection.point[k].x = p.first;
          detection.point[k].y = p.second;
        }
        detections_tmp.push_back(detection);
      }
      allDetections.push_back(detections_tmp);

      drawCoordInWarpImg(im, detections_tmp, confidence_threshold);
      std::stringstream ss;
      ss << i;
      std::string s;
      ss >> s;
      s = s + ".jpg";
      cv::imwrite(s, im);
    }

    std::vector<int> filtedIndexs;
    applyNMS4Detections(allDetections, filtedIndexs, params, confidence_threshold,
        rotaMats);

    std::vector<std::pair<double, double> > boxesCoords;
    int current = 0;
    for(int j = 0; j < allDetections.size(); j++)
    {
      cv::Mat rotaMat = rotaMats[j];
      std::vector<Detection> detections = allDetections[j];
      convertWarpCoord2Pano(boxesCoords, detections, filtedIndexs, 
          current, params, rotaMat);
    }
    drawCoordInPanoImg(img, boxesCoords);
    cv::imwrite("out.jpg", img);
    //cv::imshow("a", img);
    //cv::waitKey(0);
  }
  gettimeofday(&end,NULL);
  timer = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
  std::cout << timer/1000 << std::endl;
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
