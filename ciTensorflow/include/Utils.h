#ifndef TF_DETECTOR_EXAMPLE_UTILS_H
#define TF_DETECTOR_EXAMPLE_UTILS_H

#endif //TF_DETECTOR_EXAMPLE_UTILS_H

#include <math.h>
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <regex>
#include <numeric>      // std::iota

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/cc/client/client_session.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cinder/app/App.h"
#include "cinder/gl/gl.h"

#include "CinderOpenCv.h"



namespace tf {

	using tensorflow::Tensor;
	using tensorflow::Status;
	using tensorflow::string;
	
	Status readLabelsMapFile(const string &fileName, std::map<int, string> &labelsMap);

	Status loadGraph(const string &graph_file_name,
		std::unique_ptr<tensorflow::Session> *session);

	Status readTensorFromMat(const cv::Mat &mat, Tensor &outTensor);

	void surfaceTocvMat(ci::Surface8u surf, cv::Mat *cvmt);
	void cvMatToSurface(cv::Mat cvmt, ci::Surface8u *surf);
	void cvMatToTexture(cv::Mat cvmt, ci::gl::TextureRef tex);

	void drawBoundingBoxOnImage(cv::Mat &image, double xMin, double yMin, double xMax, double yMax, double score, std::string label, bool scaled);

	void drawBoundingBoxesOnImage(cv::Mat &image,
		tensorflow::TTypes<float>::Flat &scores,
		tensorflow::TTypes<float>::Flat &classes,
		tensorflow::TTypes<float, 3>::Tensor &boxes,
		std::map<int, string> &labelsMap,
		std::vector<size_t> &idxs);

	void GetBoundingBoxesOnImage(
		tensorflow::TTypes<float>::Flat &scores,
		tensorflow::TTypes<float>::Flat &classes,
		tensorflow::TTypes<float, 3>::Tensor &boxes,
		std::vector<size_t> &idxs,
		std::vector<double>& rXmin,
		std::vector<double>& rYmin,
		std::vector<double>& rXmax,
		std::vector<double>& rYmax,
		std::vector<double>& rScore);

	double IOU(cv::Rect box1, cv::Rect box2);

	std::vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &scores,
		tensorflow::TTypes<float, 3>::Tensor &boxes,
		double thresholdIOU, double thresholdScore);

}