#pragma once

#include "utils.h"


namespace tf {


	class ciTensorflow {
	public:
		ciTensorflow();
		~ciTensorflow();
		void updateTensor();
		void resize(ci::vec2 size);
		bool isReady() { return gInferReady; };
		cv::Mat input;

	private:

		void tf_infer();
		void tf_pixtopixinfer();

		int face_detected;
		bool gInferReady = true;
		// FPS counter
	//	int iFrame = 0;

		ci::gl::TextureRef	mTexture;

		tensorflow::TensorShape shape;
		std::vector<Tensor> outputs;
		// Set input & output nodes names
		std::string inputLayer = "image_tensor:0";
		std::vector<std::string> outputLayer = { "detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0" };
		// Load labels map from .pbtxt file
		std::map<int, std::string> labelsMap = std::map<int, std::string>();
		cv::Mat output;
		std::unique_ptr<tensorflow::Session> session;

		std::mutex gLockBuffer;
		std::vector<double> gXmin, gYmin, gXmax, gYmax, gScore;
		std::vector<double> gXminBuffer, gYminBuffer, gXmaxBuffer, gYmaxBuffer, gScoreBuffer;

		std::shared_ptr<std::thread> tf_thread;

	};



}