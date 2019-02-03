#include "tf.h"


namespace tf {

	ciTensorflow::ciTensorflow() {
		ci::Surface8u surface(ci::loadImage(ci::app::loadAsset("dfw.jpg")));
		input = cv::Mat(toOcv(surface));
		// output;
		// Mat fakeMat(mat.rows, mat.cols, CV_32FC3, p);
		//	cv::medianBlur( input, output, 11 );
		cv::Sobel(input, input, CV_8U, 0, 1);
		//	cv::threshold( input, output, 128, 255, CV_8U );

		mTexture = ci::gl::Texture::create(ci::fromOcv(input));

		// Set dirs variables
		string ROOTDIR = "../";
		string LABELS = "model/labels_map.pbtxt";
		string GRAPH = "model/model.pb";

		string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
		ci::app::console() << "INFO: graphPath:" << graphPath << std::endl;
		Status loadGraphStatus = loadGraph(graphPath, &session);
		if (!loadGraphStatus.ok()) {
			ci::app::console() << "ERROR: loadGraph(): ERROR " << loadGraphStatus << std::endl;
			//return -1;
		}
		else
			ci::app::console() << "INFO: loadGraph(): frozen graph loaded" << std::endl;

		Status readLabelsMapStatus = readLabelsMapFile(tensorflow::io::JoinPath(ROOTDIR, LABELS), labelsMap);
		if (!readLabelsMapStatus.ok()) {
			ci::app::console() << "ERROR: readLabelsMapFile(): ERROR " << loadGraphStatus << std::endl;
			//return -1;
		}
		
		ci::app::console() << "INFO: readLabelsMapFile(): labels map loaded with " << labelsMap.size() << " label(s)" << std::endl;
		for (int l = 0;l < labelsMap.size(); l++)
		{
			ci::app::console() << l << ": " << labelsMap[l] << std::endl;
		}
		shape = tensorflow::TensorShape();
		shape.AddDim(1);
		shape.AddDim((int64)mTexture->getHeight());
		shape.AddDim((int64)mTexture->getWidth());
		shape.AddDim(3);

		//tf_thread->join();
		tf_thread = nullptr;
		resize(mTexture->getSize());

	}
	void ciTensorflow::tf_infer()
	{
		// Convert mat to tensor
		Tensor tensor = Tensor(tensorflow::DT_FLOAT, shape);
		Status readTensorStatus = readTensorFromMat(output, tensor);
		if (!readTensorStatus.ok()) {
			ci::app::console() << "ERROR: Mat->Tensor conversion failed: " << readTensorStatus << std::endl;
			return;
		}

		double thresholdScore = 0.3;
		double thresholdIOU = 0.5;

		// Run the graph on tensor

		outputs.clear();
		Status runStatus = session->Run({ { inputLayer, tensor } }, outputLayer, {}, &outputs);
		if (!runStatus.ok()) {
			ci::app::console() << "ERROR: Running model failed: " << runStatus << std::endl;
			return;
		}
		// Extract results from the outputs vector
		tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
		tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
		tensorflow::TTypes<float, 3>::Tensor boxes = outputs[0].flat_outer_dims<float, 3>();

		std::vector<size_t> goodIdxs = filterBoxes(scores, boxes, thresholdIOU, thresholdScore);
		face_detected = goodIdxs.size();

		// Get bboxes coord
		gXminBuffer.resize(0);
		gYminBuffer.resize(0);
		gXmaxBuffer.resize(0);
		gYmaxBuffer.resize(0);
		GetBoundingBoxesOnImage(scores, classes, boxes, goodIdxs, gXminBuffer, gYminBuffer, gXmaxBuffer, gYmaxBuffer, gScoreBuffer);
		gLockBuffer.lock();
		std::swap(gXmin, gXminBuffer);
		std::swap(gYmin, gYminBuffer);
		std::swap(gXmax, gXmaxBuffer);
		std::swap(gYmax, gYmaxBuffer);
		std::swap(gScore, gScoreBuffer);
		gLockBuffer.unlock();
		gInferReady = true;

	}
	void ciTensorflow::tf_pixtopixinfer()
	{
		// Convert mat to tensor
		Tensor tensor = Tensor(tensorflow::DT_FLOAT, shape);
		Status readTensorStatus = readTensorFromMat(output, tensor);
		if (!readTensorStatus.ok()) {
			ci::app::console() << "ERROR: Mat->Tensor conversion failed: " << readTensorStatus << std::endl;
			return;
		}

		double thresholdScore = 0.3;
		double thresholdIOU = 0.5;

		// Run the graph on tensor
		outputs.clear();
		Status runStatus = session->Run({ { inputLayer, tensor } }, outputLayer, {}, &outputs);
		if (!runStatus.ok()) {
			ci::app::console() << "ERROR: Running model failed: " << runStatus << std::endl;
			return;
		}
		// Extract results from the outputs vector
		tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
		tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
		tensorflow::TTypes<float, 3>::Tensor boxes = outputs[0].flat_outer_dims<float, 3>();

		std::vector<size_t> goodIdxs = filterBoxes(scores, boxes, thresholdIOU, thresholdScore);
		face_detected = goodIdxs.size();

		// Get bboxes coord
		gXminBuffer.resize(0);
		gYminBuffer.resize(0);
		gXmaxBuffer.resize(0);
		gYmaxBuffer.resize(0);
		GetBoundingBoxesOnImage(scores, classes, boxes, goodIdxs, gXminBuffer, gYminBuffer, gXmaxBuffer, gYmaxBuffer, gScoreBuffer);
		gLockBuffer.lock();
		std::swap(gXmin, gXminBuffer);
		std::swap(gYmin, gYminBuffer);
		std::swap(gXmax, gXmaxBuffer);
		std::swap(gYmax, gYmaxBuffer);
		std::swap(gScore, gScoreBuffer);
		gLockBuffer.unlock();
		gInferReady = true;

	}

	void ciTensorflow::updateTensor()
	{
		
	//	cvtColor(input, input, cv::COLOR_BGR2RGB);


		//fps = getFrameRate();

		if (gInferReady)
		{
			if (tf_thread)
			{
				tf_thread->join();
			}
			input.copyTo(output);
			tf_thread.reset(new std::thread(&ciTensorflow::tf_infer, this));
			gInferReady = false;
		}
		//ci::app::console() << "got to here.." << std::endl;
		//if (iFrame == 0)
		{
//			tf_thread->join();
//			tf_thread = nullptr;
		}
	//	iFrame++;
	//	ci::app::console() << "got to hereA" << std::endl;
		// draw latest result
	//	cvtColor(input, input, cv::COLOR_BGR2RGB);
	//	ci::app::console() << "got to hereB" << std::endl;
		gLockBuffer.lock();
		for (int i = 0; i < gXmin.size(); i++)
		{
		drawBoundingBoxOnImage(input, gYmin[i], gXmin[i], gYmax[i], gXmax[i], gScore[i], "face", true);
		}
		gLockBuffer.unlock();
		cv::putText(input, std::to_string(face_detected) + " faces", cv::Point(0, input.rows), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255));
		cv::putText(input, std::to_string(ci::app::getFrameRate()).substr(0, 3) + " fps", cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255));
	//	ci::app::console() << "got to here" << std::endl;
		//cv::imshow("Result", input);
		

	}

	void ciTensorflow::resize(ci::vec2 size)
	{
		shape = tensorflow::TensorShape();
		shape.AddDim(1);
		shape.AddDim((int64)size.x);
		shape.AddDim((int64)size.y);
		shape.AddDim(3);
	}




	ciTensorflow::~ciTensorflow() {

	}
}