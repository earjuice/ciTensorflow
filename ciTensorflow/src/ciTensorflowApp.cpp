#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/ImageIo.h"


#include "CiSpoutIn.h"
#include "tf.h"


using namespace ci;
using namespace ci::app;
//using namespace msa::tf;

class ciTensorflowApp : public App {
public:
	void setup() override;
	void update() override;
	void draw() override;
	void resize() override;
	void updateTensor();

	gl::TextureRef	mTexture;
	ci::SpoutIn	mSpoutIn;
	tf::ciTensorflow ciTF;

};


void ciTensorflowApp::setup()
{


}


void ciTensorflowApp::resize()
{
	ci::app::setWindowSize(mSpoutIn.getSize());
	mTexture = mSpoutIn.receiveTexture();
	ciTF.resize(mSpoutIn.getSize());
}
void ciTensorflowApp::update()
{
	if (mSpoutIn.getSize() != app::getWindowSize()) {
		resize();
	}
	auto tx = Surface8u(mSpoutIn.receiveTexture()->createSource(), SurfaceConstraintsDefault(), false);

	ciTF.input = cv::Mat(ci::toOcv(tx));

	ciTF.updateTensor();

	mTexture = ci::gl::Texture::create(ci::fromOcv(ciTF.input));
}

void ciTensorflowApp::draw()
{

	gl::clear();
	gl::draw(mTexture);


}

CINDER_APP(ciTensorflowApp, RendererGl)
