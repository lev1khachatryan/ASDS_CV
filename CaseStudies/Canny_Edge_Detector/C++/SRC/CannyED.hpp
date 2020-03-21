#ifndef __CANNY_EDGE__
#define __CANNY_EDGE__

#include <opencv2\opencv.hpp>

using namespace cv;

class CannyED
{
	public:
		CannyED():ks32NonMaxFiltLen(5){};                             /* constructor */
		CannyED(Mat &sMatInput, Mat &sMatOutput, double dThreshold1, double dThreshold2);
		~CannyED(){};

	private:
		const int ks32NonMaxFiltLen;//Filter size for non-max suppression
		Mat sMatCannyEdge;
		/* ====================  METHODS       ======================================= */
		void NonMaxSuppression(Mat &sMatGrad, Mat &sMatThinEdge, Mat &sMatEdgeOrientation);
		void Hysteresis(Mat &sMatThinEdge, Mat &sMatEdgeMap, double dThOne, double dThTwo);
		void Trace(Mat &sMatThinEdge, Mat &sMatEdgeMap, int s32Y, int s32X, double dThOne);

}; /* -----  end of class CannyED  ----- */
#endif
