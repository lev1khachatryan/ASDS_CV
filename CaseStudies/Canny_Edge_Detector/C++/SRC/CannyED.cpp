#include <CannyED.hpp>
#include <iostream>

using namespace std;

#pragma comment(lib, "opencv_core300d.lib")
#pragma comment(lib, "opencv_highgui300d.lib")
#pragma comment(lib, "opencv_imgcodecs300d.lib")
#pragma comment(lib, "opencv_imgproc300d.lib")

/**<
@param   sMatInput - Input Gray image
@param  sMatOutput - Edge Map
@param dThreshold1 - Hysteresis low threshold
@param dThreshold2 - Hysteresis high threshold
*/

CannyED::CannyED(Mat &sMatInput, Mat &sMatOutput, double dThreshold1, double dThreshold2):ks32NonMaxFiltLen(5)
{
	Mat sMatInputGauss;
	Mat sMatSobelX;
	Mat sMatSobelY;
	Mat sMatSobelGrad;
	Mat sMatEdgeThin;
	Mat sMatEdgeFinal;

	Mat sMatEdgeOrientation = Mat(Size(sMatInput.cols, sMatInput.rows), CV_16UC1);
	sMatCannyEdge = Mat::zeros(Size(sMatInput.cols, sMatInput.rows), CV_8UC1);

	//Sobel-X
	int ks32KernelX[3][3] = {
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1
	};

	//Sobel-Y
	int ks32KernelY[3][3] = {
		-1, -2, -1,
		 0, 0, 0,
		+1, +2, 1
	};

	Mat sMatKernelX = Mat(Size(3, 3), CV_32SC1, ks32KernelX);
	Mat sMatKernelY = Mat(Size(3, 3), CV_32SC1, ks32KernelY);

	GaussianBlur(sMatInput, sMatInputGauss, Size(5, 5), 10);
	sMatInputGauss = sMatInput;
	filter2D(sMatInputGauss, sMatSobelX, CV_32FC1, sMatKernelX, Point(-1, -1));
	filter2D(sMatInputGauss, sMatSobelY, CV_32FC1, sMatKernelY, Point(-1, -1));

	Mat sMatSobel = sMatSobelX.mul(sMatSobelX) + sMatSobelY.mul(sMatSobelY);
	sqrt(sMatSobel, sMatSobelGrad);

	sMatSobelGrad.convertTo(sMatSobelGrad, CV_8UC1);

	for (int i = 0; i < sMatSobelX.rows; i++){
		for (int j = 0; j < sMatSobelX.cols; j++){
			float fValX  = sMatSobelX.at<float>(i, j);
			float fValY  = sMatSobelY.at<float>(i, j);
			float fAngle = fastAtan2(fValY, fValX);

			if (fAngle > 180)
				fAngle = fAngle - 180;

			if ((fAngle > 0 && fAngle <= 22.5) || (fAngle > 157.5 && fAngle <= 180))
				fAngle = 0;
			else if (fAngle > 22.5 && fAngle <= 67.5)
				fAngle = 45;
			else if (fAngle > 67.5 && fAngle <= 112.5)
				fAngle = 90;
			else if (fAngle > 112.5 && fAngle <= 157.5)
				fAngle = 135;

			sMatEdgeOrientation.at<unsigned short>(i, j) = (unsigned short)fAngle;
		}
	}

	NonMaxSuppression(sMatSobelGrad, sMatEdgeThin, sMatEdgeOrientation);
	sMatOutput = sMatEdgeThin;
	Hysteresis(sMatEdgeThin, sMatCannyEdge, dThreshold1, dThreshold2);
	sMatOutput = sMatCannyEdge;

	return;
}

/**<
@return void
@param   sMatGrad - Sobel Edge Map
@param  sMatOutputGrad - Output Edge Thin Map
@param  sMatEdgeOrientation - Orientation of edge
*/
void CannyED::NonMaxSuppression(Mat &sMatGrad, Mat &sMatOutputGrad, Mat &sMatEdgeOrientation)
{
	Mat             sMatGradPad;
	vector<Point2i> vecIndex;
	int             s32Pad      = ks32NonMaxFiltLen >> 1;
	Point pOne, pTwo, pThree, pFour;
	vector<uchar> vecPixel(ks32NonMaxFiltLen);

	copyMakeBorder(sMatGrad, sMatGradPad, s32Pad, s32Pad, s32Pad, s32Pad, BORDER_REPLICATE);
	sMatOutputGrad.create(Size(sMatGrad.cols, sMatGrad.rows), CV_8UC1);

	for (int i = s32Pad; i < sMatGradPad.rows - s32Pad; i++){
		for (int j = s32Pad; j < sMatGradPad.cols - s32Pad; j++){
			int ii = i - s32Pad;
			int jj = j - s32Pad;
			unsigned short u16Angle = sMatEdgeOrientation.at<unsigned short>(ii, jj);
			switch ( u16Angle ) {
				case 0:	
					pOne   = (Point(0, 1)); pTwo   = (Point(0, 2)); pThree = (Point(-1, 0)); pFour  = (Point(-2, 0));
					break;
				case 45:	
					pOne   = (Point(-1, -1)); pTwo   = (Point(-2, -2)); pThree = (Point(1, 1)); pFour  = (Point(2, 2));
					break;
				case 90:	
					pOne   = (Point(-1, 0)); pTwo   = (Point(-2, 0)); pThree = (Point(1, 0)); pFour  = (Point(2, 0));
					break;
				case 135:	
					pOne   = (Point(-1, 1)); pTwo   = (Point(-2, 2)); pThree = (Point(1, -1)); pFour  = (Point(2, -2));
					break;
				default:	
					break;
			}				/* -----  end switch  ----- */
			
			//four neighboring pixels
			vecPixel.at(0) = sMatGradPad.at<uchar>(i + pTwo.x   , j + pTwo.y);
			vecPixel.at(1) = sMatGradPad.at<uchar>(i + pOne.x   , j + pOne.y);
			vecPixel.at(2) = sMatGradPad.at<uchar>(i , j);
			vecPixel.at(3) = sMatGradPad.at<uchar>(i + pThree.x , j + pThree.y);
			vecPixel.at(4) = sMatGradPad.at<uchar>(i + pFour.x  , j + pFour.y);

			vector<uchar>::iterator itr = max_element(vecPixel.begin(), vecPixel.end());
			int s32Index = itr - vecPixel.begin();
			// Index of the current pixel in stored at index=2
			if ( 2 != s32Index ) {
				sMatOutputGrad.at<uchar>(ii, jj) = 0;
			} else {
				sMatOutputGrad.at<uchar>(ii, jj) = sMatGradPad.at<uchar>(i, j);
			}

		}
	}
}

/**<
@return void
@param   sMatThinEdge - Edge map after NMS
@param  sMatOutEdgeMap - Final Edge map
@param  dThOne - Lower hysteresis threshold
@param  dThTwo - Upper hysteresis threshold
*/
#if 0
void CannyED::Hysteresis(Mat &sMatGrad, Mat &sMatOutputGrad, double dThOne, double dThTwo)
{
#define NUM_NEIGHBORS (8)

	Mat sMatMaskOne, sMatMaskTwo, sMatMaskOneTwo;
	const int ky[NUM_NEIGHBORS] = {-1, -1, -1, 0, 0, 1, 1, 1};
	const int kx[NUM_NEIGHBORS] = {-1, 0, -1, -1, 1, -1, 0, 1};

	// Pixels < dThOne are eliminated
	sMatMaskOne = (sMatGrad < dThOne & sMatGrad > 0);
	sMatGrad.setTo(0, sMatMaskOne);

	sMatMaskTwo    = (sMatGrad >= dThTwo);
	sMatMaskOneTwo = (sMatGrad < dThTwo & sMatGrad >=dThOne);

	sMatMaskTwo.setTo(2, sMatMaskTwo);
	sMatMaskOneTwo.setTo(1, sMatMaskOneTwo);

	sMatMaskTwo = sMatMaskTwo + sMatMaskOneTwo;
	copyMakeBorder(sMatMaskTwo, sMatMaskTwo, 1, 1, 1, 1, BORDER_REPLICATE);

	for(int i = 1; i < sMatMaskTwo.rows - 1; i++){
		for (int j = 1; j < sMatMaskTwo.cols - 1; j++){
			uchar u8PixVal = sMatMaskTwo.at<uchar>(i, j);
			vector<uchar> vNeighbors(NUM_NEIGHBORS);
			if(1 == u8PixVal){
				vNeighbors.at(0) = sMatMaskTwo.at<uchar>(i - ky[0], j - kx[0]);
				vNeighbors.at(1) = sMatMaskTwo.at<uchar>(i - ky[1], j - kx[1]);
				vNeighbors.at(2) = sMatMaskTwo.at<uchar>(i - ky[2], j - kx[2]);
				vNeighbors.at(3) = sMatMaskTwo.at<uchar>(i - ky[3], j - kx[3]);
				vNeighbors.at(4) = sMatMaskTwo.at<uchar>(i - ky[4], j - kx[4]);
				vNeighbors.at(5) = sMatMaskTwo.at<uchar>(i - ky[5], j - kx[5]);
				vNeighbors.at(6) = sMatMaskTwo.at<uchar>(i - ky[6], j - kx[6]);
				vNeighbors.at(7) = sMatMaskTwo.at<uchar>(i - ky[7], j - kx[7]);
				if(find(vNeighbors.begin(), vNeighbors.end(), 2) != vNeighbors.end())
					sMatMaskTwo.at<uchar>(i, j) = 2;
				//else
				//	sMatMaskTwo.at<uchar>(i, j) = 0;

			}
		}
	}

	sMatMaskTwo = sMatMaskTwo(Range(1, sMatMaskTwo.rows - 1), Range(1, sMatMaskTwo.cols - 1));
	sMatMaskTwo = (sMatMaskTwo == 2);

	sMatGrad.copyTo(sMatOutputGrad, sMatMaskTwo);
	sMatOutputGrad = (sMatOutputGrad != 0);
}
#else
void CannyED::Hysteresis(Mat &sMatThinEdge, Mat &sMatOutEdgeMap, double dThOne, double dThTwo)
{
	int s32Height = sMatThinEdge.rows;
	int s32Width  = sMatThinEdge.cols;

	for (int i = 1; i < s32Height - 1; i++){
		for (int j = 1; j < s32Width - 1; j++){
			uchar u8PixVal = sMatThinEdge.at<uchar>(i, j);
			if ( u8PixVal > dThTwo){
				Trace(sMatThinEdge, sMatOutEdgeMap, i, j, dThOne);
			}
		}
	}

}
#endif

void CannyED::Trace(Mat &sMatThinEdge, Mat &sMatEdgeMap, int s32Y, int s32X, double dThOne)
{
#define NUM_NEIGHBORS (8)

	const int ky[NUM_NEIGHBORS] = {-1, -1, -1, 0, 0, 1, 1, 1};
	const int kx[NUM_NEIGHBORS] = {-1, 0, -1, -1, 1, -1, 0, 1};

	int s32Width = sMatThinEdge.cols;
	int s32Height = sMatThinEdge.rows;
	if(sMatEdgeMap.at<uchar>(s32Y, s32X) == 0) {
		sMatEdgeMap.at<uchar>(s32Y, s32X) = 255;
		for(int i = 0; i < NUM_NEIGHBORS; i++){
			int ii = s32Y + ky[i];
			int jj = s32X + kx[i];
			if(ii >= 0 && jj >= 0 && ii < s32Height && jj < s32Width){
				uchar u8PixValOne = sMatThinEdge.at<uchar>(s32Y + ky[i], s32X + kx[i]);
				uchar u8PixValTwo = sMatEdgeMap.at<uchar>(s32Y + ky[i], s32X + kx[i]);
				if(u8PixValOne > dThOne && u8PixValTwo != 255){
					Trace(sMatThinEdge, sMatEdgeMap, ii, jj, dThOne);
				}
			}
		}
	}
}