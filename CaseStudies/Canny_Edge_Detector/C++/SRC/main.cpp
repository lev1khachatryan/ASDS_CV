#include <string>
#include <iostream>
#include <opencv2\opencv.hpp>
#include <CannyED.hpp>

#pragma comment(lib, "opencv_core300d.lib") // core functionalities
#pragma comment(lib, "opencv_highgui300d.lib") //GUI
#pragma comment(lib, "opencv_imgcodecs300d.lib")
#pragma comment(lib, "opencv_imgproc300d.lib") // Histograms, Edge detection

using namespace cv;
using namespace std;

#define PI (3.1412)

/* 
* ===  FUNCTION  ======================================================================
*         Name:  main
*  Description:  
* =====================================================================================
*/

int main ( int argc, char *argv[] )
{
	Mat sMatInput = imread("..\\flower.jpg", IMREAD_GRAYSCALE);
	Mat sMatOutput;

	String strWinIn = "Input";
	String strWinOut = "Output";

	CannyED(sMatInput, sMatOutput, 10, 40);
	namedWindow(strWinIn, WINDOW_NORMAL);
	namedWindow(strWinOut, WINDOW_NORMAL);
	imshow(strWinIn, sMatInput);
	imshow(strWinOut, sMatOutput);

	imwrite("Input.jpg", sMatInput);
	imwrite("OutputEdge.jpg", sMatOutput);
	waitKey(0);
	return EXIT_SUCCESS;

} /* ----------  end of function main  ---------- */

