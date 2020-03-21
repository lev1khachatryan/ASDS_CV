#include "hough.h"
#include <cmath>
#include <iostream>
#include <string.h>
#include <stdlib.h>


#define DEG2RAD 0.017453293f

namespace keymolen {

	Hough::Hough():_accu(0), _accu_w(0), _accu_h(0), _img_w(0), _img_h(0)
	{

	}

	Hough::~Hough() {
		if(_accu)
			free(_accu);
	}


	int Hough::Transform(unsigned char* img_data, int w, int h)
	{
		_img_w = w;
		_img_h = h;

		//Create the accu
		double hough_h = ((sqrt(2.0) * (double)(h>w?h:w)) / 2.0);
		_accu_h = hough_h * 2.0; // -r -> +r
		_accu_w = 180;

		_accu = (unsigned int*)calloc(_accu_h * _accu_w, sizeof(unsigned int));

		double center_x = w/2;
		double center_y = h/2;


		for(int y=0;y<h;y++)
		{
			for(int x=0;x<w;x++)
			{
				if( img_data[ (y*w) + x] > 250 )
				{
					for(int t=0;t<180;t++)
					{
						double r = ( ((double)x - center_x) * cos((double)t * DEG2RAD)) + (((double)y - center_y) * sin((double)t * DEG2RAD));
						_accu[ (int)((round(r + hough_h) * 180.0)) + t]++;
					}
				}
			}
		}

		return 0;
	}

	std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > Hough::GetLines(int threshold)
	{
		std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines;

		if(_accu == 0)
			return lines;

		for(int r=0;r<_accu_h;r++)
		{
			for(int t=0;t<_accu_w;t++)
			{
				if((int)_accu[(r*_accu_w) + t] >= threshold)
				{
					//Is this point a local maxima (9x9)
					int max = _accu[(r*_accu_w) + t];
					for(int ly=-4;ly<=4;ly++)
					{
						for(int lx=-4;lx<=4;lx++)
						{
							if( (ly+r>=0 && ly+r<_accu_h) && (lx+t>=0 && lx+t<_accu_w)  )
							{
								if( (int)_accu[( (r+ly)*_accu_w) + (t+lx)] > max )
								{
									max = _accu[( (r+ly)*_accu_w) + (t+lx)];
									ly = lx = 5;
								}
							}
						}
					}
					if(max > (int)_accu[(r*_accu_w) + t])
						continue;


					int x1, y1, x2, y2;
					x1 = y1 = x2 = y2 = 0;

					if(t >= 45 && t <= 135)
					{
						//y = (r - x cos(t)) / sin(t)
						x1 = 0;
						y1 = ((double)(r-(_accu_h/2)) - ((x1 - (_img_w/2) ) * cos(t * DEG2RAD))) / sin(t * DEG2RAD) + (_img_h / 2);
						x2 = _img_w - 0;
						y2 = ((double)(r-(_accu_h/2)) - ((x2 - (_img_w/2) ) * cos(t * DEG2RAD))) / sin(t * DEG2RAD) + (_img_h / 2);
					}
					else
					{
						//x = (r - y sin(t)) / cos(t);
						y1 = 0;
						x1 = ((double)(r-(_accu_h/2)) - ((y1 - (_img_h/2) ) * sin(t * DEG2RAD))) / cos(t * DEG2RAD) + (_img_w / 2);
						y2 = _img_h - 0;
						x2 = ((double)(r-(_accu_h/2)) - ((y2 - (_img_h/2) ) * sin(t * DEG2RAD))) / cos(t * DEG2RAD) + (_img_w / 2);
					}

					lines.push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1,y1), std::pair<int, int>(x2,y2)));

				}
			}
		}
		
		std::cout << "lines: " << lines.size() << " " << threshold << std::endl;
		return lines;
	}

	const unsigned int* Hough::GetAccu(int *w, int *h)
	{
		*w = _accu_w;
		*h = _accu_h;

		return _accu;
	}
}
