#ifndef HOUGH_H_
#define HOUGH_H_

#include <vector>

namespace keymolen {

	class Hough {
	public:
		Hough();
		virtual ~Hough();
	public:
		int Transform(unsigned char* img_data, int w, int h);
		std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > GetLines(int threshold);
		const unsigned int* GetAccu(int *w, int *h);
	private:
		unsigned int* _accu;
		int _accu_w;
		int _accu_h;
		int _img_w;
		int _img_h;
	};

}

#endif /* HOUGH_H_ */
