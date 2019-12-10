#ifndef __BLUR_H_
#define __BLUR_H_

#include "pixel.h"
#include <vector>

void blur(std::vector<Pixel> &pixels,double sigma,int width,int height);


#endif
