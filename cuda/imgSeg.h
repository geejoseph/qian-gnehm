#ifndef __IMGSEG_H_
#define __IMGSEG_H_

#include <iostream>
#include <vector>
#include "pixel.h"

void seq_process(std::vector<Pixel>& pixels,int width, int height);

void cu_process(std::vector<Pixel> &pixels,int width, int height);
#endif
