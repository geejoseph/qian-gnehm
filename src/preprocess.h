#ifndef __PREPROCESS_H_
#define __PREPROCESS_H_

#include <vector>
#include <tuple>
#include "pixel.h"

std::tuple<std::vector<std::vector<Pixel>>,int,int> preprocess(std::string image);

#endif
