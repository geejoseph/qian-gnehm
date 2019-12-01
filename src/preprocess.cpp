#include <iostream>
#include <boost/gil/extension/io/jpeg_io.hpp>
#include <string>
#include <tuple>
#include <vector>
#include "preprocess.h"

std::tuple<std::vector<std::vector<Pixel>>,int,int> preprocess(std::string image){

  boost::gil::rgb8_image_t img;
  boost::gil::jpeg_read_image(image,img);
  int height = img.height();
  int width = img.width();
  std::vector<std::vector<Pixel>> pixels(height, 
      std::vector<Pixel>(width));

  for(int row = 0; row<height; row++){
    for(int col = 0; col<width; col++){
      boost::gil::rgb8_pixel_t px = *const_view(img).at(col,row);
      Pixel temp;
      temp.r = (int)px[0];
      temp.g = (int)px[1];
      temp.b = (int)px[2];
      pixels[row][col] = temp;//Pixel((int)px[0],(int)px[1],(int)px[2]);
    }
  }
  return std::make_tuple(pixels,height,width);
}
