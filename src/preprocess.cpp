#include <iostream>
#include <boost/gil/extension/io/jpeg_io.hpp>
#include <string>
#include <tuple>
#include <vector>

std::tuple<std::vector<std::vector<std::vector<int>>>,int,int> preprocess(std::string image){

  boost::gil::rgb8_image_t img;
  boost::gil::jpeg_read_image(image,img);
  int height = img.height();
  int width = img.width();
  std::vector<std::vector<std::vector<int>>> pixels(height, 
      std::vector<std::vector<int>>(width, std::vector<int>(3)));

  for(int row = 0; row<height; row++){
    for(int col = 0; col<width; col++){
      boost::gil::rgb8_pixel_t px = *const_view(img).at(row,col);
      pixels[row][col][0] = (int)px[0];
      pixels[row][col][1] = (int)px[1];
      pixels[row][col][2] = (int)px[2];
    }
  }
  return std::make_tuple(pixels,height,width);
}
