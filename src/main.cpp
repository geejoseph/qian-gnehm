#include "preprocess.h"
#include <iostream>
#include <tuple>
#include <vector>
#include "pixel.h"
#include "imgSeg.h"
#include "blur.h"
#include <string>
#include <cstdlib>
#include <fstream>
#include <omp.h>

using namespace std;


static void savePPM(const vector<vector<Pixel>> pixels,
    const string name, int width, int height){
  Pixel temp[height][width];
  for(int i=0;i<height;i++){
    for(int j=0;j<width;j++){
      Pixel newPixel;
      //s
      Pixel temp2 = pixels[i][j];
      newPixel.r = temp2.r;
      newPixel.g = temp2.g;
      newPixel.b = temp2.b;
      temp[i][j] = newPixel;
      //printf("r:%d g:%d b:%d\n",(int)newPixel.r,(int)newPixel.g,(int)newPixel.b);
    }
  }
  //cout<<"about to save"<<endl;
  ofstream file(name,ios::out | ios::binary);
  file<<"P6\n" << width<<" "<<height<<"\n"<<255<<"\n";
  //cout<<"check"<<endl;
  file.write((char *)(&temp[0][0]),width*height*sizeof(Pixel));
  return;
}

int main(int argc, char **argv){
  string filename = std::string(argv[1]);
  //cout<<filename<<endl;
  //cout<<"before preprocess"<<endl;
  tuple<vector<vector<Pixel>>,int,int> imgData =
    preprocess(filename);
  int height = get<1>(imgData);
  int width = get<2>(imgData);
  cout<<"width: "<<width<< ", height: "<<height<<endl;
  vector<vector<Pixel>> img = get<0>(imgData);
  /*savePPM(img, "img/Cathedral_of_Learning.ppm",width,height);
  return 0;*/
  double sigma = 0.8;
  blur(img, sigma);
  //cout<<"done blurring"<<endl;
  //savePPM(img,"img/einstein.ppm",width,height);
  //return 0;
  process(img,width,height);
  //postprocess(img,width,height);
  //cout<<"after preprocess"<<endl;
  //cout<<"new width"<<img[0].size()<<" new height "<<img.size()<<endl;
  savePPM(img, "result.ppm",width,height);
  //savePPM(img,"img/einstein.ppm",width,height);
  //cout<<"cleanup"<<endl;
  return 0;
}
