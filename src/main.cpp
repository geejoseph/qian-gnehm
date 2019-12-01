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

using namespace std;


static void savePPM(const vector<vector<Pixel>> pixels,const string name, int width, int height){
  Pixel temp[height][width];
  for(int i=0;i<height;i++){
    for(int j=0;j<width;j++){
      Pixel newPixel;
      Pixel temp2 = pixels[i][j];
      newPixel.r = temp2.r;
      newPixel.g = temp2.g;
      newPixel.b = temp2.b;
      temp[i][j] = newPixel;
    }
  }
  ofstream file(name,ios::out | ios::binary);
  file<<"P6\n" << width<<" "<<height<<"\n"<<255<<"\n";
  file.write((char *)(&temp[0][0]),width*height*sizeof(Pixel));

}
int main(){
  
  cout<<"before preprocess"<<endl;
  tuple<vector<vector<Pixel>>,int,int> imgData = 
    preprocess("img/einstein.jpg");
  int height = get<1>(imgData);
  int width = get<2>(imgData);
  vector<vector<Pixel>> img = get<0>(imgData);
  /*savePPM(img, "img/Cathedral_of_Learning.ppm",width,height);
  return 0;*/
  blur(img,0.8);
  cout<<"done blurring"<<endl;
  savePPM(img,"img/einstein.ppm",width,height);
  return 0;
  process(img,width,height);

  //postprocess(img,width,height);
  cout<<"after preprocess"<<endl;
  savePPM(img,"imt/Cathedral_of_Learning.ppm",width,height);
  return 0;
}
