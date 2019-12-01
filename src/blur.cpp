#include <vector>
#include "pixel.h"
#include "blur.h"
#include <cmath>
#include <iostream>

//CEILING HALF
#define HALF_KERN_SIZE 2
#define KERNEL_SIZE 2 * HALF_KERN_SIZE + 1
static void fillKernel(double sigma, double (&kernel)[KERNEL_SIZE][KERNEL_SIZE]){
  double r;
  double denom = 2.0 * sigma * sigma;

  double sum = 0.0;
  for(int x = - HALF_KERN_SIZE; x< HALF_KERN_SIZE;x++){
    for(int y = -HALF_KERN_SIZE; y<HALF_KERN_SIZE;y++){
      r = sqrt(x*x + y*y);
      kernel[x+HALF_KERN_SIZE][y+HALF_KERN_SIZE] = (exp(-(r*r)/denom))/(M_PI * denom);
      sum += kernel[x+HALF_KERN_SIZE][y+HALF_KERN_SIZE];
    }
  }
  for(int i=0;i<KERNEL_SIZE;i++){
    for(int j = 0;j<KERNEL_SIZE;j++){
      kernel[i][j] = kernel[i][j]/sum;
    }
  }
  return;
}

static void printKernel(const double (&kernel)[KERNEL_SIZE][KERNEL_SIZE]){
  for(int i=0;i<KERNEL_SIZE;i++){
    for(int j=0;j<KERNEL_SIZE;j++){
      std::cout<<kernel[i][j]<<"\t";
    }
    std::cout<<std::endl;
  }
}
static Pixel convolve(const std::vector<std::vector<Pixel>> &pixels,int i ,
    int j, const double kernel[KERNEL_SIZE][KERNEL_SIZE]){
  Pixel newP;
  newP.r=0;
  newP.g=0;
  newP.b=0;
  for(int x=0;x<KERNEL_SIZE;x++){
    for(int y=0;y<KERNEL_SIZE;y++){
      Pixel p =  pixels[i-HALF_KERN_SIZE + x][j - HALF_KERN_SIZE + y];
      double weight = kernel[x][y];
      newP.r += p.r * weight;
      newP.g += p.g * weight;
      newP.b += p.b * weight;
    }
  }
  return newP;
}
void blur(std::vector<std::vector<Pixel>> &pixels,double sigma){
  double kernel[KERNEL_SIZE][KERNEL_SIZE];
  std::vector<std::vector<Pixel>> newPixels(pixels.size(),std::vector<Pixel>(pixels[0].size()));
  fillKernel(sigma,kernel);
  printKernel(kernel);
  for(int i=HALF_KERN_SIZE;i<(int)pixels.size()-HALF_KERN_SIZE;i++){
    for(int j=HALF_KERN_SIZE;j<(int)pixels[0].size()-HALF_KERN_SIZE;j++){
      newPixels[i][j] = convolve(pixels, i,j,kernel);
    }
    //std::cout<<"iterations"<<std::endl;
  }

  std::cout<<"finished convolving";
  for(int i=HALF_KERN_SIZE;i<(int)pixels.size() - HALF_KERN_SIZE;i++){
    for(int j = HALF_KERN_SIZE;j<(int)pixels[0].size() - HALF_KERN_SIZE;j++){
      pixels[i][j] = newPixels[i][j];
    }
  }
  std::cout<<"about to return";
  return;
}
