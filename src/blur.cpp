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
static Pixel convolve(const std::vector<Pixel> &pixels,int i ,
    int j, const double kernel[KERNEL_SIZE][KERNEL_SIZE],int width,int height){
  Pixel newP;
  newP.r=0;
  newP.g=0;
  newP.b=0;
  for(int x=0;x<KERNEL_SIZE;x++){
    for(int y=0;y<KERNEL_SIZE;y++){
      Pixel p =  pixels[(i-HALF_KERN_SIZE + x)*width + j - HALF_KERN_SIZE + y];
      double weight = kernel[x][y];
      newP.r += p.r * weight;
      newP.g += p.g * weight;
      newP.b += p.b * weight;
    }
  }
  return newP;
}
void blur(std::vector<Pixel> &pixels,double sigma,int width, int height){
  double kernel[KERNEL_SIZE][KERNEL_SIZE];
  std::vector<Pixel> newPixels(width*height);
  fillKernel(sigma,kernel);
  printKernel(kernel);
  for(int i=HALF_KERN_SIZE;i<height-HALF_KERN_SIZE;i++){
    for(int j=HALF_KERN_SIZE;j<width-HALF_KERN_SIZE;j++){
      newPixels[i*width+j] = convolve(pixels, i,j,kernel,width,height);
    }
    //std::cout<<"iterations"<<std::endl;
  }

  for(int i=HALF_KERN_SIZE;i<height - HALF_KERN_SIZE;i++){
    for(int j = HALF_KERN_SIZE;j<width - HALF_KERN_SIZE;j++){
      pixels[i*width+j] = newPixels[i*width+j];
    }
  }
  return;
}
