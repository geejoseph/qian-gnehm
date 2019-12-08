#include <iostream>
#include <string>
#include <vector>
#include "pixel.h"
#include "imgSeg.h"
#include <cstdint>
#include <assert.h>
#include <cuda.h>
#include <driver_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16
int gwidth;
int gheight;
__constant__ int global_width;
__constant__ int global_height;

__device__ __inline__ bool mergeCriterion(Pixel p1, Pixel p2, int t){
  return ((int)p1.r - (int)p2.r)*((int)p1.r - (int)p2.r) + ((int)p1.g - (int)p2.g)*((int)p1.g -(int) p2.g) +
    ((int)p1.b - (int)p2.b) * ((int)p1.b - (int)p2.b) < t*t;
}

__device__ Pixel newColor(Pixel A, Pixel B, int sizeA, int sizeB){
  int totalSize = sizeA + sizeB;
  Pixel newP;
  newP.r = (uint8_t)(((int)A.r * sizeA +(int) B.r * sizeB)/(totalSize));
  newP.g = (uint8_t)(((int)A.g * sizeA + (int)B.g * sizeB)/(totalSize));
  newP.b = (uint8_t)(((int)A.b * sizeA + (int)B.b * sizeB)/(totalSize));
  return newP;
}

__device__ int find(int *next,int srow,int scol){
  //int row = srow;
  //int col = scol;
  int pos = srow * global_width + scol;
  while(1){
    //std::cout<<"in find"<<std::endl;
    int index = next[pos];
    if(index == -1){
      next[pos] = pos;
      next[srow * global_width + scol] = pos;
      return pos;
    }

    if(index == pos){
      next[srow* global_width + scol] = index;
      return index;
    }
    pos = index;
  }
}

__device__ void verify_edge(Pixel *pixels_cu, int *next_cu,
    int *size_cu, int col1, int row1, int col2, int row2) {
  //sanity check
  //assert(col1< global_width && col1>=0 && col2 < global_width && col2 >= 0 && row1 < global_height && row1 >=0
   //   && row2 < global_height && row2 >=0);

  int aIndex = find(next_cu,row1,col1);
  //int aRow = aIndex / global_width;
  //int aCol = aIndex % global_width;

  int bIndex = find(next_cu,row2,col2);
  //int bRow = bIndex / global_width;
  //int bCol = bIndex % global_width;

  if(aIndex !=  bIndex){
    Pixel A = pixels_cu[aIndex];
    int aSize = size_cu[aIndex];

    Pixel B = pixels_cu[bIndex];
    int bSize = size_cu[bIndex];

    if(mergeCriterion(A,B,30)){
      //printf("A r: %d g: %d b: %d B r: %d g: %d b: %d\n",A.r,A.g,A.b,B.r,B.g,B.b);
      if(aSize>bSize){
        pixels_cu[aIndex] = newColor(A,B, aSize, bSize);
        //printf("newCol r: %d g: %d b: %d\n",pixels[aRow][aCol].r,pixels[aRow][aCol].g,pixels[aRow][aCol].b);
        next_cu[bIndex] = aIndex;
        size_cu[aIndex] += bSize;
      }
      else{
        pixels_cu[bIndex] = newColor(A, B, aSize, bSize);

        //printf("newCol r: %d g: %d b: %d\n",pixels[aRow][aCol].r,pixels[aRow][aCol].g,pixels[aRow][aCol].b);
        next_cu[aIndex] = bIndex;
        size_cu[bIndex] += aSize;
      }
    }
  }
  return;

}


static void attempt_1(std::vector<std::vector<Pixel>> &pixels, int width, int height);
//static void attempt_2(int *pixels, int width, int height);
//static void attempt_3(int *pixels, int width, int height);

void cu_process(std::vector<std::vector<Pixel>> &pixels, int width, int height){
  gwidth = width;
  gheight = height;
  attempt_1(pixels,width,height);
  
  return;
}

void __global__ rowComp(Pixel *pixels_cu, int *next_cu, int *size_cu, int offset,int start, int size){
  int y = blockDim.y*blockIdx.y *size + threadIdx.y * size;
  int x = blockIdx.x * blockDim.x  * offset + threadIdx.x*offset + start;
  if(x+1>=global_width){
    return;
  }
  for(int i=0;i<size;i++){
    if(y+i>=global_height){
      break;
    }
    verify_edge(pixels_cu,next_cu,size_cu,x,y+i,x+1,y+i);
  }
}

void __global__ colComp(Pixel *pixels_cu, int *next_cu, int *size_cu, int offset,int start, int size){
  int x = blockDim.x *blockIdx.x *size + threadIdx.x * size;
  int y = blockIdx.y *blockDim.y * offset + threadIdx.y * offset + start;
  if(y+1>=global_height){
    return;
  }
  //get x and y somehow :)
  for(int i=0;i<size;i++){
    if(x+i>=global_width){
      break;
    }
    verify_edge(pixels_cu,next_cu,size_cu,x+i,y,x+i,y+1);
  }
}

void __global__ rowDiagComp(Pixel *pixels_cu,int *next_cu, int *size_cu, int offset, int start,int size){
  int y = blockDim.y *blockIdx.y*size + threadIdx.y * size;
  int x = blockIdx.x * blockDim.x * offset + threadIdx.x*offset + start;
  if(x+1>=global_width){
    return;
  }
  for(int i=0;i<size;i++){
    if(y+i+1>=global_height){
      break;
    }
    verify_edge(pixels_cu,next_cu,size_cu,x,y+i,x+1,y+i+1);
    verify_edge(pixels_cu,next_cu,size_cu,x+1,y+i,x,y+i+1);

  }
}

void __global__ colDiagComp(Pixel *pixels_cu, int *next_cu, int *size_cu, int offset, int start, int size){
  int x= blockDim.x *blockIdx.x*size + threadIdx.x * size;
  int y = blockIdx.y * blockDim.y * offset+threadIdx.y * offset + start;
  if(y+1>=global_width){
    return;
  }
  for(int i=0;i<size;i++){
    if(x+i+1>=global_width){
      break;
    }
    verify_edge(pixels_cu,next_cu,size_cu,x+i,y,x+i+1,y+1);
    verify_edge(pixels_cu,next_cu,size_cu,x+i+1,y,x+i,y+1);
  }
}
void __global__ redirect(Pixel *pixels_cu, int *next_cu){
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(y*global_width + x < global_width * global_height){
    int actual = find(next_cu, x,y);
    pixels_cu[y*global_width + x]  = pixels_cu[actual];
  }
  __syncthreads();
}

static void attempt_1(std::vector<std::vector<Pixel>> &pixels, int width, int height){
  Pixel *pixels_cu;
  std::vector<int> size(width * height,0);
  std::vector<int> next(width * height,-1);
  int *size_cu;
  int *next_cu;

  cudaMalloc((void **)&size_cu, sizeof(int)*width*height);
  cudaMalloc((void **)&next_cu, sizeof(int)*width*height);
  cudaMalloc((void **)&pixels_cu,sizeof(Pixel)*width*height);
  cudaMemcpy(pixels_cu,(Pixel*)&pixels[0][0],sizeof(Pixel)*width*height,cudaMemcpyHostToDevice);
  cudaMemcpy(size_cu,(int*)&size[0],sizeof(int)*width*height,cudaMemcpyHostToDevice);
  cudaMemcpy(next_cu,(int*)&next[0],sizeof(int)*width*height,cudaMemcpyHostToDevice);

  cudaMemcpy(&global_width,&gwidth,sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(&global_height,&gheight,sizeof(int),cudaMemcpyHostToDevice);
  int start = 0;
  int offset = 2;
  int rsize = 1;
  int csize = 1;
  while(start < width - 1 || start < height -1 ){
    int blockWidth = std::max(csize,BLOCK_WIDTH);
    int blockHeight = std::max(rsize,BLOCK_HEIGHT);
    int rWidthNum = (gwidth + blockWidth-1)/blockWidth;
    int rHeightNum = (gheight + blockHeight -1)/blockHeight;
    dim3 gridDim(rWidthNum,rHeightNum);
    dim3 blockDim((blockWidth+rsize-1)/rsize,(blockHeight+csize-1)/csize);
    rowComp<<< gridDim,blockDim  >>>(pixels_cu,next_cu,size_cu,offset,start,rsize);
    cudaDeviceSynchronize();

    csize *= 2;
    blockWidth = std::max(csize,BLOCK_WIDTH);
    rWidthNum = (gwidth + blockWidth-1)/blockWidth;
    dim3 gridDim2(rWidthNum,rHeightNum);
    dim3 blockDim2((blockWidth+rsize-1)/rsize,(blockHeight+csize-1)/csize);

    rowDiagComp<<<gridDim2,blockDim2 >>>(pixels_cu,next_cu,size_cu,offset,start,rsize/2);
    cudaDeviceSynchronize();
    colComp<<<gridDim2,blockDim2 >>>(pixels_cu,next_cu,size_cu,offset,start,csize);
    cudaDeviceSynchronize();

    rsize *= 2;
    blockHeight = std::max(rsize,BLOCK_HEIGHT);
    rHeightNum = (gheight + blockHeight -1)/blockHeight;
    dim3 gridDim3(rWidthNum,rHeightNum);
    dim3 blockDim3((blockWidth+rsize-1)/rsize,(blockHeight+csize-1)/csize);
    colDiagComp<<<gridDim3,blockDim3  >>>(pixels_cu,next_cu,size_cu,offset,start,csize/2);
    cudaDeviceSynchronize();
    start = 2*(start +1) - 1;
    offset *= 2;
  }
  dim3 gridDim((gwidth + BLOCK_WIDTH -1) /BLOCK_WIDTH,(gheight + BLOCK_HEIGHT -1)/BLOCK_HEIGHT);
  dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
  redirect<<<gridDim,blockDim  >>>(pixels_cu,next_cu);
  return;
}

/*
static void attempt_2(int *pixels, int width, int height){


}

static void attempt_3(int *pixels, int width, int height){


}
*/
