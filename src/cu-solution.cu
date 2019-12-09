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
#include "CycleTimer.h"

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n",
    cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
int global_width;
int global_height;

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

__device__ int find(int *next,int srow,int scol,int global_width,int global_height){
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
    int *size_cu, int col1, int row1, int col2, int row2,int global_width, int global_height) {
  //sanity check
  assert(col1< global_width);
  assert(col1>=0);
  assert(col2 < global_width);
  assert(col2 >= 0);
  assert(row1 < global_height);
  assert(row1 >=0);
  assert(row2 < global_height);
  assert(row2 >=0);

  int aIndex = find(next_cu,row1,col1,global_width,global_height);
  //int aRow = aIndex / global_width;
  //int aCol = aIndex % global_width;

  int bIndex = find(next_cu,row2,col2,global_width,global_height);
  //int bRow = bIndex / global_width;
  //int bCol = bIndex % global_width;

  assert(aIndex!= -1 && bIndex != -1);
  assert(aIndex<global_width*global_height && bIndex<global_height*global_width);
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


static void attempt_1(std::vector<Pixel> &pixels, int width, int height);
//static void attempt_2(int *pixels, int width, int height);
//static void attempt_3(int *pixels, int width, int height);

void cu_process(std::vector<Pixel> &pixels, int width, int height){
  global_width = width;
  global_height = height;
  attempt_1(pixels,width,height);
  
  return;
}

__global__ void rowComp(Pixel *pixels_cu, int *next_cu, int *size_cu,int start, int rsize,int csize,int global_width,
                        int global_height){
  int y = blockDim.y*blockIdx.y *rsize + threadIdx.y * rsize;
  int x = blockIdx.x * blockDim.x  * csize + threadIdx.x*csize + start;
  //printf("x: %d, y: %d\n global_width: %d\n",x,y,global_width);
  if(x+1>=global_width){
    return;
  }
  //printf("here\n");
  for(int i=0;i<rsize;i++){
    if(y+i>=global_height){
      break;
    }
    //printf("about to ver edge\n");
    verify_edge(pixels_cu,next_cu,size_cu,x,y+i,x+1,y+i,global_width,global_height);
  }
}

__global__ void colComp(Pixel *pixels_cu, int *next_cu, int *size_cu,int start, int rsize, int csize, int global_width,
                        int global_height){
  int x = blockDim.x *blockIdx.x * csize + threadIdx.x * csize;
  int y = blockIdx.y *blockDim.y * rsize + threadIdx.y * rsize + start;
  if(y+1>=global_height){
    return;
  }
  //get x and y somehow :)
  for(int i=0;i<csize;i++){
    if(x+i>=global_width){
      break;
    }
    verify_edge(pixels_cu,next_cu,size_cu,x+i,y,x+i,y+1,global_width,global_height);
  }
}

__global__ void rowDiagComp(Pixel *pixels_cu,int *next_cu, int *size_cu, int start,int rsize, int csize, 
                            int global_width,int global_height){
  int y = blockDim.y *blockIdx.y*rsize + threadIdx.y * rsize;
  int x = blockIdx.x * blockDim.x * csize + threadIdx.x*csize + start;
  if(x+1>=global_width){
    return;
  }
  for(int i=0;i<rsize - 1;i++){
    if(y+i+1>=global_height){
      break;
    }
    verify_edge(pixels_cu,next_cu,size_cu,x,y+i,x+1,y+i+1,global_width,global_height);
    verify_edge(pixels_cu,next_cu,size_cu,x+1,y+i,x,y+i+1,global_width,global_height);

  }
}

__global__ void colDiagComp(Pixel *pixels_cu, int *next_cu, int *size_cu, int start, int rsize,int csize,
                            int global_width, int global_height){
  int x= blockDim.x *blockIdx.x*csize + threadIdx.x * csize;
  int y = blockIdx.y * blockDim.y * rsize+threadIdx.y * rsize + start;
  if(y+1>=global_height){
    return;
  }
  for(int i=0;i<csize-1;i++){
    if(x+i+1>=global_width){
      break;
    }
    verify_edge(pixels_cu,next_cu,size_cu,x+i,y,x+i+1,y+1,global_width,global_height);
    verify_edge(pixels_cu,next_cu,size_cu,x+i+1,y,x+i,y+1,global_width,global_height);
  }
}
__global__ void redirect(Pixel *pixels_cu, int *next_cu,int global_width,int global_height){
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(y*global_width + x < global_width * global_height){
    int actual = find(next_cu, y,x,global_width,global_height);
    pixels_cu[y*global_width + x]  = pixels_cu[actual];
  }
  __syncthreads();
}
__global__ void show(){
  //printf("AAAAAAAAAAAAA\n");
  /*for(int i=0;i<global_width*global_height;i++){
    printf("%d\n",next_cu[i]);
  }*/
}
//relatively naive approach, every phase gets its own kernel function, that does work appropriately
static void attempt_1(std::vector<Pixel> &pixels, int width, int height){
  Pixel *pixels_cu;
  std::vector<int> size(width * height,1);
  std::vector<int> next(width * height,-1);
  int *size_cu;
  int *next_cu;

  double ms= CycleTimer::currentSeconds();
  cudaMalloc((void **)&size_cu, sizeof(int)*width*height);
  cudaMalloc((void **)&next_cu, sizeof(int)*width*height);
  cudaMalloc((void **)&pixels_cu,sizeof(Pixel)*width*height);
  cudaMemcpy(pixels_cu,(Pixel*)&pixels[0],sizeof(Pixel)*width*height,cudaMemcpyHostToDevice);
  cudaMemcpy(size_cu,(int*)&size[0],sizeof(int)*width*height,cudaMemcpyHostToDevice);
  cudaMemcpy(next_cu,(int*)&next[0],sizeof(int)*width*height,cudaMemcpyHostToDevice);

  double me = CycleTimer::currentSeconds();

  printf("cuda overhead: %.3f ms\n",1000.f *(me-ms)); 

  int start = 0;
  //what it should look like after a global fn call
  int rsize = 1;
  int csize = 2;
  while(start < width - 1 || start < height -1 ){
    double s = CycleTimer::currentSeconds();
    int blockWidth = std::max(csize,BLOCK_WIDTH);
    int blockHeight = std::max(rsize,BLOCK_HEIGHT);
    int rWidthNum = (global_width + blockWidth-1)/blockWidth;
    int rHeightNum = (global_height + blockHeight -1)/blockHeight;
    dim3 rgridDim(rWidthNum,rHeightNum);
    dim3 rblockDim((blockWidth+csize-1)/csize,(blockHeight+rsize-1)/rsize);
    rowComp<<< rgridDim, rblockDim  >>>(pixels_cu,next_cu,size_cu,start,rsize,csize,global_width,global_height);
    cudaThreadSynchronize();
    
    //both diagonal ops are 'effectless'
    rowDiagComp<<<rgridDim,rblockDim >>>(pixels_cu,next_cu,size_cu,start,rsize,csize,global_width,global_height);
    cudaDeviceSynchronize();

    
    rsize *= 2;
    blockHeight = std::max(rsize,BLOCK_HEIGHT);
    rHeightNum = (global_height + blockHeight - 1)/blockHeight;
    dim3 cgridDim(rWidthNum,rHeightNum);
    dim3 cblockDim((blockWidth +csize-1)/csize,(blockHeight + rsize-1)/rsize);

    colComp<<<cgridDim,cblockDim >>>(pixels_cu,next_cu,size_cu,start,rsize,csize,global_width,global_height);
    cudaDeviceSynchronize();

    
    colDiagComp<<<cgridDim,cblockDim >>>(pixels_cu,next_cu,size_cu,start,rsize,csize,global_width,global_height);
    cudaDeviceSynchronize();

    start = 2*(start +1) - 1;
    csize *= 2;

    double e = CycleTimer::currentSeconds();
    printf("iter time: %.3f ms\n",1000.f *(e-s));
  }
  double rdirS = CycleTimer::currentSeconds();
  dim3 gridDim((global_width + BLOCK_WIDTH -1) /BLOCK_WIDTH,(global_height + BLOCK_HEIGHT -1)/BLOCK_HEIGHT);
  dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
  redirect<<<gridDim,blockDim  >>>(pixels_cu,next_cu,global_width,global_height);
  cudaMemcpy((Pixel*)&pixels[0],pixels_cu,sizeof(Pixel)*width*height,cudaMemcpyDeviceToHost);
  double rdirE = CycleTimer::currentSeconds();

  printf("Redir time: %.3f ms\n", 1000.f *(rdirE-rdirS));
  return;
}

__global__ void shared_process(int *pixels_cu, int *size_cu, int *next_cu,width,height){
    __shared__ pixels_temp[SHARED_BLOCK_DIM*SHARED_BLOCK_DIM];
    __shared__ size_temp[SHARED_BLOCK_DIM*SHARED_BLOCK_DIM];
    __shared__ next_temp[SHARED_BLOCK_DIM*SHARED_BLOCK_DIM];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tempIndex = threadIdx.y * blockDim.x + threadIdx.x;
    int actualIndex = y *width + x;
    pixels_temp[tempIndex] = pixels_cu[actualIndex];
    size_temp[tempIndex] = size_cu[actualIndex];
    next_temp[tempIndex] = next_cu[actualIndex];
    
    __syncthreads();

    int start = 0;
    int rsize = 1;
    int csize = 2;
    while(start < SHARED_BLOCK_DIM-1){
      //manual rowcomp rowdiag, col comp and coldiag
    }
    pixels_cu[actualIndex] = pixels_temp[tempIndex];
    size_cu[actualIndex] = size_temp[tempIndex];
    next_cu[actualIndex] = size_temp[tempIndex];
    return;
}
/* a shared memory approach
   step one involves using shared memory
   second is traditional
   */
static void attempt_2(int *pixels, int width, int height){
  if(width<SHARED_BLOCK_DIM || height < SHARED_BLOCK_DIM){
    attempt_1(pixels,width,height);
    return;
  }
  else{
    Pixel *pixels_cu;
    std::vector<int> size(weight*height,-1);
    std::vector<int> next(weight*height,1);
    int *size_cu;
    int *next_cu;

    cudaMalloc((void**)&pixels_cu,sizeof(Pixel)*width*height);
    cudaMalloc((void**)&size_cu,sizeof(int)*width*height);
    cudaMalloc((void**)&next_cu,sizeof(int)*width*height);

    cudaMemcpy(pixels_cu,(Pixel*)&pixels[0],sizeof(Pixel)*width*height,cudaMemcpyHostToDevice);
    cudaMemcpy(next_cu,(int*)&next[0],sizeof(int)*width*height,cudaMemcpyHostToDevice);
    cudaMemcpy(size_cu(int*)&size[0],sizeof(int)*width*height,cudaMemcpyHostToDevice);

    int widthNum = (width+SHARED_BLOCK_DIM-1)/SHARED_BLOCK_DIM;
    int heightNum = (height + SHARED_BLOCK_DIM-1)/SHARED_BLOCK_DIM;
    dim3 gridDim(widthNum,heightNum);
    dim3 blockDim(SHARED_BLOCK_DIM,SHARED_BLOCK_DIM);
    shared_process<<<gridDim,blockDim>>>(pixels_cu,size_cu,next_cu,width,height);
    cudaThreadSynchronize();
    int start = 31;
    int rsize = 32;
    int csize = 64;
    //copy from attempt1
    while(start<width -1 || start < height-1){
      int blockWidth = csize;
      int blockHeight = rsize;
      int rWidthNum = (width + blockWidth-1)/blockWidth;
      int rHeightNum = (height + blockHeight-1)/blockHeight;
      dim3 rgridDim(rWidthNum,rHeightNum);
      dim3 rblockDim((blockWidth+csize-1)/csize,(blockHeight+rsize-1)/rsize);

      rowComp<<<rgridDim,rblockDim>>>(pixels_cu,next_cu,size_cu,start,rsize,csize,width,height);
      cudaThreadSynchronize();

      rowDiagComp<<<rgridDim,rblockDim>>>(pixels_cu,next_cu,size_cu,start,rsize,csize,width,height);
      cudaThreadSynchronize();

      rsize *=2;
      blockHeight = rsize;
      rHeightNum = (height + blockHeight-1)/blockHeight;
      dim3 cgridDim(rWidthNum,rHeightNum);
      dim3 cblockDim((blockWidth+csize-1)/csize,(blockHeight+rsize-1)/rsize);
      colComp<<<cgridDim,cblockDim>>>(pixels_cu,next_cu,size_cu,start,rsize,csize,width,height);
      cudaThreadSynchronize();

      colDiagComp<<<cgridDim,cblockDim>>>(pixels_cu,next_cu,size_cu,start,rsize,csize,global_width,global_height);
      cudaThreadSynchronize();

      start = 2*(start +1 )-1;
      csize *= 2;
    }
    redirect<<<gridDim,blockDim>>>(pixels_cu,next_cu,global_width,global_height);
    cudaMemcpy((Pixel*)&pixels[0],pixels_cu,sizeof(Pixel)*width*height,cudaMemcpyDeviceToHost);
  }
  return;
}

/*
static void attempt_3(int *pixels, int width, int height){


}
*/
