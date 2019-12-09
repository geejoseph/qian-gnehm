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
#define SHARED_BLOCK_DIM 32
#define CHUNK_SIZE 512
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
  for(int i=0;i<global_width*global_height;i++){
    assert(pos>=0);
    assert(pos<= global_width*global_height);
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
  assert(false);
}

/*
__device__ int shared_find(int *next_temp,int sr,int sc, int width,int height){
  assert(sr<32);
  assert(sc<32);
  assert(sr>=0);
  assert(sc>=0);
  assert(false);
  assert(blockDim.x == 32);
  int pos = sr *blockDim.x + sc;
  while(1){
    assert(pos>=0);
    assert(pos<1024);
    int index = next_temp[pos];
    //assert(index<1024);
    //assert(index>=-1);
    if(index == -1){
      next_temp[pos]=pos;
      next_temp[sr * blockDim.x + sc] = pos;
      return pos;
    }
    if( index == pos){
      next_temp[sr * blockDim.x + sc] = index;
      return index;
    }
    pos = index;
  }
}
*/
//local next_temp will need to have the values of the actual next array
//note that it returns the local pos, and just check into the shared array to find actual

__device__ __inline__ int shared_find(int *next_temp, int sr, int sc, int width, int height){
  int xStart = blockIdx.x * blockDim.x;
  int yStart = blockIdx.y * blockDim.y;
  int sX = sc - xStart;
  int sY = sr - yStart;
  //assert(blockDim.x==32);
  //assert(sY>=0);
  //assert(sY<32);
  //assert(sX>=0);
  //assert(sX<32);
  int start = xStart + yStart *width;
  int pos = sY * blockDim.x +sX;
  while(1){
    //assert(pos>=0);
    //assert(pos<1024);
    int index = next_temp[pos];
    if(index == -1){
      int posy = pos/blockDim.x;
      int posx = pos % blockDim.x;
      int temp = start + posx + posy * width;
      next_temp[pos] = temp;
      next_temp[sY * blockDim.x + sX] = temp;
      return pos;
    }
    int checkY = index/width -yStart;
    int checkX = index%width - xStart;
    if(checkX + checkY*blockDim.x == pos){
      next_temp[sY * blockDim.x + sX] = index;
      return pos;
    }
    pos = checkX + checkY * blockDim.x;
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

  int bIndex = find(next_cu,row2,col2,global_width,global_height);

  assert(aIndex!= -1 && bIndex != -1);
  assert(aIndex<global_width*global_height && bIndex<global_height*global_width);
  if(aIndex !=  bIndex){
    Pixel A = pixels_cu[aIndex];
    int aSize = size_cu[aIndex];

    Pixel B = pixels_cu[bIndex];
    int bSize = size_cu[bIndex];

    if(mergeCriterion(A,B,30)){
      if(aSize>bSize){
        pixels_cu[aIndex] = newColor(A,B, aSize, bSize);
        next_cu[bIndex] = aIndex;
        size_cu[aIndex] += bSize;
      }
      else{
        pixels_cu[bIndex] = newColor(A, B, aSize, bSize);
        next_cu[aIndex] = bIndex;
        size_cu[bIndex] += aSize;
      }
    }
  }
  return;

}

__device__ void shared_verify_edge(Pixel *pixels_temp, int *next_temp, int *size_temp, int c1, int r1, 
                                    int c2, int r2, int width, int height){
  assert(c1< width);
  assert(c1>=0);
  assert(c2 < width);
  assert(c2 >= 0);
  assert(r1 < height);
  assert(r1 >=0);
  assert(r2 < height);
  assert(r2 >=0);
  int aIndex = shared_find(next_temp,r1,c1,width,height);
  int bIndex = shared_find(next_temp,r2,c2,width,height);

  assert(aIndex!= -1 && bIndex != -1);
  assert(aIndex<1024 && bIndex<1024);
  if(aIndex != bIndex){
    Pixel A = pixels_temp[aIndex];
    int aSize = size_temp[aIndex];

    Pixel B = pixels_temp[bIndex];
    int bSize = size_temp[bIndex];

    if(mergeCriterion(A,B,30)){
      if(aSize>bSize){
        pixels_temp[aIndex] = newColor(A,B,aSize,bSize);
        next_temp[bIndex] = next_temp[aIndex];
        size_temp[aIndex] += bSize;
      }
      else{
        pixels_temp[bIndex] = newColor(A,B,aSize,bSize);
        next_temp[aIndex] =next_temp[bIndex];
        size_temp[bIndex] += aSize;
  
      }
    }
  }
  return;

}
static void attempt_1(std::vector<Pixel> &pixels, int width, int height);
static void attempt_2(std::vector<Pixel> &pixels, int width, int height);
static void attempt_3(std::vector<Pixel> &pixels, int width, int height);

void cu_process(std::vector<Pixel> &pixels, int width, int height){
  global_width = width;
  global_height = height;
  attempt_3(pixels,width,height);
  
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
/**********************************************************************************
*
*
*  SHARED MEMORY ATTEMPT
*
*
**********************************************************************************/

__device__ __inline__ void sharedRowComp(Pixel *pixels_temp,int *next_temp, int *size_temp, int start, int rsize, int csize,
                              int width, int height){
  //determine if this thread needs to do work
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int locX = threadIdx.x;
  int locY = threadIdx.y;
  if(x>=start && (x-start) % (csize) == (0) && (x+1)<width && y % rsize == 0){  
    //printf("here x %d y %d\n",x,y);
    for(int i=0;i<rsize;i++){
      if(y+i>=height){
        break;
      }
      shared_verify_edge(pixels_temp,next_temp, size_temp,x,y+i,x+1,y+i,width,height);
    }
  }
  return;
}

__device__ __inline__ void sharedColComp(Pixel *pixels_temp, int *next_temp, int *size_temp, int start, int rsize, int csize,
                              int width, int height){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int locX = threadIdx.x;
  int locY = threadIdx.y;
  if(y>=start && (y-start) % (rsize) == (0) && y+1<height && x % csize == 0){
    for(int i=0;i<csize;i++){
      if(x+i>=width){
        break;
      }
      shared_verify_edge(pixels_temp,next_temp,size_temp,x+i,y,x+i,y+1,width,height);
    }
  }
  return;
}

__device__ __inline__ void sharedRowDiagComp(Pixel *pixels_temp, int *next_temp, int *size_temp, int start, int rsize, int csize,
                                  int width, int height){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int locX = threadIdx.x;
  int locY = threadIdx.y;
  if(x>= start && (x-start) %(csize) == (0) && x+1<width && y %rsize == 0){
    for(int i=0;i<rsize-1;i++){
      if(y+i+1>=height){
        break;
      }
      shared_verify_edge(pixels_temp,next_temp,size_temp,x,y+i,x+1,y+i+1,width,height);
      shared_verify_edge(pixels_temp,next_temp,size_temp,x+1,y+i,x,y+i+1,width,height);
    }
  }
  return; 
}

__device__ __inline__ void sharedColDiagComp(Pixel *pixels_temp, int *next_temp, int *size_temp, int start, int rsize, int csize,
                                  int width, int height){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int locX = threadIdx.x;
  int locY = threadIdx.y;
  if(y>= start && (y-start) % (rsize) == (0) && y+1<height && x % csize == 0){
    for(int i=0;i<csize-1;i++){
      if(x+i+1>=width){
        break;
      }
      shared_verify_edge(pixels_temp,next_temp,size_temp,x+i,y,x+i+1,y+1,width,height);
      shared_verify_edge(pixels_temp,next_temp,size_temp,x+i+1,y,x+i,y+1,width,height);
    }
  }
  return;
}

__global__ void shared_process(Pixel *pixels_cu, int *next_cu, int *size_cu,int width,int height){
    __shared__ Pixel pixels_temp[SHARED_BLOCK_DIM*SHARED_BLOCK_DIM];
    __shared__ int size_temp[SHARED_BLOCK_DIM*SHARED_BLOCK_DIM];
    __shared__ int next_temp[SHARED_BLOCK_DIM*SHARED_BLOCK_DIM];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tempIndex = threadIdx.y * blockDim.x + threadIdx.x;
    int actualIndex = y *width + x;
    if(actualIndex>=0 &&  actualIndex<width*height && tempIndex < SHARED_BLOCK_DIM*SHARED_BLOCK_DIM && tempIndex>=0){
      
      pixels_temp[tempIndex] = pixels_cu[actualIndex];
      size_temp[tempIndex] = size_cu[actualIndex];
      next_temp[tempIndex] = next_cu[actualIndex];
      
    
      __syncthreads();
       
      int start = 0;
      int rsize = 1;
      int csize = 2;
      while(start < SHARED_BLOCK_DIM-1){
        //manual rowcomp rowdiag, col comp and coldiag
        sharedRowComp(pixels_temp,next_temp,size_temp,start,rsize,csize,width,height);
        __syncthreads();
        //break;

        sharedRowDiagComp(pixels_temp,next_temp,size_temp,start,rsize,csize,width,height);
        __syncthreads();

        rsize *=2;

        sharedColComp(pixels_temp,next_temp,size_temp,start,rsize,csize,width,height);
        __syncthreads();

        sharedColDiagComp(pixels_temp,next_temp,size_temp,start,rsize,csize,width,height);
        __syncthreads();

        start = 2*(start+1)-1;
        csize *= 2;
      }
    
      pixels_cu[actualIndex] = pixels_temp[tempIndex];
      size_cu[actualIndex] = size_temp[tempIndex];
      int tempX= next_temp[tempIndex] % blockDim.x;
      int tempY = next_temp[tempIndex] / blockDim.x;
      int store = tempX + (blockIdx.x*blockDim.x) + (tempY + (blockIdx.y*blockDim.y))*width;

      next_cu[actualIndex] =next_temp[tempIndex];
      //assert(store<width*height);
      
    //printf("next cu %d x %d y %d\n",next_temp[actualIndex],x,y);
    }
    //__syncthreads(); 
    return;
}
/* a shared memory approach
   step one involves using shared memory
   second is traditional
   */
static void attempt_2(std::vector<Pixel> &pixels, int width, int height){
  if(width<SHARED_BLOCK_DIM || height < SHARED_BLOCK_DIM){
    attempt_1(pixels,width,height);
    return;
  }
  else{
    Pixel *pixels_cu;
    std::vector<int> size(width*height,1);
    std::vector<int> next(width*height,-1);
    int *size_cu;
    int *next_cu;
    double mS = CycleTimer::currentSeconds();
    cudaMalloc((void**)&pixels_cu,sizeof(Pixel)*width*height);
    cudaMalloc((void**)&size_cu,sizeof(int)*width*height);
    cudaMalloc((void**)&next_cu,sizeof(int)*width*height);

    cudaMemcpy(pixels_cu,(Pixel*)&pixels[0],sizeof(Pixel)*width*height,cudaMemcpyHostToDevice);
    cudaMemcpy(next_cu,(int*)&next[0],sizeof(int)*width*height,cudaMemcpyHostToDevice);
    cudaMemcpy(size_cu,(int*)&size[0],sizeof(int)*width*height,cudaMemcpyHostToDevice);

    double mE = CycleTimer::currentSeconds();

    printf("cuda overhead %.3f ms\n",1000.f * (mE-mS));

    int widthNum = (width+SHARED_BLOCK_DIM-1)/SHARED_BLOCK_DIM;
    int heightNum = (height + SHARED_BLOCK_DIM-1)/SHARED_BLOCK_DIM;
    //dim3 gridDim(1,1);
    //dim3 blockDim(2,2);
    dim3 gridDim(widthNum,heightNum);
    dim3 blockDim(SHARED_BLOCK_DIM,SHARED_BLOCK_DIM);
    
    double sS = CycleTimer::currentSeconds();
    shared_process<<<gridDim,blockDim>>>(pixels_cu,next_cu,size_cu,width,height);
    cudaCheckError(cudaThreadSynchronize());
    double sE = CycleTimer::currentSeconds();
    printf("shared memory time %.3f ms\n",1000.f * (sE-sS));
    //return; 
    /*cudaMemcpy((int*)&next[0],next_cu,sizeof(int)*width*height,cudaMemcpyDeviceToHost);
    for(int i=0;i<width*height;i++){
      printf("%d\n",next[i]);
    }
    return;*/
    int start = 31;
    int rsize = 32;
    int csize = 64;
    //copy from attempt1
    while(start<width -1 || start < height-1){
      double s = CycleTimer::currentSeconds();

      int blockWidth = std::max(csize,BLOCK_WIDTH);
      int blockHeight = std::max(rsize,BLOCK_HEIGHT);
      int rWidthNum = (width + blockWidth-1)/blockWidth;
      int rHeightNum = (height + blockHeight-1)/blockHeight;
      dim3 rgridDim(rWidthNum,rHeightNum);
      dim3 rblockDim((blockWidth+csize-1)/csize,(blockHeight+rsize-1)/rsize);

      rowComp<<<rgridDim,rblockDim>>>(pixels_cu,next_cu,size_cu,start,rsize,csize,width,height);
      //cudaThreadSynchronize();
      cudaCheckError(cudaThreadSynchronize());

      //printf("row\n");

      rowDiagComp<<<rgridDim,rblockDim>>>(pixels_cu,next_cu,size_cu,start,rsize,csize,width,height);
      cudaThreadSynchronize();
      //printf("rowdia\n");

      rsize *=2;
      blockHeight = std::max(rsize,BLOCK_HEIGHT);
      rHeightNum = (height + blockHeight-1)/blockHeight;
      dim3 cgridDim(rWidthNum,rHeightNum);
      dim3 cblockDim((blockWidth+csize-1)/csize,(blockHeight+rsize-1)/rsize);

      colComp<<<cgridDim,cblockDim>>>(pixels_cu,next_cu,size_cu,start,rsize,csize,width,height);
      cudaThreadSynchronize();
      //printf("col\n");

      colDiagComp<<<cgridDim,cblockDim>>>(pixels_cu,next_cu,size_cu,start,rsize,csize,width,height);
      cudaThreadSynchronize();
      //printf("coldia\n");

      start = 2*(start +1 )-1;
      csize *= 2;

      double e = CycleTimer::currentSeconds();

      printf("iter time: %.3f ms\n",1000.f *(e-s));
    }

    double rS = CycleTimer::currentSeconds();
    dim3 gD((width+BLOCK_WIDTH-1)/BLOCK_WIDTH,(height+BLOCK_HEIGHT-1)/BLOCK_HEIGHT);
    dim3 bD(BLOCK_WIDTH,BLOCK_HEIGHT);
    redirect<<<gD,bD>>>(pixels_cu,next_cu,width,height);
    cudaMemcpy((Pixel*)&pixels[0],pixels_cu,sizeof(Pixel)*width*height,cudaMemcpyDeviceToHost);
    double rE = CycleTimer::currentSeconds();
    printf("Redir time %.3f ms\n",1000.f *(rE-rS));
  }
  return;
}

/*********************************************************************


  ATTEMPT 3 "HYBRID SOLUTION"



  *******************************************************************/

__global__ void seq_continue(Pixel *pixels_cu,int * next_cu, int *size_cu,int csize, int width, int height){
  int start = CHUNK_SIZE *2 - 1;
  int offset = csize;
  int limit;
  while(start< width -1 || start < height -1){
     //double s = CycleTimer::currentSeconds();
    //Comparing along row
    for(int y =0;y<height;y++){
      for(int x = start;x<=width-offset;x+=offset){
        verify_edge(pixels_cu,next_cu,size_cu,x,y,x+1,y,width,height);
      }   
    }   
    //std::cout<<"row comparison done"<<std::endl;
    for(int y = 0; y<height; y+=offset/2){
      for(int x = start;x<=width-offset;x+=offset){
        limit = offset/2 - 1;
        //guarantee y+limit <= height
        if(y + limit > height){
          limit = height - y -1; 
        }   
        for(int n=0;n<limit;n++){
          verify_edge(pixels_cu,next_cu,size_cu,x,y+n,x+1,y+n+1,width,height);
          verify_edge(pixels_cu,next_cu,size_cu,x+1,y+n,x,y+n+1,width,height);
        }   
      }   
    }   

    //std::cout<<"second loop done"<<std::endl;
    for ( int y = start; y<=height-offset; y += offset){
      for(int x =0 ; x< width; x++){
        verify_edge(pixels_cu,next_cu,size_cu,x,y,x,y+1,width,height);
      }   
    }   
    //std::cout<<"third loop done"<<std::endl;
    for(int y = start; y <= height-offset; y+= offset){
      for(int x = 0; x<= width-offset; x+=offset){
        limit = offset -1; 
        if(x+limit>width){
          limit =width - x -1; 
        }   
        for( int n = 0;n<limit;n++){
          verify_edge(pixels_cu,next_cu,size_cu,x+n,y,x+n+1,y+1,width,height);
          verify_edge(pixels_cu,next_cu,size_cu,x+n+1,y,x+n,y+1,width,height);
        }   
      }   
    }   
    start = 2*(start+1)-1;
    offset *=2;

    //double e = CycleTimer::currentSeconds();

    //printf("Iter time: %.3f ms\n",1000.f * (e-s)); 

  }


}

static void attempt_3(std::vector<Pixel> &pixels, int width, int height){
    Pixel *pixels_cu;
    std::vector<int> size(width*height,1);
    std::vector<int> next(width*height,-1);
    int *size_cu;
    int *next_cu;

    double ms = CycleTimer::currentSeconds();
    cudaMalloc((void**) &size_cu, sizeof(int) *width*height);
    cudaMalloc((void**)&next_cu,sizeof(int)*width*height);
    cudaMalloc((void**)&pixels_cu,sizeof(Pixel)*width*height);
    cudaMemcpy(pixels_cu,(Pixel*)&pixels[0],sizeof(Pixel)*width*height,cudaMemcpyHostToDevice);
    cudaMemcpy(size_cu,(int*)&size[0],sizeof(int)*width*height,cudaMemcpyHostToDevice);
    cudaMemcpy(next_cu,(int*)&next[0],sizeof(int)*width*height,cudaMemcpyHostToDevice);

    double me = CycleTimer::currentSeconds();

    printf("cuda overhead: %.3f ms\n",1000.f *(me-ms));
    
    int start = 0;
    int rsize = 1;
    int csize = 2;
    while(start< std::min(CHUNK_SIZE,width-1) || start < std::min(CHUNK_SIZE,height-1)){
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
    double scs= CycleTimer::currentSeconds();
    seq_continue<<<1,1>>>(pixels_cu,next_cu,size_cu,csize,width,height);
    cudaDeviceSynchronize();
    double sce = CycleTimer::currentSeconds();

    printf("seq part : %.3f ms\n",1000.f *(sce-scs));

    double rdirS = CycleTimer::currentSeconds();
    dim3 gridDim((global_width + BLOCK_WIDTH -1) /BLOCK_WIDTH,(global_height + BLOCK_HEIGHT -1)/BLOCK_HEIGHT);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
    redirect<<<gridDim,blockDim  >>>(pixels_cu,next_cu,global_width,global_height);
    cudaMemcpy((Pixel*)&pixels[0],pixels_cu,sizeof(Pixel)*width*height,cudaMemcpyDeviceToHost);
    double rdirE = CycleTimer::currentSeconds();

    printf("redir: %.3f ms\n",1000.f *(rdirE-rdirS));

}

