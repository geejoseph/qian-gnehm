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
  assert(col1< global_width && col1>=0 && col2 < global_width && col2 >= 0 && row1 < global_height && row1 >=0
      && row2 < global_height && row2 >=0);

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
  if(y+1>=global_width){
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
  printf("AAAAAAAAAAAAA\n");
  /*for(int i=0;i<global_width*global_height;i++){
    printf("%d\n",next_cu[i]);
  }*/
}
static void attempt_1(std::vector<Pixel> &pixels, int width, int height){
  Pixel *pixels_cu;
  std::vector<int> size(width * height,1);
  std::vector<int> next(width * height,-1);
  int *size_cu;
  int *next_cu;
  //show<<<1,1>>>();
  //return;
  cudaMalloc((void **)&size_cu, sizeof(int)*width*height);
  cudaMalloc((void **)&next_cu, sizeof(int)*width*height);
  cudaMalloc((void **)&pixels_cu,sizeof(Pixel)*width*height);
  cudaMemcpy(pixels_cu,(Pixel*)&pixels[0],sizeof(Pixel)*width*height,cudaMemcpyHostToDevice);
  cudaMemcpy(size_cu,(int*)&size[0],sizeof(int)*width*height,cudaMemcpyHostToDevice);
  cudaMemcpy(next_cu,(int*)&next[0],sizeof(int)*width*height,cudaMemcpyHostToDevice);

  int start = 0;
  //what it should look like after a global fn call
  int rsize = 1;
  int csize = 2;
  int count=0;
  while(start < width - 1 || start < height -1 ){
    int blockWidth = std::max(csize,BLOCK_WIDTH);
    int blockHeight = std::max(rsize,BLOCK_HEIGHT);
    int rWidthNum = (global_width + blockWidth-1)/blockWidth;
    int rHeightNum = (global_height + blockHeight -1)/blockHeight;
    dim3 rgridDim(rWidthNum,rHeightNum);
    dim3 rblockDim((blockWidth+csize-1)/csize,(blockHeight+rsize-1)/rsize);
    printf("before rowcomp rWidthNum: %d rHeightNum: %d blockX: %d blockY: %d\n",rWidthNum,rHeightNum,
        (blockWidth+csize-1)/csize,(blockHeight+rsize-1)/rsize);
    rowComp<<< rgridDim, rblockDim  >>>(pixels_cu,next_cu,size_cu,start,rsize,csize,global_width,global_height);
    cudaCheckError(cudaThreadSynchronize());
    
    if(count ==10){
      return;
    }   
    printf("finished row comp\n");
    //cudaMemcpy(&next[0],next_cu,sizeof(int)*width*height,cudaMemcpyDeviceToHost);
    
    //both diagonal ops are 'effectless'
    rowDiagComp<<<rgridDim,rblockDim >>>(pixels_cu,next_cu,size_cu,start,rsize,csize,global_width,global_height);
    cudaDeviceSynchronize();

    printf("finished rowdiag\n");
    
    rsize *= 2;
    blockHeight = std::max(rsize,BLOCK_HEIGHT);
    rHeightNum = (global_height + blockHeight - 1)/blockHeight;
    dim3 cgridDim(rWidthNum,rHeightNum);
    dim3 cblockDim((blockWidth +csize-1)/csize,(blockHeight + rsize-1)/rsize);

    colComp<<<cgridDim,cblockDim >>>(pixels_cu,next_cu,size_cu,start,rsize,csize,global_width,global_height);
    cudaDeviceSynchronize();

    printf("finished col comp\n");
    
    colDiagComp<<<cgridDim,cblockDim >>>(pixels_cu,next_cu,size_cu,start,rsize,csize,global_width,global_height);
    cudaDeviceSynchronize();

    printf("finished coldiag\n");
    start = 2*(start +1) - 1;
    csize *= 2;
    count++;
  }
  dim3 gridDim((global_width + BLOCK_WIDTH -1) /BLOCK_WIDTH,(global_height + BLOCK_HEIGHT -1)/BLOCK_HEIGHT);
  dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
  redirect<<<gridDim,blockDim  >>>(pixels_cu,next_cu,global_width,global_height);
  cudaMemcpy((Pixel*)&pixels[0],pixels_cu,sizeof(Pixel)*width*height,cudaMemcpyDeviceToHost);
  cudaFree(pixels_cu);
  cudaFree(next_cu);
  cudaFree(size_cu);
  return;
}

/*
static void attempt_2(int *pixels, int width, int height){


}

static void attempt_3(int *pixels, int width, int height){


}
*/
