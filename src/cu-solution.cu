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

int global_width;
int global_height;
inline bool mergeCriterion(Pixel p1, Pixel p2, int t){
  return ((int)p1.r - (int)p2.r)*((int)p1.r - (int)p2.r) + ((int)p1.g - (int)p2.g)*((int)p1.g -(int) p2.g) +
    ((int)p1.b - (int)p2.b) * ((int)p1.b - (int)p2.b) < t*t;
}

Pixel newColor(Pixel A, Pixel B, int sizeA, int sizeB){
  int totalSize = sizeA + sizeB;
  Pixel newP;
  newP.r = (uint8_t)(((int)A.r * sizeA +(int) B.r * sizeB)/(totalSize));
  newP.g = (uint8_t)(((int)A.g * sizeA + (int)B.g * sizeB)/(totalSize));
  newP.b = (uint8_t)(((int)A.b * sizeA + (int)B.b * sizeB)/(totalSize));
  return newP;
}

int find(std::vector<std::vector<int>> &next,int srow,int scol){
  int row = srow;
  int col = scol;
  while(1){
    //std::cout<<"in find"<<std::endl;
    int index = next[row][col];
    if(index == -1){
      int actualIndex = row *global_width + col;
      next[row][col] = actualIndex;
      next[srow][scol] = actualIndex;
      return actualIndex;
    }
    int nextRow = index/global_width;
    int nextCol = index % global_width;
    if(nextRow == row && nextCol == col){
      next[srow][scol] = index;
      return index;
    }
    row = nextRow;
    col = nextCol;
    //std::cout<<"here"<<std::endl;
  }
}

void verify_edge(std::vector<std::vector<Pixel>> &pixels, std::vector<std::vector<int>> &next,
    std::vector<std::vector<int>> &size, int col1, int row1, int col2, int row2) {
  //sanity check
  assert(col1< global_width && col1>=0 && col2 < global_width && col2 >= 0 && row1 < global_height && row1 >=0
      && row2 < global_height && row2 >=0);

  int aIndex = find(next,row1,col1);
  int aRow = aIndex / global_width;
  int aCol = aIndex % global_width;

  int bIndex = find(next,row2,col2);
  int bRow = bIndex / global_width;
  int bCol = bIndex % global_width;

  if(aRow != bRow || aCol != bCol){
    Pixel A = pixels[aRow][aCol];
    int aSize = size[aRow][aCol];

    Pixel B = pixels[bRow][bCol];
    int bSize = size[bRow][bCol];

    if(mergeCriterion(A,B,30)){
      printf("A r: %d g: %d b: %d B r: %d g: %d b: %d\n",A.r,A.g,A.b,B.r,B.g,B.b);
      if(aSize>bSize){
        pixels[aRow][aCol] = newColor(A,B, aSize, bSize);
        printf("newCol r: %d g: %d b: %d\n",pixels[aRow][aCol].r,pixels[aRow][aCol].g,pixels[aRow][aCol].b);
        next[bRow][bCol] = aIndex;
        size[aRow][aCol] += bSize;
      }
      else{
        pixels[bRow][bCol] = newColor(A, B, aSize, bSize);

        printf("newCol r: %d g: %d b: %d\n",pixels[aRow][aCol].r,pixels[aRow][aCol].g,pixels[aRow][aCol].b);
        next[aRow][aCol] = bIndex;
        size[bRow][bCol] += aSize;
      }
    }
  }
  return;

}

void process(std::vector<std::vector<Pixel>>& pixels,int width, int height){
  global_height = height;
  global_width = width;
  std::vector<std::vector<int>> next(height,std::vector<int>(width,-1));
  std::vector<std::vector<int>> size(height,std::vector<int>(width,1));
  int start = 0;
  int offset = 2;
  int limit;
  while(start < width - 1 || start < height -1){
    //Comparing along row
    for(int y =0;y<height;y++){
      for(int x = start;x<=width-offset;x+=offset){
        verify_edge(pixels,next,size,x,y,x+1,y);
      }
    }
    std::cout<<"row comparison done"<<std::endl;
    int count = 0;
    for(int y = 0; y<height; y*=2){
      if(count == 1){
        y = offset/2;
      }
      for(int x = start;x<=width-offset;x+=offset){
        limit = offset/2 - 1;
        //guarantee y+limit <= height
        if(y + limit > height){
          limit = height - y -1;
        }
        for(int n=0;n<limit;n++){
          verify_edge(pixels,next,size,x,y+n,x+1,y+n+1);
          verify_edge(pixels,next,size,x+1,y+n,x,y+n+1);
        }
      }
      count++;
    }

    std::cout<<"second loop done"<<std::endl;
    for ( int y = start; y<=height-offset; y += offset){
      for(int x =0 ; x< width; x++){
        verify_edge(pixels,next,size,x,y,x,y+1);
      }  
    }
    std::cout<<"third loop done"<<std::endl;
    for(int y = start; y <= height-offset; y+= offset){
      for(int x = 0; x<= width-offset; x+=offset){
        limit = offset -1;
        if(x+limit>width){
          limit =width - x -1;
        }
        for( int n = 0;n<limit;n++){
          verify_edge(pixels,next,size,x+n,y,x+n+1,y+1);
          verify_edge(pixels,next,size,x+n+1,y,x+n,y+1);
        }
      }
    }
    start = 2*(start+1)-1;
    offset *=2;
  }

  for(int i=0;i<height;i++){
    for(int j=0;j<width;j++){
      int index = find(next,i,j);
      int row = index / global_width;
      int col = index % global_width;
      pixels[i][j]=pixels[row][col];
    }
  }
  return;
}

void cu_process(std::vector<std::vector<Pixel>> &pixels, int width, int height){

  return;
}
