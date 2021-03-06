#include <iostream>
#include <string>
#include <vector>
#include "pixel.h"
#include "imgSeg.h"
#include <cstdint>
#include <assert.h>
#include <omp.h>
double omp_get_wtime(void);

int t = 30;

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
  //printf("A r: %d A g: %d A b: %d B r: %d B g: %d B b: %d A size: %d B size: %d n r: %d n g: %d n b: %d\n",
  //    A.r,A.g,A.b,B.r,B.g,B.b,findA.size,findB.size,newP.r,newP.g,newP.b);
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

    if(mergeCriterion(A,B,t)){
      //printf("A r: %d g: %d b: %d B r: %d g: %d b: %d\n",A.r,A.g,A.b,B.r,B.g,B.b);
      if(aSize>bSize){
        pixels[aRow][aCol] = newColor(A,B, aSize, bSize);
        //printf("newCol r: %d g: %d b: %d\n",pixels[aRow][aCol].r,pixels[aRow][aCol].g,pixels[aRow][aCol].b);
        next[bRow][bCol] = aIndex;
        size[aRow][aCol] += bSize;
      }
      else{
        pixels[bRow][bCol] = newColor(A, B, aSize, bSize);

        //printf("newCol r: %d g: %d b: %d\n",pixels[aRow][aCol].r,pixels[aRow][aCol].g,pixels[aRow][aCol].b);
        next[aRow][aCol] = bIndex;
        size[bRow][bCol] += aSize;
      }
    }
  }
  return;

}

void process(std::vector<std::vector<Pixel>>& pixels,int width, int height){
  double start_time, end_time;
  start_time = omp_get_wtime();
  global_height = height;
  global_width = width;

  // Allocation
  double alloc_start_time, alloc_end_time;
  alloc_start_time = omp_get_wtime();
  std::vector<std::vector<int>> next(height,std::vector<int>(width,-1));
  std::vector<std::vector<int>> size(height,std::vector<int>(width,1));
  alloc_end_time = omp_get_wtime();
  int start = 0;
  int offset = 2;
  int limit;
  int iteration = 0;

  // Sweeping
  double sweeps_start_time, sweeps_end_time;
  //double iteration_start_time, iteration_end_time, middle_time,
  //       horizontal_time, diag1_time, vertical_time, diag2_time;
  sweeps_start_time = omp_get_wtime();
  //printf("iteration times (ms): ");
  while(start < width - 1 || start < height -1){
    //iteration_start_time = omp_get_wtime();
    //Comparing along row
    for(int y =0;y<height;y++){
      for(int x = start;x<=width-2;x+=offset){
        verify_edge(pixels,next,size,x,y,x+1,y);
      }
    }
    /*
    if (iteration == 1) {
      middle_time = omp_get_wtime();
      horizontal_time = middle_time - iteration_start_time;
    }
    */
    //std::cout<<"row comparison done"<<std::endl;
    // y = 0, offset/2, offset, 2*offset, 4*offset, ..., height - 2
    int count = 0;
    for(int y = 0; y<=height-2; y*=2){
      if (count == 1) {
        y = offset/2;
        if (y > height - 2) {
          break;
        }
      }
      for(int x = start;x<=width-2;x+=offset){
        limit = offset/2 - 1;
        //guarantee y+limit <= height
        if(y + limit >= height){
          limit = height - y -2;
        }
        for(int n=0;n<limit;n++){
          verify_edge(pixels,next,size,x,  y+n,x+1,y+n+1);
          verify_edge(pixels,next,size,x+1,y+n,x,  y+n+1);
        }
      }
      count++;
    }
    /*
    if (iteration == 1) {
      diag1_time = omp_get_wtime() - middle_time;
      middle_time += diag1_time;
    }
    */

    //std::cout<<"second loop done"<<std::endl;
    for ( int y = start; y<=height-2; y += offset){
      for(int x =0 ; x< width; x++){
        verify_edge(pixels,next,size,x,y,x,y+1);
      }
    }
    /*
    if (iteration == 1) {
      vertical_time = omp_get_wtime() - middle_time;
      middle_time += vertical_time;
    }
    */
    //std::cout<<"third loop done"<<std::endl;
    for(int y = start; y <= height-2; y+= offset){
      for(int x = 0; x<= width-2; x+=offset){
        limit = offset -1;
        if(x+limit>=width){
          limit =width - x -2;
        }
        for( int n = 0;n<limit;n++){
          verify_edge(pixels,next,size,x+n,y,x+n+1,y+1);
          verify_edge(pixels,next,size,x+n+1,y,x+n,y+1);
        }
      }
    }
    /*
    if (iteration == 1) {
      diag2_time = omp_get_wtime() - middle_time;
    }
    */
    start = 2*(start+1)-1;
    offset *=2;
    //iteration_end_time = omp_get_wtime();
    //printf("%f ", (iteration_end_time - iteration_start_time)*1000);
    iteration++;
  }
  //printf("\n");
  sweeps_end_time = omp_get_wtime();

  // Updating final colors for all pixels
  double update_start_time, update_end_time;
  update_start_time = omp_get_wtime();
  for(int i=0;i<height;i++){
    for(int j=0;j<width;j++){
      int index = find(next,i,j);
      int row = index / global_width;
      int col = index % global_width;
      pixels[i][j]=pixels[row][col];
    }
  }
  end_time = omp_get_wtime();
  update_end_time = end_time;
  double process_time, sweeps_time, alloc_time, update_time;
  process_time = end_time - start_time;
  sweeps_time = sweeps_end_time - sweeps_start_time;
  alloc_time = alloc_end_time - alloc_start_time;
  update_time = update_end_time - update_start_time;
  printf("process: %f ms\nsweeps: %f\nalloc: %f ms\nupdate: %f ms\n",
          process_time*1000, sweeps_time*1000, alloc_time*1000,
          update_time*1000);
  /*
  printf("second iteration: %f %f %f %f\n", horizontal_time*1000,
          diag1_time*1000, vertical_time*1000, diag2_time*1000);
  */
  return;
}
