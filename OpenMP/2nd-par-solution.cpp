#include <iostream>
#include <string>
#include <vector>
#include "pixel.h"
#include "imgSeg.h"
#include <cstdint>
#include <cmath>
#include <assert.h>
#include <omp.h>
double omp_get_wtime(void);

int t = 30;
//int chunkfactor = 20;

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
  //assert(col1< global_width && col1>=0 && col2 < global_width && col2 >= 0 && row1 < global_height && row1 >=0
      //&& row2 < global_height && row2 >=0);

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

// Parallelizing as much as can with one task per "worst-case region area"
void process(std::vector<std::vector<Pixel>>& pixels, int width, int height){
  double start_time, end_time, alloc_start_time, alloc_end_time,
         sweeps_start_time, sweeps_end_time, iteration_start_time,
         iteration_end_time, update_start_time, update_end_time, process_time,
         alloc_time, sweeps_time, update_time;
  start_time = omp_get_wtime();
  global_height = height;
  global_width = width;

  // Allocation
  alloc_start_time = omp_get_wtime();
  std::vector<std::vector<int>> next(height,std::vector<int>(width,-1));
  std::vector<std::vector<int>> size(height,std::vector<int>(width,1));
  alloc_end_time = omp_get_wtime();
  int start = 0;
  int offset = 2;
  int iteration = 0;
  int limit;

  //int chunksize;
  //int n_procs = omp_get_num_procs();

  int run_length = 1; //each run is a piece of split up row or col,
  // this is the length of the dimension along which we move in the loop
  // n_regions halves after every col and row sweep
  // but var needs to be halved before to keep track
  // run_length doubles before every row sweep
  int n_cols, n_regions;

  // Sweeping
  sweeps_start_time = omp_get_wtime();
  while (start < width - 1 || start < height -1){
    iteration_start_time = omp_get_wtime();

    //Comparing along row
    //chunksize = std::max(1, /(n_procs*chunkfactor));
    n_cols = (width - 1 - start)/offset;
    n_regions = n_cols*height;
    //#pragma omp parallel for //schedule(dynamic, chunksize)
    for (int r = 0; r < n_regions; r++) {
      int x = start + (r/height)*offset;
      //for (int x = start; x <= width - 2; x += offset){
      int y_start = r % height;
      int y_end = std::min(height, y_start + run_length);
      for (int y = y_start; y < y_end; y++){
        verify_edge(pixels,next,size,x,y,x+1,y);
      }
    }
    //std::cout<<"row comparison done"<<std::endl;

    // y = 0, offset/2, offset, 2*offset, 4*offset, ..., height - 2
    //chunksize = std::max(1, /(n_procs*chunkfactor));
    int count;
    #pragma omp parallel for private(limit, count) //schedule(dynamic, chunksize)
    for(int x = start; x <= width-2; x += offset){
      count = 0;
      for(int y = 0; y <= height-2; y *= 2){
        if (count == 1) {
          y = offset/2;
          if (y > height - 2) {
            break;
          }
        }
        limit = offset/2 - 1;
        //guarantee y+limit <= height
        if(y + limit >= height){
          limit = height - y - 2;
        }
        for(int n = 0; n < limit; n++){
          verify_edge(pixels,next,size,x,  y+n,x+1,y+n+1);
          verify_edge(pixels,next,size,x+1,y+n,x,  y+n+1);
        }
        count++;
      }
    }
    //std::cout<<"second loop done"<<std::endl;

    // Swapped x and y
    //chunksize = std::max(1, /(n_procs*chunkfactor));
    #pragma omp parallel for //schedule(dynamic, chunksize)
    for (int y = start; y <= height-2; y += offset){
      for (int x = 0; x < width; x++) {
        verify_edge(pixels,next,size,x,y,x,y+1);
      }
    }
    //std::cout<<"third loop done"<<std::endl;

    //chunksize = std::max(1, /(n_procs*chunkfactor));
    #pragma omp parallel for private(limit) //schedule(dynamic, chunksize)
    for (int y = start; y <= height-2; y += offset){
      for (int x = 0; x <= width-2; x += offset) {
        limit = offset - 1;
        if (x + limit >= width) {
          limit = width - x - 2;
        }
        for (int n = 0; n < limit; n++){
          verify_edge(pixels,next,size,x+n,y,x+n+1,y+1);
          verify_edge(pixels,next,size,x+n+1,y,x+n,y+1);
        }
      }
    }

    start = 2*(start+1)-1;
    offset *= 2;
    iteration_end_time = omp_get_wtime();
    printf("iteration %d took %f ms\n", iteration,
            (iteration_end_time - iteration_start_time)*1000);
    iteration++;
  }
  sweeps_end_time = omp_get_wtime();

  // Updating final colors for all pixels
  update_start_time = omp_get_wtime();
  #pragma omp parallel for //schedule(static)
  for (int i=0;i<height;i++){
      for (int j=0;j<width;j++){
      int index = find(next,i,j);
      int row = index / global_width;
      int col = index % global_width;
      pixels[i][j]=pixels[row][col];
    }
  }
  end_time = omp_get_wtime();
  update_end_time = end_time;
  process_time = end_time - start_time;
  sweeps_time = sweeps_end_time - sweeps_start_time;
  alloc_time = alloc_end_time - alloc_start_time;
  update_time = update_end_time - update_start_time;
  printf("process: %f ms\nsweeps: %f\nalloc: %f ms\nupdate: %f ms\n",
          process_time*1000, sweeps_time*1000, alloc_time*1000,
          update_time*1000);
  return;
}
