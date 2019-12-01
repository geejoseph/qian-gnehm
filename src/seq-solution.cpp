#include <iostream>
#include <string>
#include <vector>
#include "pixel.h"
#include "metadata.h"
#include "imgSeg.h"

inline bool mergeCriterion(Pixel p1, Pixel p2, int t){
  return (p1.r - p2.r)*(p1.r - p2.r) + (p1.g - p2.g)*(p1.g - p2.g) +
    (p1.b - p2.b) * (p1.b - p2.b) < t*t;
}

Pixel newColor(Pixel A, Pixel B, Metadata findA, Metadata findB){
  int totalSize = findA.size + findB.size;
  int newR = (A.r * findA.size + B.r * findB.size)/(totalSize);
  int newG = (A.g * findA.size + B.g * findB.size)/(totalSize);
  int newB = (A.b * findA.size + B.b * findB.size)/(totalSize);
  return Pixel(newR,newG,newB);
}

Metadata find(std::vector<std::vector<Metadata>> &meta,int srow,int scol){
  int row = srow;
  int col = scol;
  while(1){
    Metadata m = meta[row][col];
    if(m.nrow == row && m.ncol == col){
      meta[srow][scol].nrow = row;
      meta[srow][scol].ncol = col;
      return m;
    }
    else if (meta[row][col].nrow == -1 && meta[row][col].ncol == -1){
      meta[row][col].nrow = row;
      meta[row][col].ncol = col;
      meta[srow][scol].nrow = row;
      meta[srow][scol].ncol = col;
      return meta[row][col];
    }
    row = meta[row][col].nrow;
    col = meta[row][col].ncol;
  }
}

void verify_edge(std::vector<std::vector<Pixel>> &pixels, std::vector<std::vector<Metadata>> &meta, int row1, int col1, int row2, int col2) {
  Metadata findA = find(meta,row1,col1);
  Metadata findB = find(meta,row2,col2);
  if(findA.nrow != findB.nrow || findA.ncol != findB.ncol){
    Pixel A = pixels[findA.nrow][findA.ncol];
    Pixel B = pixels[findB.nrow][findB.ncol];
    if(mergeCriterion(A,B,4)){
      if(findA.size>findB.size){
        pixels[findA.nrow][findA.ncol] = newColor(A,B, findA, findB);
        findB.nrow = findA.nrow;
        findB.ncol = findA.ncol;
        findA.size += findB.size;
      }
      else{
        pixels[findB.nrow][findB.nrow] = newColor(A, B, findA, findB);
        findA.nrow = findB.nrow;
        findA.ncol = findB.ncol;
        findB.size += findB.size;
      }
    }
  }
  return;

}


void process(std::vector<std::vector<Pixel>>& pixels,int width, int height){
  std::vector<std::vector<Metadata>> meta(height,std::vector<Metadata>(width));
  int start = 0;
  int offset = 2;
  int limit;
  while(start < width - 1 || start < height -1){
    //Comparing along row
    for(int y =0;y<height;y++){
      for(int x = start;x<width;x+=offset){
        verify_edge(pixels,meta,x,y,x+1,y);
      }
    }
    int count = 0;
    for(int y = 0; y<height; y*=2){
      if(count == 1){
        y = offset/2;
      }
      for(int x = start;x<width;x+=offset){
        limit = offset/2 - 1;
        //guarantee y+limit <= height
        if(y + limit > height){
          limit = height - y -1;
        }
        for(int n=0;n<limit;n++){
          verify_edge(pixels,meta,x,y+n,x+1,y+n+1);
          verify_edge(pixels,meta,x+1,y+n,x,y+n+1);
        }
      }
      count++;
    }

    for ( int y = start; y<height; y += offset){
      for(int x =0 ; x< width; x++){
        verify_edge(pixels,meta,x,y,x,y+1);
      }  
    }

    for(int y = start; y < height; y+= offset){
      for(int x = 0; x<width; x+=offset){
        limit = offset -1;
        if(x+limit>width){
          limit =width - x -1;
        }
        for( int n = 0;n<limit;n++){
          verify_edge(pixels,meta,x+n,y,x+n+1,y+1);
          verify_edge(pixels,meta,x+n+1,y,x+n,y+1);
        }
      }
    }
    start = 2*(start+1)-1;
    offset *=2;
  }
  return;
}
