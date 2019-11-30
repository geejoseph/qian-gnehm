#include <iostream>
#include <string>
#include <vector>
#include <pixel.h>


inline bool merge(Pixel p1, Pixel p2, t){
  return (p1.r - p2.r)*(p1.r - p2.r) + (p1.g - p2.g)*(p1.g - p2.g) +
    (p1.b - p2.b) * (p1.b - p2.b) < t*t;
}

int find(int index[][],int srow,int scol){
  int row = srow;
  int col = scol;
  if(index[row][col]== row*width +col){
    index[srow][scol] = row *width + col;
    return row * width + col;
  }
  else if (index[row][col] == -1){
    index[row][col] = row * width + col;
    index[srow][scol] = row * width + col;
    return row * width + col;
  }

}

bool verify_edge(std::vector<std::vector<Pixel>> pixels, int row1, int col1, int row2, int col2) {
  

}

struct region {



}

void process(std::vector<std::vector<Pixel>> pixels,int width, int height){
  int index[width][height] = { -1 };
  int iteration = 0;
  int start = 0;
  int offset = 2;

}
