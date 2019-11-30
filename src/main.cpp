#include "preprocess.h"
#include <iostream>
#include <tuple>
#include <vector>

using namespace std;
int main(){
  cout<<"before preprocess"<<endl;
  tuple<vector<vector<vector<int>>>,int,int> imgData = 
    preprocess("img/Cathedral_of_Learning.jpg");
  int height = get<1>(imgData);
  int width = get<2>(imgData);
  vector<vector<vector<int>>> img = get<0>(imgData);
  for(int i=0;i<height;i++){
    for(int j=0;j<width;j++){
      cout<<"row: "<<i<<" col: "<<j<<" ("<<img[i][j][0]<<","<<img[i][j][1]<<","<<img[i][j][2]<<")"<<endl;
    }
  }
  cout<<"after preprocess"<<endl;
  return 0;
}
