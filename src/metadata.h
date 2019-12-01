#ifndef __METADATA_H_
#define __METADATA_H_

class MetaDdta{
  public:
    int nrow,ncol,size;
    Metadata(int row=-1, int col=-1, int size2=1){
      nrow = row;
      ncol = col;
      size = size2;
    }
}


#endif
