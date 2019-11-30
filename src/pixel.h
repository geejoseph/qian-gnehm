#ifndef __PIXEL_H_
#define __PIXEL_H_


class Pixel
{
  public:
    int r,g,b;
    Pixel() = default;
    Pixel(int x,int y, int z){
      r=x;
      g=y;
      b=z;
    }
    inline int getR(const Pixel x){
      return x.r;
    }
    inline int getG(const Pixel x){
      return x.g;
    }
    inline int getB(const Pixel x){
      return x.b;
    }
};
#endif
