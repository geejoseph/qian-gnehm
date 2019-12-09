// Verifies edge between pixel (x1, y2) and pixel (x2, y2)
verify_edge(int x1, int y1, int x2, int y2);

int width, height; // given

int iteration = 0;
int start = 0;
int offset = 2;

int x, y, limit, n;
while (start < width-1 || start < height-1){
  for (y = 0; y < height; y++) {
    for (x = start; x <= width-2; x += offset) {
      verify_edge(x, y, x+1, y);
    }
  }

  // y = 0, offset/2, offset, 2*offset, 4*offset, ..., height - 2
  int count = 0;
  for (y = 0; y <= height-2; y *= 2) {
    if (count == 1) {
      y = offset/2;
    }
    for (x = start; x <= width-2; x += offset) {
      limit = offset/2 - 1;
      if (y+limit >= height) {
        limit = height-y-2;
      }

      for (n = 0; n < limit; n++) {
        verify_edge(x,  my+n, x+1, y+n+1);
        verify_edge(x+1, y+n, x,   y+n+1);
      }
      count++;
    }
  }

  for (y = start; y <= height-2; y += offset) {
    for (x = 0; x < width; x++) {
      verify_edge(x, y, x, y+1);
    }
  }

  for (y = start; y <= height-2; y += offset) {
    for (x = 0; x <= width-2; x += offset) {
      limit = offset-1;
      if (x+limit >= width) {
        limit = width-x-2;
      }

      for (n = 0; n < limit; n++) {
        verify_edge(x+n,   y, x+n+1, y+1);
        verify_edge(x+n+1, y, x+n,   y+1);
      }
    }
  }

  iteration++;
  start = 2*(start+1)-1; // start = 2^iteration - 1
  offset *= 2;
}
