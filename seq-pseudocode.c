// Predefined functions and values

// Verifies edge between pixel (x1, y2) and pixel (x2, y2)
verify_edge(int x1, int y1, int x2, int y2);

int width, height;

// Begin pseudocode

int iteration = 0;
int start = 0;
int offset = 2;
while (start < width - 1 || start < height - 1){
  for (int y = 0; y < height; y++) {
    for (int x = start; x < width; x += offset) {
      verify_edge(x, y, x + 1, y);
    }
  }
  
  // LOOP BELOW NOT DONE..
  for (int y = 0;  // not done, weird?
    for (int x = start; x < width; x += offset) {
      int limit = offset/2 - 1;
      if (y + limit > height) {
        limit = height - y - 1;
      }
      
      for (int n = 0; n < limit; n++) {
        verify_edge(x, y + n, x + 1, y + n + 1);
        verify_edge(x + 1, y + n, x, y + n + 1);
      }
    }
  }
  
  for (y = start; y < height; y += offset) {
    for (x = 0; x < width; x++) {
      verify_edge(x, y, x, y + 1);
    }
  }
  
  for (y = start; y < height; y += offset) {
    for (x = 0; 
  }
  
  iteration++;
  start = 2*(start + 1) - 1;
  offset *= 2;
}
