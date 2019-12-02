// Verifies edge between pixel (x1, y2) and pixel (x2, y2)
verify_edge(int x1, int y1, int x2, int y2);

struct region {
    
}
// Need pixel/node struct. To store size, next, rgb color
// New color
// Merge criterion
// Need to implement find and join

int width, height; // given

int iteration = 0;
int start = 0;
int offset = 2;

// Can change order of sweep, paper order is biased to horizontal?


int x, y, limit, n;
while (start < width - 1 || start < height - 1){
  for (y = 0; y < height; y++) {
    for (x = start; x <= width-offset; x += offset) {
      verify_edge(x, y, x + 1, y);
    }
  }
  
  // HAD TO SPLIT LOOP INTO 2 to have:
  // y = 0, offset/2, offset, 2*offset, 4*offset, ..., height - 2
  
  // y = 0
  y = 0;
  for (x = start; x < width; x += offset) {
    limit = offset/2 - 1;
    if (y + limit > height) { // limit > height
      limit = height - y - 1; // limit = height - 1 
    }
    
    for (n = 0; n < limit; n++) {
      verify_edge(x, y + n, x + 1, y + n + 1);
      verify_edge(x + 1, y + n, x, y + n + 1);
    }
  }

  // y = offset/2, offset, 2*offset, 4*offset, ..., height - 2 (ceiling)
  for (y = offset/2; y < height; y *= 2) ;  // careful loops forever if offset = 0
    for (x = start; x < width; x += offset) {
      limit = offset/2 - 1;
      if (y + limit > height) {
        limit = height - y - 1;
      }
      
      for (n = 0; n < limit; n++) {
        verify_edge(x, y + n, x + 1, y + n + 1);
        verify_edge(x + 1, y + n, x, y + n + 1);
      }
    }
  }

  // Rest is normal
  
  for (y = start; y < height; y += offset) {
    for (x = 0; x < width; x++) {
      verify_edge(x, y, x, y + 1);
    }
  }
  
  for (y = start; y < height; y += offset) {
    for (x = 0; x < width; x += offset) {
      limit = offset - 1;
      if (x + limit > width) {
        limit = width - x - 1; 
      }
      
      for (n = 0; n < limit; n++) {
        verify_edge(x + n, y, x + n + 1, y + 1);
        verify_edge(x + n + 1, y, x + n, y + 1);
      }
    }
  }
            
  iteration++;
  start = 2*(start + 1) - 1; // start = 2^iteration - 1
  offset *= 2;
}

// Post-processing: get rid of regions that are too small
