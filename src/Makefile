
CXX=g++ -m64 -std=c++11
CXXFLAGS=-Iobjs/ -O3 -Wall
LIBS += -L/usr/local/lib -ljpeg
LDFLAGS =-L/usr/local/depot/cuda-8.0/lib64/ -lcudart
NVCC = nvcc -std=c++11
NVCCFLAGS = -O3 -m64 --gpu-architecture compute_35
APP_NAME=solution
OBJDIR=objs
default: $(APP_NAME)

.PHONY: dirs clean

dirs:
		/bin/mkdir -p $(OBJDIR)/

clean:
		/bin/rm -rf $(OBJDIR) *~ $(APP_NAME)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/preprocess.o $(OBJDIR)/seq-solution.o $(OBJDIR)/blur.o $(OBJDIR)/cu-solution.o

$(APP_NAME): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) $(LIBS) -o $@ -lm -lpthread $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o  $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
#$(OBJDIR)/%.o: %.cpp
#	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: preprocess.h imgSeg.h blur.h

$(OBJDIR)/preprocess.o: preprocess.h

$(OBJDIR)/seq-solution.o: imgSeg.h

$(OBJDIR)/blur.o: blur.h
#$(OBJDIR)/main.o: CycleTimer.h

