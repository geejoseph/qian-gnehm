
CXX=g++ -m64 -std=c++11
CXXFLAGS=-Iobjs/ -O3 -Wall
LIBS += -L/usr/local/lib -ljpeg

APP_NAME=solution
OBJDIR=objs
default: $(APP_NAME)

.PHONY: dirs clean

dirs:
		/bin/mkdir -p $(OBJDIR)/

clean:
		/bin/rm -rf $(OBJDIR) *~ $(APP_NAME)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/preprocess.o $(OBJDIR)/seq-solution.o $(OBJDIR)/blur.o

$(APP_NAME): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) $(LIBS) -o $@ $(OBJS) -lm -lpthread -fopenmp

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o  $@

#$(OBJDIR)/%.o: %.cpp
#	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: preprocess.h imgSeg.h blur.h

$(OBJDIR)/preprocess.o: preprocess.h

$(OBJDIR)/seq-solution.o: imgSeg.h

$(OBJDIR)/blur.o: blur.h
#$(OBJDIR)/main.o: CycleTimer.h

