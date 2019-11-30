
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

OBJS=$(OBJDIR)/main.o $(OBJDIR)/preprocess.o 

$(APP_NAME): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) $(LIBS) -o $@ $(OBJS) -lm -lpthread

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o  $@

#$(OBJDIR)/%.o: %.cpp
#	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: preprocess.h

$(OBJDIR)/preprocess.o: preprocess.h

#$(OBJDIR)/main.o: CycleTimer.h

