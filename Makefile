CXX = g++
CXXFLAGS = -std=c++11 -Wall -O2
HDRS = LidDrivenCavity.h SolverCG.h 
OBJS = LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o
LDLIBS = -lblas -lboost_program_options 

%.o : %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

solver: $(OBJS)
	$(CXX) -o $@ $^ $(LDLIBS)

doc:
	doxygen Doxyfile
all: solver