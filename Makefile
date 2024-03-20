CC = mpicxx
CXXFLAGS = -std=c++14 -Wall -O2 -fopenmp -g
HDRS = LidDrivenCavity.h SolverCG.h 
OBJS = LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o
LDLIBS = -lblas -lboost_program_options -fopenmp

%.o : %.cpp $(HDRS)
	$(CC) $(CXXFLAGS) -o $@ -c $<

solver: $(OBJS)
	$(CC) -o $@ $^ $(LDLIBS)

doc:
	doxygen Doxyfile

all: solver

clean: -rm -f $(OBJS) solver
