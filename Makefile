CC = mpicxx
CXXFLAGS = -std=c++14 -Wall -g -O2 -ftree-vectorize -fopenmp 
HDRS = LidDrivenCavity.h SolverCG.h 
OBJS = LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o
LDLIBS = -lblas -lboost_program_options -fopenmp

%.o : %.cpp $(HDRS)
	$(CC) $(CXXFLAGS) -o $@ -c $<

solver: $(OBJS)
	$(CC) -o $@ $^ $(LDLIBS)

solver_test: solver_test.o $(OBJS)
	$(CC) $(CXXFLAGS) -o $@ $^ $(LDLIBS) -lboost_unit_test_framework

test: solver_test
	./solver_test

doc:
	doxygen Doxyfile

all: solver solver_test

clean:
	-rm -f $(OBJS) solver solver_test solver_test.o


