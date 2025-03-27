# Makefile for EM Scattering Simulation

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -I./ -I/usr/include/eigen3

SRCS = \
    main.cpp \
    SystemSolver.cpp \
    SystemAssembler.cpp \
    FieldCalculatorDipole.cpp \
    FieldCalculatorUPW.cpp \
    SurfaceSphere.cpp \
    Dipole.cpp

OBJS = $(SRCS:.cpp=.o)

EXEC = EMScatteringSim

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $(EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(EXEC)
	./$(EXEC)

clean:
	rm -f $(OBJS) $(EXEC)
