CXX = g++
CXXFLAGS = -std=c++17 -I/usr/include/eigen3 -I../../ForwardSolver/
TARGET = PN_dipole_solver

SRC = PN_dipole_solver.cpp ../../ForwardSolver/FieldCalculatorDipole.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
