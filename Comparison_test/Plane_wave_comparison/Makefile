CXX = g++
CXXFLAGS = -std=c++17 -I/usr/include/eigen3 -I../../AForwardSolver/
TARGET = PN_plane_wave_solver

SRC = PN_plane_wave_solver.cpp ../../AForwardSolver/FieldCalculatorUPW.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
