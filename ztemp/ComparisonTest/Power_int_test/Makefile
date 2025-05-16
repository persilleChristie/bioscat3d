CXX = g++
CXXFLAGS = -std=c++17 -I/usr/include/eigen3 -I/home/pernillechristie/rapidjson/include
TARGET = PN_power_int

SRC = PN_power_int.cpp ../../ForwardSolver/FieldCalculatorTotal.cpp ../../ForwardSolver/MASSystem.cpp ../../ForwardSolver/SurfacePlane.cpp ../../ForwardSolver/FieldCalculatorDipole.cpp ../../ForwardSolver/FieldCalculatorUPW.cpp ../../ForwardSolver/SystemAssembler.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
