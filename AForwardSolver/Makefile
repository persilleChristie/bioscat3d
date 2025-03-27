# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2

# Only look for .cpp files in the current directory
SOURCES = main.cpp \
          Dipole.cpp \
          FieldCalculatorDipole.cpp \
          FieldCalculatorUPW.cpp \
          SurfaceSphere.cpp \
          SystemAssembler.cpp \
          SystemSolver.cpp \
          LinearSystemSolver.cpp
# Remove Surfacemain.cpp from this list
OBJECTS := $(SOURCES:.cpp=.o)




# Output binary
TARGET = scatterSim

# Build target
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) *.o FilesCSV/*.csv
