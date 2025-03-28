# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2

# Detect OS and set platform-specific options
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
    # On Linux, Eigen is usually installed via apt in /usr/include/eigen3
    CXXFLAGS += -I/usr/include/eigen3
endif

# Source files (no Surfacemain.cpp)
SOURCES = main.cpp \
          Dipole.cpp \
          FieldCalculatorDipole.cpp \
          FieldCalculatorUPW.cpp \
          SurfaceSphere.cpp \
          SystemAssembler.cpp \
          SystemSolver.cpp \
          LinearSystemSolver.cpp

OBJECTS := $(SOURCES:.cpp=.o)

# Output binary
TARGET = scatterSim

# Default target
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) *.o FilesCSV/*.csv
