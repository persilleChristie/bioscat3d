.PHONY: all clean

CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 \
  -I/usr/include/eigen3 \
  -I/home/tripleskull/libs/rapidjson/include \
  -I.. \
  -I../ForwardSolver \
  -I/home/pernillechristie/rapidjson/include

TARGET = CrankNicolson

SRC = mainCrankNicolson.cpp \
      ../MASSystem.cpp \
      ../FieldCalculatorDipole.cpp \
      ../FieldCalculatorUPW.cpp \
      ../FieldCalculatorTotal.cpp \
      ../SystemAssembler.cpp \
      ../SurfacePlane.cpp \
      ../CrankNicolson.cpp

# Extract object file names from sources (in current dir)
OBJ = $(notdir $(SRC:.cpp=.o))

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^
	@echo "✅ Build complete: $(TARGET)"

# Generic rule to compile each .cpp to .o
%.o: ../%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

mainCrankNicolson.o: mainCrankNicolson.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
