CXX      := g++
CXXFLAGS := -std=c++17 -Wall -O2

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    CXXFLAGS += -I/usr/include/eigen3
endif
ifeq ($(UNAME_S),Darwin)
    CXXFLAGS += -I/usr/local/include/eigen3
endif

SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)
TARGET = scatterSim

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	-rm -f $(TARGET) *.o
