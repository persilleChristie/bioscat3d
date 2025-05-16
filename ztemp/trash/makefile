IDIR ="C:/vcpkg/installed/x64-windows/include/eigen3"
CC=g++
CFLAGS=-I$(IDIR)

#ODIR=obj
#LDIR =../lib

#LIBS=-lm

DEPS = matrixToCSVfile.h
OBJ = matrixToCSVfile.o testSphere.o 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

hellomake: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)