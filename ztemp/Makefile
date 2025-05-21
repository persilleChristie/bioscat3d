PYTHON_BIN := /home/tripleskull/miniconda/envs/mp/bin/python
PYTHON_VER := 3.8
PYLIB_DIR  := /home/tripleskull/miniconda/envs/mp/lib

CXX := c++
CXXFLAGS := -O3 -Wall -std=c++17 $(shell $(PYTHON_BIN) -m pybind11 --includes)
LDFLAGS  := -L$(PYLIB_DIR) -Wl,-rpath,$(PYLIB_DIR) -lpython$(PYTHON_VER)

main: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o main $(LDFLAGS)

clean:
	rm -f main

# Makefile to build main.cpp using active Python env with pybind11

# PYTHON := $(shell which python)
# PYTHON_VER := $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
# PYTHON_LIB := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

# CXX := c++
# CXXFLAGS := -O3 -Wall -std=c++17 $(shell $(PYTHON) -m pybind11 --includes)
# LDFLAGS := -L$(PYTHON_LIB) -Wl,-rpath,$(PYTHON_LIB) -lpython$(PYTHON_VER)

# main: main.cpp
# 	$(CXX) $(CXXFLAGS) main.cpp -o main $(LDFLAGS)

# clean:
# 	rm -f main

# PYTHON := /home/tripleskull/miniconda/envs/mp312/bin/python
# PYLIB  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
# SITEPKG := $(shell $(PYTHON) -c "import site; print(site.getsitepackages()[0])")

# CXX := c++
# CXXFLAGS := -O3 -Wall -std=c++17 $(shell $(PYTHON) -m pybind11 --includes) -DSITE_PACKAGES=\"$(SITEPKG)\"
# LDFLAGS := -L$(PYLIB) -Wl,-rpath,$(PYLIB) -lpython3.12

# main: main.cpp
# 	$(CXX) $(CXXFLAGS) main.cpp -o main $(LDFLAGS)

# clean:
# 	rm -f main


