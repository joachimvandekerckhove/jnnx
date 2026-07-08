# Stand-alone unit tests for jnnx/cpp/sl_math.cpp

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -I../../jnnx/cpp

SL_MATH_SRC = ../../jnnx/cpp/sl_math.cpp
TEST_SRC = test_sl_math.cpp

.PHONY: all test clean

all: test_sl_math

test_sl_math: $(SL_MATH_SRC) $(TEST_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_SRC) $(SL_MATH_SRC)

test: test_sl_math
	./test_sl_math

clean:
	rm -f test_sl_math
