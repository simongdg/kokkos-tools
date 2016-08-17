CXX=g++
CXXFLAGS=-O3 -std=c++11 -g
SHARED_CXXFLAGS=-shared -fPIC

all: kp_kernel_autoTuner.so kp_reader

kp_reader: kp_reader.cpp kp_kernel_autoTuner.so
	$(CXX) $(CXXFLAGS) -o kp_reader kp_reader.cpp

kp_kernel_autoTuner.so: kp_kernel_autoTuner.cpp kp_kernel_info.h
	$(CXX) $(SHARED_CXXFLAGS) $(CXXFLAGS) -o $@ kp_kernel_autoTuner.cpp

clean:
	rm *.so kp_reader
