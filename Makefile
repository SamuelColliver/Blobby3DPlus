# set the path to DNest4
DNEST4_PATH = ..

# use clang++ with libc++ to match DNest4 compilation
CXX = clang++ -stdlib=libc++ -isysroot $(shell xcrun --show-sdk-path)
CC = clang -isysroot $(shell xcrun --show-sdk-path)

FFTW_PREFIX = /opt/homebrew/opt/fftw
CXXFLAGS = -std=c++11 -O3 -Wall -Wextra -pedantic -DNDEBUG -I$(FFTW_PREFIX)/include -stdlib=libc++
LIBS = -ldnest4 -lpthread -lfftw3
LDFLAGS = -L$(FFTW_PREFIX)/lib -stdlib=libc++ -lc

default:
	make noexamples -C $(DNEST4_PATH)/DNest4/code
	$(CXX) -I $(DNEST4_PATH) $(CXXFLAGS) -c src/*.cpp
	$(CXX) -pthread -L $(DNEST4_PATH)/DNest4/code $(LDFLAGS) -o Blobby3D *.o $(LIBS)
	rm *.o

nolib:
	$(CXX) -I $(DNEST4_PATH) $(CXXFLAGS) -c src/*.cpp
	$(CXX) -pthread -L $(DNEST4_PATH)/DNest4/code $(LDFLAGS) -o Blobby3D *.o $(LIBS)
	rm *.o

clean:
	rm -f *.o
	rm -f Blobby3D