print-%: ; @echo $*=$($*)

dir_guard = @mkdir -pv $(@D)

PTPACK_HOME = ..

include_dirs = $(PTPACK_HOME)/include $(PTPACK_HOME)/3rd-party

lib_dir = $(PTPACK_HOME)/build/lib
ptpack_lib = $(lib_dir)/libptpack.a

CXXFLAGS := -Wall -O3 -m64 -std=c++11
CXXFLAGS += $(addprefix -I,$(include_dirs))

CXX := g++

all : build/lowrank_hawkes_lastfm build/learning_lowrank_hawkes

build/learning_lowrank_hawkes : learning_lowrank_hawkes.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

build/lowrank_hawkes_lastfm : lowrank_hawkes_lastfm.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

clean:
	rm -rf build

-include $(DEPS)