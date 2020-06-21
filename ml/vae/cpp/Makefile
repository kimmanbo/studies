# Makefile: vaecpp

TORCH_DIR = /usr/local/package/libtorch
TORCH_INCDIR = $(TORCH_DIR)/include
TORCH_INCDIR += $(TORCH_DIR)/include/torch/csrc/
TORCH_INCDIR += $(TORCH_DIR)/include/torch/csrc/api/include
TORCH_LIBDIR = $(TORCH_DIR)/lib
TORCH_LIBS := -L$(TORCH_LIBDIR) -ltorch -ltorch_cpu -lc10
TORCH_LIBS += -ltorch_cuda -lc10_cuda

CUDA_DIR = /usr/local/cuda
CUDA_INCDIR = $(CUDA_DIR)/include
CUDA_LIBDIR = $(CUDA_DIR)/lib64
CUDA_LIBS := -L$(CUDA_LIBDIR) -lcudnn -lcudart -lcurand -lcusolver
CUDA_LIBS += -L$(CUDA_LIBDIR)/stubs -lcuda

TARGET = main

SRC_DIRS ?= ./
BUILD_DIR ?= ./build

SRCS := $(shell find $(SRC_DIRS) \( -name *.cpp -print -or -name *.c -print \))
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) \(-type d -print\)) $(TORCH_INCDIR) $(CUDA_INCDIR)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

LDFLAGS = $(TORCH_LIBS) $(CUDA_LIBS) -lpthread

CPPFLAGS := $(INC_FLAGS) -fPIC -MMD -MP -O2 -Wall
CXXFLAGS := -std=c++17

$(BUILD_DIR)/$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.c.o: %.c
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -r $(BUILD_DIR) || true
