CXX := g++
INCLUDE := -IuTensor -I$(PWD)
FLAGS := -std=c++11 -g
UTENSOR_SRC := uTensor/uTensor
SRC_FILE := $(UTENSOR_SRC)/*/*.cpp
HEADER_FILE := $(UTENSOR_SRC)/*/*.hpp

all: model compile

model:
	@make models/cifar10_cnn.cpp

model-pc:
	utensor-cli convert cifar10_cnn.pb \
	--output-nodes=fully_connect_2/logits --transform-methods=-inline \
	-D constants/cifar10_cnn

compile-pc: model-pc pc_main

compile-mbed: model
	mbed compile -m auto -t GCC_ARM --app-config=mbed_app.json -D $$(EMBED_DATA_DIR) --profile=uTensor/build_profile/debug.json -f

models/cifar10_cnn.cpp: cifar10_cnn.pb
	utensor-cli convert cifar10_cnn.pb --output-nodes=fully_connect_2/logits --transform-methods=-inline

pc_main: $(HEADER_FILE) $(SRC_FILE) pc_main.cpp
	$(CXX) $(SRC_FILE) models/*.cpp pc_main.cpp -o pc_main $(INCLUDE) $(FLAGS)

debug: model-pc pc_main
	lldb -f ./pc_main



.PHONY: clean
clean:
	rm -rfv models constants BUILD pc_main *.dSYM