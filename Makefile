all: model compile

model:
	@make models/cifar10_cnn.cpp

models/cifar10_cnn.cpp: cifar10_cnn.pb
	utensor-cli convert cifar10_cnn.pb --output-nodes=fully_connect_2/logits --transform-methods=-inline

compile:
	mbed compile -m auto -t GCC_ARM --app-config=mbed_app.json --profile=uTensor/build_profile/debug.json -f

compile-pc: model


.PHONY: clean
clean:
	rm -rfv models constants BUILD