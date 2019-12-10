# Copyright 2020 Toyota Research Institute.  All rights reserved.

DEPTH_TYPE ?= None
CROP ?= None
SAVE_OUTPUT ?= None

PYTHON ?= python
DOCKER_IMAGE ?= packnet-sfm:master-latest
DOCKER_OPTS := --name packnet-sfm --rm -it \
            -e DISPLAY=${DISPLAY} \
            -e XAUTHORITY \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
			-v ~/.cache:/root/.cache \
			-v /data:/data \
			-v ${PWD}:/workspace/self-supervised-learning \
			-v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0 \
			-v /dev/null:/dev/raw1394 \
			-w /workspace/self-supervised-learning \
			--shm-size=444G \
			--privileged \
			--network=host

.PHONY: all clean docker-build

all: clean

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf


docker-build:
	docker build \
		-t ${DOCKER_IMAGE} . -f docker/Dockerfile

docker-start-interactive: docker-build
	nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
		bash

docker-evaluate-depth: docker-build
	nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
            bash -c "bash scripts/evaluate_depth.sh ${MODEL} ${INPUT_PATH} ${DEPTH_TYPE} ${CROP} ${SAVE_OUTPUT}"

