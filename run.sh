#!/bin/bash
model = $1
mkdir cfg
python3 weight_extractor.py $model
tar xfv imagenette_image.tar
cd mlpack-loader && git clone --single-branch --branch BranchForConverter https://github.com/kartikdutt18/models.git
cd models && mkdir build
case "$model" in
  "darknet19") cd build && cmake ../ && sudo make -j2 && ./bin/test_network ;;
  "yolov1_tiny") cd build && cmake ../ && sudo make -j2 && ./bin/yolo_test ;;
