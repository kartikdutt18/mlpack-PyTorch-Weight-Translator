#!/bin/bash
python3 weight_extractor.py darknet19
cd mlpack-loader && git clone https://github.com/kartikdutt18/models.git
cd models && mkdir build 
cd build && cmake ../ && sudo make -j2 && ./bin/imagenette_trainer