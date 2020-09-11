# mlpack-PyTorch-Weight-Translator
Simple Repository to transfer weights from PyTorch to mlpack.


#### Aim:
1. Generate CSV files for weights, biases and all trainable parameters from PyTorch models.
2. Generate XML file for PyTorch model that holds the structure as well as files necessary to reproduce the model.
3. Create a parser in C++ which loads all weights and biases from XML file to the mlpack model.
4. Test it on DarkNet 19 and DarkNet 53 model.
5. Verify output for each layer and add tests for layers that were missing tests. [Add tests for Conv. Layer](https://github.com/mlpack/mlpack/pull/2548).
6. Create sample notebooks to see if models match in mlpack and PyTorch.
7. Create bash file to test the models.
8. Match the accuracies of the models and save the models weights.

#### Status :
Complete. Since, this was a different repository, I have created a bash file for testing.


### Requirements :
 ```
  python >= 3.x
  mlpack
  Armadillo      >= 8.400.0
  Boost (program_options, math_c99, unit_test_framework, serialization,
         spirit) >= 1.58
  CMake          >= 3.3.2
  ensmallen      >= 2.10.0
 ```
 
 ### Usage :
 For detailed use, create a python model add it to models folder and pass the model to weight_converter.py. Repeat for C++ side with weight_converter.cpp
 
 For testing, use the following commands :
 ```
 ### Tests darknet model converted from PyTorch.
 ./run.sh
 ```
 
 Expected Output : `Accuracy : 0.7236842105`
 
 ```
 ### Tests YOLO model.
 ./run.sh yolov1_tiny
 ```
 
 Expected Output : `IoU between prediction from PyTorch and mlpack is : 1.0`
 
