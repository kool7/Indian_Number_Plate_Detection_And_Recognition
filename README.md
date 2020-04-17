# Indian Number Plate Detection And Recognition

Number Plate detection and recognition using tensorflow object detection API and OpenCV. 

## Requirements 
  * OpenCV
  * python 3.6
  * pytessearct
  * tensorflow 1.13.1
  * flask
  * flask_cors

## Steps

 * Create separate anaconda environment and name it numberplate with python version 3.6. 
   **example - conda create -n numberplate python=3.6**
   
 * Clone this repo **[Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/r1.13.0)**.
 
 * After cloning the **Tensorflow Object Detection API** get inside research/object detection folder.
 
 * Activate your new anaconda environment. 
   **example - conda activate numberplate** and start installing dependencies one by one.
   
 * pip install Cython contextlib2 pillow lxml jupyter matplotlib pandas
 
 * conda install -c anaconda protobuf
 
 * pip install opencv-python
 
 * Get inside research folder using command and run ./bin/protoc object_detection/protos/*.proto --python_out=.
 
 * Inside the research folder run these commands **python setup.py build** after that run **python setup.py install**
 
 * To add the paths to environment variables in Linux you need to type:
   FOR LINUX
    * export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/TensorFlow/models/research
    * export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/TensorFlow/models/research/object_detection
    * export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/TensorFlow/models/research/slim
    
   FOR WINDOWS
   * get inside environment variables and set the path to above folder.
   
 * Run **python run.py** for running web API or run **python detect_recog_plate.py**
   
   
