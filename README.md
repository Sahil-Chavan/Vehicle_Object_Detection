# Vehicle_Object_Detection

In the search of my first Computer Vision, Object Detection project, I landed on Vehicle Detection system.

I have built this system using Mobilenet V2 architecture, which based on Single-Shot multibox Detection (SSD) network. 

Dataset : I got my dataset through http://cbcl.mit.edu/software-datasets/streetscenes

Model Training : As stated I am using an pre-trained SSD Mobilenet model from Tensorflow Model Zoo, and applied Transfer Learning on the model for 2000 epochs. For this training I used TFOD and COCO object detection api.

End point : One can access the system either by the webpage that I have created using Flask, or by using Postman, by accessing the '/api' endpoint, which receives input image through POST File method and provides output image with predictions in base64 format.

End Result : An system which is capable of recognizing vehicles from an image or from video with great speed due to simplistic architecture of SSD mobilenet and sufficient accuracy.

Future Plans : Next I am going to train in YOLO V4/V5 model for this purpose and compare the results.

Libraries Used : OpenCV, TFOD, COCOapi, Tensorflow 2.X, Pillow, Flask, Imutils, etc.

Shout Out to all these references :

- https://sourangshupal.github.io/Tensorflow2-Object-Detection-Tutorial/
- https://www.mygreatlearning.com/blog/object-detection-using-tensorflow/
- https://docs.opencv.org/4.5.2/dd/d43/tutorial_py_video_display.html
- https://tutorial101.blogspot.com/2021/04/python-flask-upload-and-display-image.html
- https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_saved_model.html#sphx-glr-auto-examples-plot-object-detection-saved-model-py
