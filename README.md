# Aerial Airport Object Detection

This project aims to detect parked airplanes in aerial airport images using an object detection model. The model is trained on the GDIT Aerial Airport dataset. The objective is to accurately identify and localize the airplanes within the images.

## Dataset

The dataset consists of aerial images captured at airports, containing various instances of parked airplanes. All types of airplanes in the dataset have been grouped into a single classification named "airplane".

Here are some sample images from the dataset:

<br/>
<div align="center">
  <img src="https://github.com/IslamMounir/Aerial_Airport_Object_Detection/blob/main/Images/dataset_1.jpg" alt="Dataset Sample 1" width="190"/>
  <img src="https://github.com/IslamMounir/Aerial_Airport_Object_Detection/blob/main/Images/dataset_2.jpg" alt="Dataset Sample 2" width="190"/>
  <img src="https://github.com/IslamMounir/Aerial_Airport_Object_Detection/blob/main/Images/dataset_3.jpg" alt="Dataset Sample 3" width="190"/>
  <img src="https://github.com/IslamMounir/Aerial_Airport_Object_Detection/blob/main/Images/dataset_4.jpg" alt="Dataset Sample 4" width="190"/>
</div>

## Model Training

The object detection model is trained using the YOLOv8 model architecture. YOLOv8 is a state-of-the-art deep learning model known for its real-time object detection capabilities.
The model is trained to accurately identify and locate parked airplanes within the aerial airport images. 
The training is performed for multiple epochs using the "train" and "valid" datasets  with the following configurations:
- Model: YOLOv8s
- Optimizer: Adam
- Image Size: 600
- Learning Rate: 0.005

The training results are as follows:
- Precision: 0.936
- Recall: 0.911
- mAP50: 0.943
- mAP50-95: 0.539



## Model Evaluation

The trained model was evaluated on the test dataset with the following results:
- Precision: 0.942
- Recall: 0.876
- mAP50: 0.946
- mAP50-95: 0.532


## Inference

The model was used to perform inference on a set of test images. Here are some sample images along with the predicted bounding boxes:

  <br/>
<div align="center">
  <img src="https://github.com/IslamMounir/Aerial_Airport_Object_Detection/blob/main/Images/test_1.jpg" alt="Inference Sample 1" width="190"/>
  <img src="https://github.com/IslamMounir/Aerial_Airport_Object_Detection/blob/main/Images/test_2.jpg" alt="Inference Sample 2" width="190"/>
  <img src="https://github.com/IslamMounir/Aerial_Airport_Object_Detection/blob/main/Images/test_3.jpg" alt="Inference Sample 3" width="190"/>
  <img src="https://github.com/IslamMounir/Aerial_Airport_Object_Detection/blob/main/Images/test_4.jpg" alt="Inference Sample 4" width="190"/>
</div>

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine.

2. Open the `train.ipynb` notebook.

3. Run the notebook to perform training, testing and inference. The notebook contains all the necessary code and instructions for each step.

4. Make sure to modify any relevant file paths or configurations within the notebook to suit your specific setup and requirements.

Feel free to explore and customize the code within the notebook to further adapt it for your needs.


## Acknowledgments

- The dataset used in this project was collected from a publicly available source. [Click here](https://universe.roboflow.com/gdit/aerial-airport/dataset/1) to access the dataset.

