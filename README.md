# Digital Image Procesisng 
EDS6397 - Digital Image Processing - Group 9 - Weather Prediction Using images dataset

## Run the model:
1. Clone this repository to your local machine.
2. Install the required dependencies by running: pip install -r requirements.txt
3. Download the dataset from the folder 

## Introduction
This project leverages image datasets and advanced machine learning techniques to predict weather conditions. By analyzing visual cues from images, such as cloud patterns, lighting, and other features, the system predicts weather categories with high accuracy. 

### Dataset
This project combines images from two publicly available datasets to create a unified dataset for weather classification. We carefully selected relevant weather classes from both datasets and merged them to enhance the diversity and robustness of the training data.
The datasets used are: 
1. [Multi-class Weather Dataset((https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset))  
   This dataset contains images of various weather conditions, including shine & sunrise,rain.

2. [Weather Image Recognition((https://www.kaggle.com/datasets/jehanbhathena/weather-dataset))  
   This dataset provides additional weather categories with high-quality images with other classes as fogsmog,rain,snow,lighting,sandstorm.

### Data Workflow
For the implementation of our weather prediction project, we followed a structured workflow to handle and prepare the dataset:(both the datasets zip files are attached)

#### Initial Dataset:
We began with an initial dataset named DIP Final, which was created by combining and preprocessing images from two publicly available datasets mentioned above. Relevant weather classes, such as sunny, cloudy, rainy, and snowy, were carefully selected and merged into a unified structure. 

#### Data Splitting:
After preprocessing, the data was split into training, validation, and testing subsets, resulting in a structured dataset saved as Final_data. During pre processing steps, for every run new folders are created, before beginning the model building you can use the final dataset named as "Final_data"

#### Model Implementation:
The processed data from Final_data was then utilized for model training, validation, and evaluation. This ensured a streamlined and effective workflow for building and testing our machine learning models. The complete code is divided into 2 files, DIP_GROUP5_PREPROCESSING.ipynb has complete preproceesing steps performed on teh dataset and it gives a new folder of the cleaned dataset. (Final project file) has complete working code of different models implemented and their results. Comparison and evaluation of the was also done here by testing them with sample image as input.

### Technologies Used:
1. Python: Core programming language
2. TensorFlow/Keras: Deep learning framework
3. OpenCV: Image preprocessing
4. Matplotlib: Data visualization

## Flask application
The backend of the application is powered by Flask, which serves the following purposes:
1. Loading the pre-trained weather prediction model.
2. Accepting image input from the user.
3. Running the model to predict the weather condition.
4. Returning the prediction result as a response.

The frontend was developed using basic HTML, CSS, and JavaScript integrated with Flask. It provides:
1. A simple file upload form for users to upload weather images.
2. A "Predict" button to send the uploaded image to the backend for analysis.
3. Display of the predicted weather condition on the same page.

## Authors
Sanjana Rao Ponaganti, Poojitha Reddy Bommu, Harshini Nimmala,Likhitha Reddy Kesara , Aditya Nidadavolu , Suguna Chandana Sibbena

