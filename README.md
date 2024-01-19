# Thyroid-Disease-Detection

# Problem Statement:
Thyroid disease is a common cause of medical diagnosis and prediction, with an onset 
that is difficult to forecast in medical research. The thyroid gland is one of our body's 
most vital organs. Thyroid hormone releases are responsible for metabolic regulation. 
Hyperthyroidism and hypothyroidism are one of the two common diseases of the thyroid 
that releases thyroid hormones in regulating the rate of body's metabolism.  
The main goal is to predict the estimated risk on a patient's chance of obtaining thyroid 
disease or not.

Deployed app link
http://localhost:8501/
# Data used

Get the data from https://archive-beta.ics.uci.edu/dataset/102/thyroid+disease  
Quinlan,Ross. (1987). Thyroid Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C5D010.

# Project Flow
![image](https://github.com/Jesi1511/Thyroid-Disease-Detection/assets/144013413/c2a83305-e3e4-4e80-89b7-cb6a23a4ccd3)


# Programming Languages Used
<img src = "https://img.shields.io/badge/-Python-3776AB?style=flat&logo=Python&logoColor=white">


# Python Libraries and tools Used
<img src="http://img.shields.io/badge/-Git-F05032?style=flat&logo=git&logoColor=FFFFFF"> <img src = "https://img.shields.io/badge/-NumPy-013243?style=flat&logo=NumPy&logoColor=white"> <img src = "https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white"> <img src="http://img.shields.io/badge/-sklearn-F7931E?style=flat&logo=scikit-learn&logoColor=FFFFFF">  <img src = "https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white"> <img src = "https://img.shields.io/badge/-mlflow-0194E2?style=flat&logo=mlflow&logoColor=white"> <img src = "https://img.shields.io/badge/-Pydantic-000000?style=flat&logoColor=white">

## Run Locally

Go to the project directory (let's say Thyroid-disease-detection)

```bash
    cd Thyroid-disease-detection
```

Create a conda environment

```bash
    conda create -n environment_name python=3.10
```

Activate the created conda environment

```bash
    conda activate environment_name
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Load the data

```bash
  python src/step1_make_dataset.py
```
Validate the loaded data

```bash
  python src/step2_training_Data_Validation.py
```
Preprocess the validated data

```bash
  python src/step3_data_preprocessing_stage1.py
```
```bash
  python src/step4_data_preprocessing_stage2.py
```
Training a machine learning model using preprocessed data and also evluating metrics

```bash
  python src/step5_model_training.py
```
Visualize the metrics in different experiments

```bash
  mlflow ui
```
Testing the code for
 - Data Loading
 - Data Validation
 - Data Preprocessing
 - Model Training and Evaluation

```bash
  pytest src/
```

Make predictions using trained model

```bash
  streamlit run src/Dashboard.py
```

🔗 Links
linkedin
![image](https://www.linkedin.com/in/jesima-parvin/)


