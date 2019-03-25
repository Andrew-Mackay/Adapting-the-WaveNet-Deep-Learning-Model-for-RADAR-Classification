Adapting the WaveNet Deep Learning Model for RADAR Classification
==============================

  An investigation into radar classification. This project adapts the WaveNet deep learning model
  to classify radar data directly from the range profiles representation. This technique is compared with the leading 
  approach in the literature of creating micro-Doppler spectrogram images from the data and classifying these images using a
  CNN.

  This project builds on work conducted by A. Angelov who collected the dataset, developed the original code for processing the data and created the CNN that was applied to the micro-Doppler signatures.

  Level 4 Honours project at the University of Glasgow by Andrew Mackay.
  The project was supervised by Professor Roderick Murray-Smith.

---

Project Setup
------------
Due to the large size of the dataset used for this project (total folder size >300GB) the data has not been included. To download the data, please use the link in the file "gdrive_data_link.env".

All experiments were created in the [Jupyter Notebook](https://jupyter.org/) format  and are stored in the 'notebooks' folder. Many IDEs have built-in support for these files however I would recommend using either [Jupyter Notebook](https://jupyter.org/install) or [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb). As Google Colaboratory works in the cloud you will not have to install any additional software to your device and the environment it provides all but one of the necessary Python packages pre-installed. The one package it does not come with is [Scikit-Optimize](https://scikit-optimize.github.io/), however, this is installed within the notebooks using a cell containing: ```! pip install git+https://github.com/scikit-optimize/scikit-optimize/ ``` (this will need to be uncommented out).

Alternatively, the folder "notesbooks_as_html" contains all the Jupyter notebooks in HTML format. This allows them to be viewed directly by use of a web browser. This format is read-only, the cells cannot be executed.

As many of the experiments conducted take a very long time to execute, the results from previous executions have been saved in the 'results' folder as .pkl files. In the notebooks, these results are then loaded back in to allow graphical visualization of the results. If you want to re-run the experiments, in each notebook there will be one or two boolean variables that must be set to true to allow overwriting of the results. These variables are often called either "OVERWRITE_RESULTS" or "SAVE_RESULTS".

If you intend not to use the Google Colaboratory environment you will need to install all the required python packages. It is recommended to handle this using the [Conda](https://conda.io/en/latest/) environment management software. To install the requirements using Conda, the command ```conda env create -f environment.yml```. The requirements have also been stored in the file "requirements.txt" and can be installed using [pip](https://pypi.org/project/pip/) with the command ```pip install -r requirements.txt```. This investigation has been conducted using Python versions >= 3.6.6 however the code may be compatible with older versions.

---

Notebook Guide
------------
### 01_dataset_composition_analysis.ipynb
This notebook explores the composition of the dataset to get a better understanding of the number of measurements and identify potential class imbalances.

### 02_interim_dataset_creation_convert_i_to_j.ipynb
The raw RADAR data is represented as complex numbers. As it uses the mathematical convention of i to represent the imaginary component this is not compatible with python which uses the engineering convention of j. This notebook replaces the i with j for each data file.

### 03_data_processing_demonstration.ipynb
Demonstration of the stages of processing applied to the radar data.

### 04_interim_dataset_creation_doppler_spectrogram.ipynb
Using the same processing as demonstrated in "03_data_processing_demonstration.ipynb", create the micro-Doppler signature images from the raw data.

### 05_processed_dataset_creation_doppler_spectrogram.ipynb
Creates the final datasets for the CNN model from the micro-Doppler spectrogram images. This notebook uses all the data from all the subjects.

### 05_processed_dataset_creation_doppler_spectrogram_without_corrupt.ipynb
Creates the final datasets for the CNN model from the micro-Doppler spectrogram images. This notebook excludes the corrupt data from the end of subject F's recordings.

### 06_processed_dataset_creation_range_FFT.ipynb
Creates the range profiles dataset. To handle the large size of data, this method saves each array as a separate file and uses a key to store the labels and file name. Intended to be used later with a Data Generator.

### 07_CNN_model_comparison.ipynb
An investigation into the effect of various filter combinations for the CNN model. To compare the different filter values, five-fold cross-validation was used. For each fold, one subject of the five total subjects (subject C being reserved for final evaluation) was withheld for evaluation whilst the model was trained on the remaining four subjects.

### 08_CNN_hyperparameter_search.ipynb
Performs a search over the hyperparameter space for the CNN model to try and find a more optimal parameter configuration.

### 09_CNN_model_early_stopping_tuning.ipynb
An investigation into the effect of early stopping on the CNN model.

### 10_CNN_final_model_evaluation.ipynb
Final evaluation of the CNN model. This notebook uses all data from all subjects for the evaluation.

### 10_CNN_final_model_evaluation_without_corrupt.ipynb
Final evaluation of the CNN model. This notebook excludes the corrupt data from subject F.

### 11_range_data_model_initial_testing.ipynb
Comparing the different WaveNet based model architectures on the range dataset.

### 12_range_data_model_further_experimentation.ipynb
Investigating different representations of the range data. Representations investigated are: the average of all cells, single cells and every second cell.

### 13_range_data_model_causal_vs_non_causal.ipynb
Comparison of the causal and non-causal variants of the range model.

### 14_range_data_model_regularization_investigation.ipynb
Investigating different regularization techniques for the range model to prevent overfitting. L2 regularization and batch normalization were considered.

### 15_range_data_model_hyperparameter_search.ipynb
Performs a search over the hyperparameter space for the range model to try and find a more optimal parameter configuration.

### 16_range_data_model_final_model_comparison.ipynb
Final evaluation of the range model. This notebook uses all data from all subjects for the evaluation.

### 16_range_data_model_final_model_comparison_without_corrupt.ipynb
Final evaluation of the range model. This notebook excludes the corrupt data from subject F.

### 17_final_evaluation_comparison.ipynb
Comparison of the results from the final evaluations of all the models.

---
Project Organization
------------

    ├── LICENSE            <- MIT License
    ├── README.md          <- README 
    ├── data    
    │   ├── interim        <- Intermediate data that has been transformed.
    │   │   ├── *.dat                 <- raw data with 'i' replaced with 'j' for complex representation
    │   │   └── doppler_spectrograms  <- Processed Doppler Spectrogram images
    |   |
    │   ├── processed      <- The final, canonical data sets for modeling.
    |   |   ├── doppler_spectrograms                   <- Doppler spectorgrams dataset with all subjects data
    |   |   ├── doppler_spectrograms_without_corrupt   <- Doppler spectorgrams dataset with corrupt data from subject F removed
    |   |   └── range_FFT                              <- range profile datasets 
    |   |       └── 3  
    |   |           ├── MTI_applied       <- range profile dataset with MTI filter (unused)    
    |   |           └── MTI_not_applied   <- range profiles dataset  
    |   |                     
    │   └── raw            <- The original, immutable data dump.
    |       ├── *.dat                    <- Raw radar data files
    |       ├── Experiment\ notes.xlsx   <- Experimental notes by A. Angelov
    |       ├── Labels.csv               <- Condensed representation of Experiment\ notes.xlsx
    |       └── raw_converted            <- raw data with 'i' replaced with 'j' for complex representation
    │
    ├── environment.yml    <- Anaconda enviroment file
    ├── gdrive_data_link.env   <- Google Drive link to the data
    ├── notebooks          <- Jupyter notebooks.
    │   ├── 01_dataset_composition_analysis.ipynb
    │   ├── 02_interim_dataset_creation_convert_i_to_j.ipynb
    │   ├── 03_data_processing_demonstration.ipynb
    │   ├── 04_interim_dataset_creation_convert_i_to_j.ipynb
    │   ├── 05_processed_dataset_creation_doppler_spectrogram.ipynb
    │   ├── 05_processed_dataset_creation_doppler_spectrogram_without_corrupt.ipynb
    │   ├── 06_processed_dataset_creation_range_FFT.ipynb
    │   ├── 07_CNN_model_comparison.ipynb
    │   ├── 08_CNN_hyperparameter_search.ipynb
    │   ├── 09_CNN_model_early_stopping_tuning.ipynb
    │   ├── 10_CNN_final_model_evaluation.ipynb
    │   ├── 10_CNN_final_model_evaluation_without_corrupt.ipynb
    │   ├── 11_range_data_model_initial_testing.ipynb
    │   ├── 12_range_data_model_further_experimentation.ipynb
    │   ├── 13_range_data_model_causal_vs_non_causal.ipynb
    │   ├── 14_range_data_model_regularization_investigation.ipynb
    │   ├── 15_range_data_model_hyperparameter_search.ipynb
    │   ├── 16_range_data_model_final_model_comparison.ipynb
    │   ├── 16_range_data_model_final_model_comparison_without_corrupt.ipynb
    │   └── 17_final_evaluation_comparison.ipynb
    │
    ├── notebooks_as_html  <- jupyter notebooks from "notebooks" folder converted to html format to allow easy viewing
    ├── results            <- Results from the investigation including graphs  
    │   ├── dataset_composition_analysis
    │   ├── data_processing_demonstration
    │   ├── CNN_model_comparison
    │   ├── CNN_hyperparameter_search
    │   ├── CNN_early_stopping_tuning
    │   ├── CNN_final_model_evaluation
    │   ├── CNN_final_model_evaluation_without_corrupt
    │   ├── range_data_model_initial_testing          
    │   ├── range_data_model_further_experimentation      
    │   ├── range_data_model_causal_vs_non_causal             
    │   ├── range_data_model_regularization                  
    │   ├── range_data_model_hyperparameter_search
    │   ├── range_data_model_final_evaluation                 
    │   ├── range_data_model_final_evaluation_without_corrupt 
    │   └── final_evaluation_comparison                      
    │
    └── requirements.txt   <- The requirements file for reproducing the analysis environment


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
