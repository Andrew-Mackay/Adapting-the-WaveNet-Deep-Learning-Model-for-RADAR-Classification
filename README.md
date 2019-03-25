Adapting the WaveNet Deep Learning Model for RADAR Classification
==============================

  Inestigation into radar classification. This projects adapts the WaveNet deep learning model
  to classify radar data by the range profiles representation. This is compared with the leading 
  approach of creating micro-Doppler spectrogram images and classifying these images with a
  CNN.

  Level 4 Honours project at the University of Glasgow by Andrew Mackay.
  Project supervised by Professor Roderick Murray-Smith.
  

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
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
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
