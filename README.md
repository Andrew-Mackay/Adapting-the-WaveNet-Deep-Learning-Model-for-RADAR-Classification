Adapting the WaveNet Deep Learning Model for RADAR Classification
==============================

  Deep learning applied to radar classification, supervised by Prof Roderick Murray-Smith.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   │   └── doppler_spectrograms
    │   ├── processed      <- The final, canonical data sets for modeling.
    |   |   ├── doppler_spectrograms
    |   |   ├── doppler_spectrograms_without_corrupt
    |   |   └── range_FFT
    │   └── raw            <- The original, immutable data dump.
    |       └── raw_converted
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    ├── environment.yml    <- 
    ├── gdrive_data_link.env   <- 
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
    ├── results   <-  
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
