This repository contains the Python functions and scripts for Optimizing, Training and Explaining ANN for selection strategies predictions as presented in [1]. 

From experimental data, relevant state variable (e.g., relative distances, velocity, direction of motion) are computed in [ExtractFeaturesFromExperimentalData_Notebook](ExtractFeaturesFromExperimentalData_Notebook.ipynb) to create the datasets from which input sequences and label for the Artificial Neural Networks (ANNs) will be extracted in `CreateFeatureProcessedDatasets_Notebook.ipynb` for two different expertise; Novice and Expert experimental data. 

The experimental data can be found in [Data/](Data/) folder of this repository. 

The datasets used in [1] are made available in the public reporitory [Datasets](https://osf.io/wgk8e/?view_only=8aec18499ed8457cb296032545963542)

Running `BayesOptForPrediction_tf1.py` will perform Bayesian Optimization for ANNs  hyperparamenters; learning rate, number of hidden layers, number of neurons in each layers and their dropout rate based on previously created processed datasets. 
From the BayesOpt_logs file select the hyperparameters for the best (i.e., lowest 'target' value) ANN to be trained. 

Running `CrossValForPrediction_tf1.py` will k-fold train -- or `TrainForPrediction_tf1.py` train once -- the ANN chosen on a subset of processed datasets for each expertise.

Running `SHAPForPrediction_tf1.py` will apply SHAP DeepExplainer [2] to the trained ANNs to obtain the ranked by importance features discussed in [1].

------------------------------------------------------------------------------------------

The models presented and analysed in [1] are made available at [Checkpoint/FinalModels/](checkpoint/FinalModels/) in this repository; a glossary of models names, ANNs hyperparamenters and training set-up are reported in `FinalModels_details.xlsx`.

Performances of the trained ANNs analysed in [1] have been measured on test samples with `Performance_Notebook.ipynb`.

SHAP results presented in [1], and downloadable from the public repository [SHAPresults](https://osf.io/wgk8e/?view_only=8aec18499ed8457cb296032545963542), can be loaded and printed with `SHAP_DeepExplainer_Notebook.ipynb`.


Additional comments are included throughout to assist with comprehension.



[1] Auletta, F., Kallen, R. W., di Bernardo, M. & Richardson, M. J. (2021). Employing Supervised Machine Leaning and Explainable-AI to Model and Understand Human Decision Making During Skillful Joint-Action.  


[2] Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems, 30, 4765-4774.

------------------------------------------------------------------------------------------
Author: F. Auletta

E-mail : fabrizia.auletta@hdr.mq.edu.au
