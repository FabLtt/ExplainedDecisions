This repository contains the Python functions and scripts for Optimizing, Training and Explaining ANN for selection strategies predictions

`BayesOpt_ForPrediction_tf1.py` perform Bayesian Optimization of learning rate, number of hidden layers, number of neurons in each layers and their dropout rate

`CrossVal_ForPrediction_tf1.py` perform Cross Validation of the ANN chosen 

`Train_ForPrediction_tf1.py` train the ANN on N_train samples and test in on N_test samples

`SHAP_ForPrediction_tf1.py` applies SHAP DeepExplainer [1] to the trained ANN to explain the N_test samples 



[1] Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems, 30, 4765-4774.

------------------------------------------------------------------------------------------
Author: F. Auletta

E-mail : fabrizia.auletta@hdr.mq.edu.au
