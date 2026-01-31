# ProDiSE-LSSVM
ProDiSE: A Proximity and Distance Scoring Based Pruning Technique for Large-Scale Supervised Learning

This repository contains the MATLAB implementation corresponding to the paper:

Anuradha Kumari, M. Tanveer
“ProDiSE: A Proximity and Distance Scoring Based Pruning Technique for Large-Scale Supervised Learning”

If you use this code in your research or applications, please cite the above paper appropriately.

For questions, issues, or bug reports, please contact: anuradhaiitg123@gmail.com

Implemented Model

This repository provides the implementation of the following model:

ProDiSE-LSSVM: Sparse Least Squares Support Vector Machine using ProDiSE-based representative sample selection.

Train.txt / Test.txt corresponds to example training and testing datasets.

File Descriptions

ProDiSE_LSSVM_main.m: Main file
ProDiSE_LSSVM_Train.m: Trains the ProDiSE-LSSVM model 
Evaluate.m: Evaluates the trained ProDiSE-LSSVM model on test data and computes classification accuracy.
kernelfun.m: Computes the kernel matrix (e.g., RBF kernel).

