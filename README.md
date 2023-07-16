# Purpose
A project created for the course of Computational Intelligence in Aristotle Univeristy of Thessaloniki 
(Dept. of Electrical and Computer Engineering)

It implements a small regression model based on a Radial Basis Function (RBF) Layer and using the Boston Housing as the dataset.
The model architecture consists of the cascade: Input Layer -> RBF Layer -> Dense Intermediate Layer 
-> Dropout Layer -> Ouput Layer/Node 

The project is Python based, utilizing Tensorflow/Keras; Requirements can be found in req.txt.

# Disclaimer
The RBF Layer is mainly based on an implementation made by Petra Vidnerova which is also supplied in the current repo.
Minor tweaks have been made.

# Details
Project_outline.pdf describes the tasks to be achieved through the implementation of the RBF Regression model, including some brief insight
into RBF Layers/Networks (Greek version provided only).

simple_RBF_application.py addresses the first task, which is basically searching for the optimal RBF layer size among 3 choices, while tracking MSE and $R^{2}$.
betas_initializer.py was added to the repo of Petra Vidnerova to account for the initialization of bias parameters as illustrated in the task outline.

fine_tune_RBF.py addresses the second task;
