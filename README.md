# Semestral Project: Predictive Coding Network Experiments
These are all the files I needed to code for my semester project with Kristína Malinovská and Tamara Bíla for my exchange at the Comenius University in Bratislava for my MEi:CogSci master's with the TERAIS group at the faculty of mathematics, physics, and informatics.

## Introduction
Predictive processing is a novel approach to understanding cognition, viewing the brain as a "prediction machine" that operates through active inference. This approach suggests that in any situation, the brain actively predicts the most probable outcomes, using both action-based responses and detailed knowledge that continuously adapts to changing tasks and contexts (Clark, 2015). Central to this approach is Bayesian learning, where the brain updates its expectations (or "posterior probability") by factoring in the likelihood of events based on prior beliefs. Predictive processing gained traction with discoveries in neuroscience that similar mechanisms underpin the brain’s functioning, particularly in the visual cortex. Rao and Ballard (1999) found that some visual effects—previously assumed to result solely from direct sensory input—are also shaped by feedback from higher brain areas. This feedback allows the brain to make efficient, layered guesses about incoming visual information, enhancing the accuracy of processing natural images. This finding allowed cognitive scientists to adjust their beliefs about how cognition in the brain works providing a new framework for the analysis and modeling of cognitive phenomena.

This approach to cognitive science has also informed the creation process of artificial systems such as neural networks. Similar to the neural networks of the brain, these artificial neural networks (ANNs) are developed on the principles of the Bayesian brain integrating previous knowledge and likelihood into the outcome calculations. This approach has been studied in the past with intriguing results and new emergent abilities (Han et al., 2018; Wen et al., 2018). These Predictive Coding Networks (PCNs) while reaching promising performance metrics, also provide a greater biological basis as their underlying mechanisms are based on the proposed neural network mechanisms of the brain. This new form of artificial neural network could also be of use in the field of robotics as neural networks have been showing great promise in the creation of more autonomous and human-like robots. 

In this study, we investigated the performance of predictive coding networks through a series of experiments designed to assess their efficiency, accuracy, and applicability to real-world tasks. We explored how varying the number of recurrent processing steps (circles) within the network affects model performance, determining an optimal threshold before diminishing returns set in. Additionally, we tested the PCN architecture on a human action recognition dataset to evaluate its ability to process complex visual information related to human movement, which has direct implications for robotics. Finally, we examined how reducing the network depth impacts stability and accuracy, shedding light on the trade-offs between model complexity and computational efficiency. These experiments provide insights into how PCNs function under different conditions and how they may be further optimized for future applications in artificial intelligence and robotics.

## Experiments
The experiments conducted were threefold: testing how the amount of locar recursion affects model performance, setsing the model on a humanoid vision dataset, and seeing the impacts of model size on performance. Below are the results highlighted.

### Circles
The first set of experiments conducted was testing the impact on the number of circles in the local recurrent process of the PCN. Han and colleagues proposed that the recursion in the local layer serves to decrease error in prediction by reducing the sum of the squared prediction error by generating predictions of the lower layer representations through feedback loops (2018). Through this experiment our objective was to find the scaling of this idea and whether at some point the performance tapers off or decreases due to the recursive layers. 
![circles graph](https://github.com/DominikZeman/SemesterProject_PCN/blob/main/pcn_data/circles_experiment.png?raw=true)

### Har
The next set of experiments was utilizing the PCN architecture on a vision dataset that contained humanoid bodies. The human action recognition dataset contains images of people doing a set of 15 different human activities from royalty-free websites (Nagadia, 2022). These activities included things such as using a laptop, sleeping, laughing, clapping, eating, and others. The dataset contained 12,600 images which were split into an 80/10/10 train/test/validation split. The prospect of the PCN architecture in a continuation of this study would be to apply it to a robotics use where both vision and proprioception data are used in tandem. This could later be used in robotics in the real world by controlling a robot with PCN outputs.

### Hidden Layers
The last experiment conducted was done to see whether there is a rapid decrease in performance if the size of the network is decreased. We wanted to see if a smaller network of 8-layers could potentially reach a similar performance as the 12-layer network. 

## Files
In this section I describe all of the aditional code that I have wrote for this project. Closer explanation for files contained in this Git, which are necessary for running the code but not written by me, can be found here: https://github.com/libilab/PCN-with-Local-Recurrent-Processing?tab=readme-ov-file

### create_dataset
#### create_dataset.py
This file can be used to create a dataset of joint angles from the iCub robot correcsponding to images of the iCub robot itself. It uses the learningimitation framework on the iCub simulator by Andrej Lucny (https://github.com/andylucny/learningImitation) and the dataset of joint angles (icubposes).

### pcn_data
#### convert_txt_to_csv.ipynb
This file converts the text files from training of the PCN model to a csv file containing the loss and accuracy at each epoch, allowing for the graphing of the model performance.
#### graphing.ipynb
This jupyter notebook compiles and graphs all the data gathered during the experiments outlined above.

### PCN_training
#### main_cifar_graph.py
This code trains the model on the CIFAR-100 dataset and collects the data neatly in a csv file for future graphing.

#### main_gbrma.py
This segment of code was not utilised. It tries to train the model on sequential data from a dataset of proprioceptive hand gestures. The sequential nature of the data makes it difficult to adjust the model architecture to be able to run the training. This experiment was therefore not pursued.

#### main_har.py
This python file trains the PCN on the human action recognition dataset and gathers the training data in a csv file.

#### split_dataset.py
This file was used for splitting the HAR dataset into a train/test/validation split as it was part of a kaggle competition and the test data was not labled.
