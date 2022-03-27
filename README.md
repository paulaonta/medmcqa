<h1 align="center">MedMCQA </h1>

<h3 align="center">MedMCQA : A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering</h3>

A large-scale, Multiple-Choice Question Answering (MCQA) dataset designed to address realworld medical entrance exam questions. 

The MedMCQA task can be formulated as X = {Q, O} where Q represents the questions in the text, O represents the candidate options, multiple candidate answers are given for each question O = {O1, O2, ..., On}. The goal is to select the single or multiple answers from the option set.


[![GitHub license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub commit](https://img.shields.io/github/last-commit/medmcqa/medmcqa)](https://github.com/medmcqa/medmcqa/commits/main)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)


## Requirements

Python 3, pytorch
To install pytorch , follow the instructions at https://pytorch.org/get-started/


## Data Download and Preprocessing

download the data from below link

data : https://drive.google.com/file/d/1-5Zl3QKNC5OuZpF8RzciJJKFq3Ug2wnT/view?usp=sharing

## Data Format

The structure of each cav file :

- `id`           : Unique id for each sample
- `question`     : Medical question (a string)
- `opa`          : Option A 
- `opb`          : Option B
- `opc`          : Option C
- `opd`          : Option D
- `cop`          : Correct option (Answer option of the question)
- `choice_type`  : Question is single choice or multi-choice
- `exp`          : Explanation of the answer
- `subject_name` : Medical Subject name of the particular question
- `topic_name`   : Medical Subject's topic name of the particular question


## Model Submission and Test Set Evaluation

For new submissions, Use any one of the below methods. 

- Create a working colab notebook with your model (only model evaluation step), where colab notebook will take input as dev-set, return predictions and score [example], send the notebook URL and description of paper to below email addresses. 
- Please fork and edit https://github.com/medmcqa/medmcqa.github.io with your results & send the code, model to 
the below emails with full description to run the code. Your code will run on the test set. In this case, you must submit your code and provide a Docker environment.


aadityaura [at] gmail.com,
logesh.umapathi [at] saama.com
