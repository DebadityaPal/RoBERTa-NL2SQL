# Data Agnostic RoBERTa-based Natural Language to SQL Query Generation
This repo contains the code for NL2SQL paper titled "Data Agnostic RoBERTa-based Natural Language to SQL Query Generation". The link to which is: https://arxiv.org/abs/2010.05243

Our code was based on the previous works introduced in the paper "Content Enhanced BERT-based Text-to-SQL Generation", the code to which can be found [here](https://github.com/guotong1988/NL2SQL-RULE). Feel free to experiment with the models.

## Getting Started
Our work was mainly done in a Google Colab workspace with a GPU runtime accelerator, in order to reproduce the results we would recommend running the code in a similar workspace. Thus we have created a notebook which can be found at:
https://colab.research.google.com/drive/1qYJTbbEXYFVdY6xae9Zmt96hkeW8ZFrn
Below are the step by step instructions to run the models:

 1. Visit https://drive.google.com/drive/folders/13f2MrdpieC9QGXM_DJnj2f1Hs6ZBh2ZT?usp=sharing. This Drive contains all the datasets and pretrained model weights. **Add a shortcut to your drive.** This step is important.
 2. Open the notebook, the link to which is provided above.
 3. Make sure the runtime accelerator is set to GPU.
 4. Mount your Google Drive and clone this repository in the runtime environment, the code for this step has already been provided in the notebook.
 5. The Notebook is now ready for use.
 
 All the package requirements have been mentioned in the notebook.

## Results

| Dev Logical Form Acc | Dev Execution Accuracy | Test Logical Form Accuracy | Test Execution Accuracy |
|:-:|:-:|:-:|:-:|
| 69.4% | 77.0% |68.9%  | 76.7% |

## References

 - https://github.com/salesforce/WikiSQL
 - https://github.com/guotong1988/NL2SQL-RULE
