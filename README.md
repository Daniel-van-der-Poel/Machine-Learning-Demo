# Machine_Learning_Demo
This demo showcases how machine learning can be used to predict late payments and defaults among customer. It uses the LendingClub dataset and the XGBoost library in an Anaconda Python 3.6 environment.

This demo is a personal CS50 final project.

To run the demo on a Windows 10 computer, follow the instructions below.

## 1.	Download this repo
url: https://github.com/Daniel-van-der-Poel/LendingClub
## 2.	Download one or more datasets
Direct links to the LendingClub datasets:
* https://resources.lendingclub.com/LoanStats3a.csv.zip
* https://resources.lendingclub.com/LoanStats3b.csv.zip
* https://resources.lendingclub.com/LoanStats3c.csv.zip
* https://resources.lendingclub.com/LoanStats3d.csv.zip
* https://resources.lendingclub.com/LoanStats_2016Q1.csv.zip (required)
* https://resources.lendingclub.com/LoanStats_2016Q2.csv.zip
* https://resources.lendingclub.com/LoanStats_2016Q3.csv.zip
* https://resources.lendingclub.com/LoanStats_2016Q4.csv.zip
* https://resources.lendingclub.com/LoanStats_2017Q1.csv.zip
* https://resources.lendingclub.com/LoanStats_2017Q2.csv.zip
* https://resources.lendingclub.com/LoanStats_2017Q3.csv.zip

Unzip the files and copy the .csv files to the .\data directory. 
## 3.	Download the XGBoost repo and the latest nightly build
* Repo: https://github.com/dmlc/xgboost
* Build: http://www.picnet.com.au/blogs/guido/post/2016/09/22/xgboost-windows-x64-binaries-for-download/

Copy the xgboost.dll file into the repo’s .\python-package\xgboost directory.
## 4.	Install the latest Anaconda Python 3.x distribution
Download page: https://www.anaconda.com/download/

Note: it’s advisable to install Anaconda in a directory whose name has no special characters.

## 5.	Open Anaconda Prompt and do the following:
### Update Conda and Anaconda
```
conda update conda
conda update anaconda
```
### Create and activate an environment for machine learning
```
conda create -n ml_1 anaconda
``` 
(this will take a few minutes)
```
activate ml_1
```

Note: if the latest version of XGBoost requires a version of Python that’s older than the once that came with Anaconda, add ‘python=3.6’ (for example) to the conda create statement.
### Add the kernel for notebooks to the environment
```
python -m ipykernel install --user --name ml_1 --display-name "Python machine learning 1"
```
### Install XGBoost
Go to the XGBoost repo’s .\python-package folder and enter:
```
pip install setup.py install
```
### Install graphviz with python bindings (required for tree visualisation)
```
conda install graphviz
conda install python-graphviz
```
## 6.	Run the notebook
* Open xgb_params.txt in the .\settings directory.
* Change the value of n_jobs to the number of threads supported by your CPU.
* Open Jupyter Notebook.
* Open LendingClub.ipynb.
* ‘Change Kernel’ to ‘Python machine learning 1’.
* Click on the first cell (the one with all the imports) and choose ‘Run Cells’.
* If there are no error messages below the cell, choose ‘Run All’ to run the whole script.

Note: for a dark notebook theme, copy custom.css to the %UserProfile%/.jupyter/Custom directory.

