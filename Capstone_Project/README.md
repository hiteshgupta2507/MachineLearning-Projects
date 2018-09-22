# Machine Learning Engineer Nanodegree
# Supervised Learning
## Project: Detect YouTube comment as Spam / not Spam

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)


### Code

Template code is provided in the `YouTube_Comment_Spam.ipynb` notebook file. You will also be required to use the included `YouTube-Spam-Collection-v1` folder which has all the dataset file to complete your work. 

### Run

In a terminal or command window, navigate to the project directory that contains `YouTube_Comment_Spam.ipynb` (that also contains this README) and run one of the following commands:

```bash
ipython notebook YouTube_Comment_Spam.ipynb
```  
or
```bash
jupyter notebook YouTube_Comment_Spam.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Data

The dataset consists of approximately 2000 data points, with each datapoint having 5 features. The dataset is obtained from https://archive.ica.uci.edu/ml/datasets/YouTube+Spam+Collection

**Features**
- `COMMENT_ID`: ID of the person who wrote the comment.
- `AUTHOR`: Person's name who wrote the comment.
- `DATE`: Date when the comment was written.
- `CONTENT`: Actual comment (message) written by an author.

**Target Variable**
- `CLASS`: 1 for Spam and 0 for not Spam
