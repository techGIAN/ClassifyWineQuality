# ClassifyWineQuality
A simple Machine Learning project using SciKit Learn through the classification of wine quality

To begin, import the required packages, namely (1) Pandas, (2) Seaborn, (3) MatPlotLib, (4) Sklearn.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
%matplotlib inline
```
`Pandas` is a very helpful library that aids in the manipulation and analysis of data. We will be working with a lot of data, hence Pandas is very important.

`Seaborn` is used for data visualization. Although not required for the purposes of our exemplar, it is still good to have it to gain some insights based on what we can see on the data through the visuals.

`MatPlotLib` is a 2D plotting library used in Python. `Seaborn`'s data visualization is based off of `MatPlotLib` - so we will import this as well. But again, is not necessary if Seaborn isn't used.

`Sklearn` (or SciKitLearn) is the machine learning library in Python, and it contains several various ML algorithms that we can use. For the purposes of this example, we will take a look into Random Forest Classifier, Support Vector, and Multi-Layer Perceptron. There are still a lot more of these algorithms; but if you are interested check out the link: https://scikit-learn.org/stable/

Let us begin by examining our data. For this, we will be using the `winequality-red.csv` dataset; you can download this dataset from Kaggle here: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

Now open your PyCharm, NotePad++, Sublime Text, or any other text-editor. Note that we will be working with Python for this example, so it is assumed that you have the basics of Python. But do not worry, if you are struggling with Python, you can just Google it up (especially the syntax), or read through here: https://www.w3schools.com/python/. Anyway, I am using the Jupyter Notebook in the Anaconda Navigator; it is very easy and convenient to use. I can easily run it and see the output right away because it compiles automatically. Below displays the two lines for loading our wine dataset into Python, store it into a dataframe called `wine` as well as displaying the first five rows of the dataframe.

```python
wine = pd.read_csv('winequality-red.csv', sep = ';')
wine.head()
```
This is the work of Pandas; it reads through our CSV file, with semi-colons as separators. It then stores it into a dataframe called `wine`. Here is what the first five rows of the dataframe looks like

