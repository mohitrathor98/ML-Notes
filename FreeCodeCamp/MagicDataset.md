
### Dataset Information

##### Source : https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope


- There is a telescope which detects different gamma rays
- There is a camera/sensors which generates different information about the rays detected.
- This helps in predecting type of gamma particle.

##### Google Colab: https://colab.research.google.com/drive/1g7ASF6B16uyT6D4B8t5gopMNGsbIc3Vh#scrollTo=2LWkTx8_eVyn

- Read the dataset using pandas.
- Labels(Column names) were missing so, added column names.

```
    # As can be seen from above, no labels are present, so we need to label the data
    cols = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
    # Assigning the labels to the column of the dataset
    df = pd.read_csv('magic04.data', names=cols)
    # getting first five rows
    df.head()
```

- Replaced data 'g' in class column with 1 and 'h' with 0 for better processing

```
    # g represents gamma particles and h represents hedrons
    # we will replace h with 0 and g with 1
    df["class"] = (df["class"] == "g").astype(int)
    # above code will check if class == g and put 1 if it else put 0
    df.head()
```

- Now we will use classification to predict the class for future data sets.

Supervised Learning
---------------------

- `Classification` - We will predict for any future set of data "class" is 'g'(1) or 'h'(0)
- `Fetures` - All the values (cols) that we will use to predict an output. In this case, from ["fLength" to "fDist"]
- `Labels` - The output that needs to be predicted using the features. In this case "class" col.
