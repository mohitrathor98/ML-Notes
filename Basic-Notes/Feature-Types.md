## Features

### Qualitative - Categorical data (finite number of categories or groups)

#### 1)  Nominal Data (no inherent order)

    No order between different kind of datasets. For ex: data of counteries or genders.

##### One-Hot Encoding

    We represent nominal data using One-Hot Encoding.

    Suppose we want to represent [USA, India, Canada, France] with numbers.
    Using One-Hot Encoding, we can do it as per below:

    If it matches some category make it "1" else, "0"
    So, USA : [1, 0, 0, 0], India: [0, 1, 0, 0], Canada: [0, 0, 1, 0] and France: [0, 0, 0, 1]


#### 2)  Ordinal Data (Inherent order)

    Certain order can be seen between different category of data.
    For ex: Data of different age groups, or different moods, etc.

    Ordinal data can be represented by the order. For ex:
    in the dataset of moods, 1 - Extremely sad, 2 - sad, 3 - ok, 4 - Happy, 5- Delighted.


### Quantative - Numberical valued data (could be discrete or continuous)
