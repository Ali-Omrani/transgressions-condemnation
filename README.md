# transgressions-condemnation
Code for twitter condemnation of moral transgressions. Collaborations with Nour Ktiely, Joe Hoover, Jordan Jillian, and Morteza Dehghani

### [Status Slides](https://docs.google.com/presentation/d/1DdwC-IWvNlPkv0pmMKqHO7Sd61Uy2LUcZl2e7c1OihE/edit?usp=sharing)

# Setup

Install python + R and libraries.

```
pip install -r requirements.txt
```

# Data

We crawled tweets mentioning 265 targets the vox article. The data is available [here](https://drive.google.com/drive/folders/1YgG65l-E99gNKhGXS00enTO3bq5r1azs?usp=share_link).

After downloading the `twitter_data.tar.zst` decompress it with 

```
 tar -I zstd -xf twitter_data.tar.zst
```

this would result in 265 `<target>.db` files, one for each target. 

## Preprocessing

### Creating the MongoDB instance

Our code operates assuming there is a local instance of MongoDB. First step is to load the twitter data for all targets into the MongoDB. You can run `sql-to-mongo.ipynb` to do so. This should create a db named `twitter` in which there is `tweets` collection with *22.6* milion records.

### Cleaning the Tweets

Our analysis involves building models for two tasks. Given a tweet, in the first task $T_1$ the model would predict if it contains condemnation (1) or not (0). Given the a tweet that was predicted as containing condemnation in $T_1$, in $T_2$, the model's goal is to categorize the severity of condemntaion into three categories. To avoid model's reliance on targets, we mask all target mentions in the tweets in `preprocess-jason-tweets.ipynb` using the `mask_all_db` function.


# Training the Models

## $T1$: Detecting condemnation of a tweet
All code for this model is currently under the `condemnation` directory.
### **Training data**
The raw data is available in `condemnation/my_dataset_loading_script/nour_jillian_condemnation_r2_to_r7_maj_vote.jsonl`. The script `my_dataset_loading_script/my_dataset_loading_script.py` creates a transformers dataset based from the data and splits into train and test.

### **Training code**
Training code with 10 fold CV is available `transgressions-condemnation/condemnation/condemnation_train.ipynb`


### **Performances**

|   <i></i>     | $F_1$         | Precision     | Recall        | Accuracy      |
|-------------  | ------------- | ------------- |-------------  |-------------  |
|   Val         | 0.82 (0.02)   | 0.78 (0.03)   | 0.86 (0.03)   |0.76 (0.03)    |
|  Test         | 0.84 (0.02)   | 0.82 (0.03)   | 0.86 (0.03)   |0.80 (0.02)    |

### **Prediction Code**

To add model predictions to all tweets in the database, run the code in `condemnation_inference.ipynb`.


-----------------------------

### $T2$: Categorizing condemnation severity of a tweet
- training data
- trainig code
- performances
- prediction code

------------------------------------

### $T3$: Estimating the ideology of a tweet's author
- data
- estimation code
- validation checks


## Regression and Analysis




