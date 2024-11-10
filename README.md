# RC
Rain Classification use machine learning method.
## Usage
### Make traindata
Please view prepare data folder to make data set for trainning and run in order:
```
cd prepare_data
python prepare_data.py
python join_csv.py
python make_train_data.py
```
After that you will have dataset of each day in train_data
### Train
Take a look at model building:
- you can run xgb_randomsearch.py for trainning model using xgboost algorithms
- Or you can train random forest model using colab
### Map generater
To generate map, in map-gen pls run:
```
python make_img.py
```
  
