import pandas as pd
import numpy as np
from train import trainer
from predict import predictor
import yaml
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

train_df = pd.read_csv(dir_path +'/data/train_df_test.csv')
test_df = pd.read_csv(dir_path +'/data/test_df_test.csv')
submission = pd.read_csv(dir_path +'/data/sample_submission.csv')

train_X = train_df.drop(['SalePrice','SalePrice_log','Id'], axis=1)
train_Y = train_df['SalePrice_log']
test_X = test_df.drop(['SalePrice','Id'], axis=1)

with open(dir_path + '/setting/params.yml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


tr = trainer()
pred = predictor()

# tr.train_lgb(train_X, train_Y, params['lgb'])
tr.train_xgb(train_X, train_Y, params['xgb'])
# tr.train_mlpr(train_X, train_Y, params=params['mlpr'])

# pred.predict_lgb(tr.models_lgb[0], test_X)
pred.predict_xgb(tr.models_xgb[0], test_X)
# pred.predict_mlpr(tr.models_mlpr[0], test_X)

# print(tr.rmses_lgb)
print(tr.rmses_xgb) #[0.11601585076934848]
# print(tr.rmses_mlpr) #[0.39306480765719265]

# weight_lgb = (1 / tr.rmses_lgb[0])
# weight_xgb = (1 / tr.rmses_xgb[0])
# weight_mlpr = (1 / tr.rmses_mlpr[0])

# preds_ans1 = pred.preds_lgb[0] * 0.5 + pred.preds_xgb[0] * 0.5
# submission['SalePrice'] = preds_ans1
# submission.to_csv(dir_path +'/submit/sample_submit03.csv', index=False)