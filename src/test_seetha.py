from pycaret.datasets import get_data
from pycaret.regression import *
data = get_data('insurance')

#basic setup
r2 = setup(data, target='charges', session_id=123,
           normalize=True,
           polynomial_features=True, trigonometry_features=True,
           feature_interaction=True,
           bin_numeric_features=['age', 'bmi'])

# create xgb regr model
xgb = create_model('xgboost')

# save model pickle
save_model(xgb, '../models/20200530_insurance_xgbreg_deployment_seetha')