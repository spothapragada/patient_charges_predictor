from pycaret.regression import *
from pycaret.datasets import get_data
import pandas as pd

data = get_data('insurance')
r2 = setup(data,
           target='charges',
           session_id=123,
           normalize=True,
           polynomial_features=True,
           trigonometry_features=True,
           feature_interaction=True,
           bin_numeric_features=['age', 'bmi'],
           )

xgb = create_model('xgboost')

# save transformation pipeline and model
save_model(xgb, '../models/20200530_insurance_xgbreg_deployment_harsha')

