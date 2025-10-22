"""Need to downgrade to python 3.12 for this"""

#scratch 

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

df = pd.read_csv('20251021_etfs_ts.csv')
df['date'] = pd.to_datetime(df['date'])
df['asset_id'] = df['asset_id'].astype('category')

#remove rows with NaN
df = df.dropna().reset_index(drop=True)

test_time = '2025-06-30'

# Filter the DataFrame
test = df[(df['date'] >= test_time)]
train = df[(df['date'] < test_time)]



train_data = TimeSeriesDataFrame.from_data_frame(
    train,
    id_column="asset_id",
    timestamp_column="date"
)
train_data.head()


predictor = TimeSeriesPredictor(
    prediction_length=3,
    freq='BME',
    path="autogluon-timeseries-model",
    target="y_excess_lead",
    eval_metric="MASE",
)

predictor.fit(
    train_data=train_data,
    presets="best_quality",
    time_limit=600,
)

# TimeSeriesDataFrame can also be loaded directly from a file
test_data = TimeSeriesDataFrame.from_data_frame(
    test,
    id_column="asset_id",
    timestamp_column="date"
)

predictions = predictor.predict(train_data)
predictions.head()

# Plot 4 randomly chosen time series and the respective forecasts
predictor.plot(test_data, predictions, quantile_levels=[0.1, 0.9])

predictor.leaderboard(train_data)


# Results show temporal fusion transformer is best model

# 	model	score_test	score_val	pred_time_test	pred_time_val	fit_time_marginal	fit_order
# 0	TemporalFusionTransformer	-0.465915	-0.507964	0.027005	0.008748	79.806317	9
# 1	SeasonalNaive	-0.490764	-0.677504	1.141785	1.221673	1.257056	1
# 2	WeightedEnsemble	-0.546567	-0.501969	0.061433	0.035164	0.240431	13
# 3	TiDE	-0.599592	-0.608309	0.022616	0.005602	58.926377	12
# 4	DirectTabular	-0.604821	-0.522416	0.017985	0.015674	2.006695	3
# 5	ChronosZeroShot[bolt_base]	-0.612706	-0.578802	1.859118	0.637200	0.732948	7
# 6	RecursiveTabular	-0.627236	-0.715248	0.015497	0.010742	2.404821	2
# 7	AutoETS	-0.636998	-0.583500	1.895800	1.743123	0.591314	6
# 8	NPTS	-0.638798	-0.633643	1.153974	1.240318	1.193552	4
# 9	PatchTST	-0.685420	-0.640631	0.009645	0.003396	22.133853	11
# 10	ChronosFineTuned[bolt_small]	-0.685780	-0.713743	0.568544	0.035460	85.471521	8
# 11	DynamicOptimizedTheta	-0.714323	-0.620905	5.050773	1.686946	0.678816	5
# 12	DeepAR	-0.772977	-0.745516	0.018534	0.015414	16.564323	10
