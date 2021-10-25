# alpha-beta

This is a Quantitive Development toolkit.

These python files were written based on Bigquant Platform APIs.
1. link: https://bigquant.com/
2. These models are all done in terms of technical analysis. 
3. In terms of fundamental logic: https://github.com/chrisshy/photovoltaic_industry

The performance of these strategies can be found here:
1. multi-factor_model_300274: https://bigquant.com/live/shared/strategy?id=51158
2. industry_based_simple_control_group: https://bigquant.com/live/shared/strategy?id=55358
3. DTW_matching_and_fuzzy_mathematics: https://bigquant.com/live/shared/strategy?id=51807
4. multi-factor_model_filtered_stocks: https://bigquant.com/live/shared/strategy?id=53321

These models are constructed based on the following logics:
## General Objective: 
Predict stock performance(s) and choose the optimal one to trade the next day.
1. Use simple traditional models(without non-linear models) to trade in a specific group of stocks.
2. Test the performance of the multi-factor model in different situations(a single stock or a bunch of stocks).
3. Directly compare the shape of the candlestick chart with traditional historical patterns 

## Specific objectives: 
1. 
2. 

## Training Period & Model Choice
1. Why RF in multi-factor: low variance of prediction results.
2. Different from grouped stocks, why begin from 2015 when training 300274: 
3. Why fuzzy mathematics: 
4. Why DTW matching:
