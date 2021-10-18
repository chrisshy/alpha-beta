def calculate(df):
    data_temp = df.copy()
#     print(data_temp)
    data_temp['ten']=data_temp['close'].rolling(10).mean()
    data_temp['twenty']=data_temp['close'].rolling(20).mean()
    length = len(data_temp)
    print(data_temp.head(5))
    list_temp = []
    
    for i in range(length):
        if i == length-1:
            list_temp.append(None)
        else:
            buy_price = list(data_temp['open'])[i+1] #当天label是明天的开盘价
            j = i + 2 # 明天买入，后天才能卖出
            while j < length:
                
                if data_temp.iloc[j,7] < data_temp.iloc[j,8]:
                    sell_price = list(data_temp['close'])[j]
                    list_temp.append(sell_price/buy_price-1)
                    break
                j = j + 1
            # 如果循环到测试集最后一行仍然没有卖出的标准，则作为待删除对象
            # 这一暴力方法可以通过扩大测试集数量来解决，因此本质上此方法最方便
            if j >= length:
                list_temp.append(None)
    #         if i == length-2
    index = data_temp.index
    ret_df = pd.DataFrame(list_temp,index=index)
    return ret_df
def createlabel(df,open,close):
#     print(df.head(5))
#     ret = df.groupby('instrument', as_index=False, sort=False, group_keys=False).apply(lambda x: pd.concat([], axis=1))
    ret = df.groupby('instrument', as_index=False, sort=False, group_keys=False).apply(lambda x: pd.concat([calculate(x)],axis=1))
    ret = ret.dropna()
    print(ret)
    return ret


# 本代码由可视化策略环境自动生成 2021年10月18日 12:58
# 本代码单元只能在可视化模式下编辑。您也可以拷贝代码，粘贴到新建的代码单元或者策略，然后修改。


def tma_newstd(df,x5,x10,x30):
    result = []
    for x5,x10,x30 in zip(x5,x10,x30):
        big = max(x5,x10,x30)
        small = min(x5,x10,x30)
        median = np.median(np.array([x5,x10,x30]))
        result.append((big/median + median/small -2)*10)
    return result

def judge(x1,x2):
    if x1>=x2:
        return 1
    else:
        return 0

def cal_qushi(df,close_0,close_2,tma5_0,tma10_0,tma20_0,tma30_0,tma5_2,tma10_2,tma20_2,tma30_2):
    x1 = []
    for close_0,close_2,tma5_0,tma10_0,tma20_0,tma30_0,tma5_2,tma10_2,tma20_2,tma30_2 in zip(close_0,close_2,tma5_0,tma10_0,tma20_0,tma30_0,tma5_2,tma10_2,tma20_2,tma30_2):
        # 这里需要解决数量级不同带来的影响
        # newstd_tma>=0，分散的越厉害，值越大，可能会达到3-4附近
        nowabove = judge(close_0,tma5_0)+judge(close_0,tma10_0)+judge(close_0,tma20_0)+judge(close_0,tma30_0)
        preabove = judge(close_2,tma5_2)+judge(close_2,tma10_2)+judge(close_2,tma20_2)+judge(close_2,tma30_2)
        # difference的大小范围是-4~4
        difference = nowabove-preabove
#         # return_weight的大小范围是0.64平方~1.44平方
#         return_weight = return_2*return_2
        # -4，-3往往伴随着较小的return2
        # 3,4往往伴随着较大的return2
        # 二者在极端情况往往相伴而生  
        x1.append(difference)
    return x1

def guanzhudu_3(df,avg_turn_4):
    avg_5_pre = avg_turn_4.shift(periods=1,axis=0)
    result = avg_turn_4-avg_5_pre
    return result

m2_user_functions_bigquant_run = {
    'tma_newstd': tma_newstd,
    'judge':judge,
    'cal_qushi':cal_qushi,
    'guanzhudu_3':guanzhudu_3
}
def tma_newstd(df,x5,x10,x30):
    result = []
    for x5,x10,x30 in zip(x5,x10,x30):
        big = max(x5,x10,x30)
        small = min(x5,x10,x30)
        median = np.median(np.array([x5,x10,x30]))
        result.append((big/median + median/small -2)*10)
    return result

def judge(x1,x2):
    if x1>=x2:
        return 1
    else:
        return 0

def cal_qushi(df,close_0,close_2,tma5_0,tma10_0,tma20_0,tma30_0,tma5_2,tma10_2,tma20_2,tma30_2):
    x1 = []
    for close_0,close_2,tma5_0,tma10_0,tma20_0,tma30_0,tma5_2,tma10_2,tma20_2,tma30_2 in zip(close_0,close_2,tma5_0,tma10_0,tma20_0,tma30_0,tma5_2,tma10_2,tma20_2,tma30_2):
        # 这里需要解决数量级不同带来的影响
        # newstd_tma>=0，分散的越厉害，值越大，可能会达到3-4附近
        nowabove = judge(close_0,tma5_0)+judge(close_0,tma10_0)+judge(close_0,tma20_0)+judge(close_0,tma30_0)
        preabove = judge(close_2,tma5_2)+judge(close_2,tma10_2)+judge(close_2,tma20_2)+judge(close_2,tma30_2)
        # difference的大小范围是-4~4
        difference = nowabove-preabove
#         # return_weight的大小范围是0.64平方~1.44平方
#         return_weight = return_2*return_2
        # -4，-3往往伴随着较小的return2
        # 3,4往往伴随着较大的return2
        # 二者在极端情况往往相伴而生  
        x1.append(difference)
    return x1

def guanzhudu_3(df,avg_turn_4):
    avg_5_pre = avg_turn_4.shift(periods=1,axis=0)
    result = avg_turn_4-avg_5_pre
    return result

m6_user_functions_bigquant_run = {
    'tma_newstd': tma_newstd,
    'judge':judge,
    'cal_qushi':cal_qushi,
    'guanzhudu_3':guanzhudu_3
}
# 回测引擎：初始化函数，只执行一次
def m14_initialize_bigquant_run(context):
    # 加载预测数据
    context.ranker_prediction = context.options['data'].read_df()

    # 系统已经设置了默认的交易手续费和滑点，要修改手续费可使用如下函数
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
#     print(context.ranker_prediction )
# 回测引擎：每日数据处理函数，每天执行一次
def m14_handle_data_bigquant_run(context, data):
    # 按日期过滤得到今日的预测数据
    ranker_prediction = context.ranker_prediction[
        context.ranker_prediction.date== data.current_dt.strftime('%Y-%m-%d')]
    # 获得收益率在前50%的阈值，此时确认可以买入
#     ranker_low = context.ranker_prediction.pred_label.quantile(q=0)
#     ranker_line = context.ranker_prediction.pred_label.quantile(q=0.5)
#     ranker_high = context.ranker_prediction.pred_label.quantile(q=1)
# 训练集是右偏分布，控制一定损失，占据少数巨额超额收益为目标，这里high用了最大值和平均值的平均代替分位数，单纯用最高值存在异常时段收益干扰
    ranker_low = -0.3731
    ranker_line = -0.0034
    ranker_high = (1.1616+0.040)/2
    ranker_range = ranker_high - ranker_line
    ranker_range_2 = ranker_line - ranker_low
    point = ranker_prediction.iloc[0,0]
    buy_signal = 0
    buy_percent = context.portfolio.positions_value/context.portfolio.portfolio_value
    if point >= ranker_line:
        buy_signal = 1
#     若确定性较高则多买入，小则少买入
        point_position = (point-ranker_line)/ranker_range
    # 有6-无穷大成把握时满仓（0.5+0.2/2）
        if point_position>=0.2:
            buy_percent = 1
    # 有5-6成把握时8-10成
        else:
            buy_percent = 0.8 + point_position
    else:
        buy_signal = 0
        point_position = (ranker_line - point)/ranker_range_2
        # 有4-5成把握（0.5-0.2/2）时 4-7成仓位 
        if point_position < 0.2:
            buy_percent = 0.8 - 1.5*point_position
        else:
             buy_percent = 0
            
        
        
        
    k = context.instruments[0] # 标的为字符串格式
    sid = context.symbol(k) # 将标的转化为equity格式
    price = data.current(sid, 'price') # 最新价格
    
    # 包含多个周期均线值的股票数据
    mavg_10 = data.history(sid, 'price', 10, '1d').mean() # 短期均线值
    mavg_20 = data.history(sid, 'price', 20, '1d').mean() # 短期均线值
    poxian = 0
    if mavg_10 < mavg_20:
        poxian = 1
    
    
    # 目前没有持仓且满足可交易且存在交易信号时进行交易
    if context.portfolio.positions[sid].amount == 0 and data.can_trade(sid) and buy_signal == 1 :
        order_target_percent(sid, buy_percent)
    
    # 目前有持仓时，要么满足条件卖出（破线），要么继续持有
    if context.portfolio.positions[sid].amount != 0:
        if poxian == 1:
            order_target_percent(sid,0)
        else:
            # 调仓问题(防止细小仓位变动带来的手续费压力)
            if abs(context.portfolio.positions_value/context.portfolio.portfolio_value-buy_percent)>=0.2:
                order_target_percent(sid,buy_percent)
            # 继续持有问题
            else:
                pass
    
    
# 回测引擎：准备数据，只执行一次
def m14_prepare_bigquant_run(context):
    pass
# 回测引擎：每个单位时间开始前调用一次，即每日开盘前调用一次。
def m14_before_trading_start_bigquant_run(context, data):
    pass


m1 = M.instruments.v2(
    start_date='2015-01-01',
    end_date='2020-12-01',
    market='CN_STOCK_A',
    instrument_list='300274.SZA',
    max_count=0
)

m7 = M.advanced_auto_labeler.v2(
    instruments=m1.data,
    label_expr="""# #号开始的表示注释
# 0. 每行一个，顺序执行，从第二个开始，可以使用label字段
# 1. 可用数据字段见 https://bigquant.com/docs/develop/datasource/deprecated/history_data.html
#   添加benchmark_前缀，可使用对应的benchmark数据
# 2. 可用操作符和函数见 `表达式引擎 <https://bigquant.com/docs/develop/bigexpr/usage.html>`_

# 计算收益：5日收盘价(作为卖出价格)除以明日开盘价(作为买入价格)
createlabel(open,close)

# 过滤掉一字涨停的情况 (设置label为NaN，在后续处理和训练中会忽略NaN的label)
where(shift(high, -1) == shift(low, -1), NaN, label)
""",
    start_date='',
    end_date='',
    benchmark='000300.SHA',
    drop_na_label=True,
    cast_label_int=False,
    user_functions={'createlabel':createlabel}
)

m3 = M.input_features.v1(
    features="""# #号开始的表示注释
# 多个特征，每行一个，可以包含基础特征和衍生特征

# 目前价格所处位置信息，定义趋势，并认为该趋势可持续
# 上升趋势时，ts_max(close_0, 63)/close_0越趋向1，close_0/ts_min(close_0, 63)越趋向大于1，而差值越趋向负数
# 下降趋势时，同理，差值越趋向正数
# 越震荡时，越趋向接近0的数字
ts_max(close_0, 63)/close_0-close_0/ts_min(close_0, 63)

# ts_max(close_0, 63)/ts_min(close_0, 63)的大小意味着区间内波动大小，意味着此时操作对应的潜在隐含超额收益大小
# 震荡时该值越大，越有操作空间
# 上升和下降趋势时，该值越大，往往操作空间越小（因为以上升为例，此时做T不如一直持有）
ts_max(close_0, 63)/ts_min(close_0, 63)


# 目前价格相对量能是否异常
# 对于价格的辅助判断变量
abs(avg_amount_0/avg_amount_2-1)>0.2
abs(avg_amount_2/avg_amount_20-1)>0.4
abs(avg_amount_2/avg_amount_20-1)>0.5

# 近期价格异常变动情况（往往出现在加速顶部）
return_2
tma_newstd(ta_sma(close_2, timeperiod=5),ta_sma(close_2, timeperiod=10),ta_sma(close_2, timeperiod=30))
cal_qushi(close_0,close_2,ta_sma_5_0,ta_sma_10_0,ta_sma_20_0,ta_sma_30_0,ta_sma(close_2, timeperiod=5),ta_sma(close_2, timeperiod=10),ta_sma(close_2, timeperiod=20),ta_sma(close_2, timeperiod=30))
avg_turn_4/avg_turn_20
guanzhudu_3(avg_turn_4)

# 距离均线位置
close_0/ta_sma(close_0, timeperiod=120)
close_0/ta_sma(close_0, timeperiod=5)
close_0/ta_sma(close_0, timeperiod=30)
close_0/ta_sma(close_0, timeperiod=60)

# k线粘合程度
max(ta_sma_10_0,ta_sma_20_0,ta_sma_30_0)/min(ta_sma_10_0,ta_sma_20_0,ta_sma_30_0)>1.05

# eps
# west_eps_ftm
# instrument"""
)

m15 = M.general_feature_extractor.v7(
    instruments=m1.data,
    features=m3.data,
    start_date='',
    end_date='',
    before_start_days=200
)

m2 = M.derived_feature_extractor.v3(
    input_data=m15.data,
    features=m3.data,
    date_col='date',
    instrument_col='instrument',
    drop_na=False,
    remove_extra_columns=False,
    user_functions=m2_user_functions_bigquant_run
)

m9 = M.join.v3(
    data1=m7.data,
    data2=m2.data,
    on='date,instrument',
    how='inner',
    sort=False
)

m10 = M.dropnan.v2(
    input_data=m9.data
)

m13 = M.random_forest_train.v2(
    training_ds=m10.data,
    features=m3.data,
    n_estimators=30,
    max_features='auto',
    max_depth=30,
    min_samples_leaf=2,
    n_jobs=1,
    random_state=0,
    algo='regressor'
)

m5 = M.instruments.v2(
    start_date=T.live_run_param('trading_date', '2020-12-01'),
    end_date=T.live_run_param('trading_date', '2021-06-28'),
    market='CN_STOCK_A',
    instrument_list="""300274.SZA
""",
    max_count=0
)

m4 = M.general_feature_extractor.v7(
    instruments=m5.data,
    features=m3.data,
    start_date='',
    end_date='',
    before_start_days=200
)

m6 = M.derived_feature_extractor.v3(
    input_data=m4.data,
    features=m3.data,
    date_col='date',
    instrument_col='instrument',
    drop_na=False,
    remove_extra_columns=False,
    user_functions=m6_user_functions_bigquant_run
)

m12 = M.dropnan.v2(
    input_data=m6.data
)

m11 = M.random_forest_predict.v2(
    model=m13.model,
    data=m12.data,
    date_col='date',
    instrument_col='instrument',
    sort=True,
    m_cached=False
)

m14 = M.trade.v4(
    instruments=m5.data,
    options_data=m11.predictions,
    start_date='',
    end_date='',
    initialize=m14_initialize_bigquant_run,
    handle_data=m14_handle_data_bigquant_run,
    prepare=m14_prepare_bigquant_run,
    before_trading_start=m14_before_trading_start_bigquant_run,
    volume_limit=0.025,
    order_price_field_buy='open',
    order_price_field_sell='close',
    capital_base=1000000,
    auto_cancel_non_tradable_orders=True,
    data_frequency='daily',
    price_type='真实价格',
    product_type='股票',
    plot_charts=True,
    backtest_only=False,
    benchmark='300274.SZA'
)