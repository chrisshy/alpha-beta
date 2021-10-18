# 本代码由可视化策略环境自动生成 2021年10月18日 13:05
# 本代码单元只能在可视化模式下编辑。您也可以拷贝代码，粘贴到新建的代码单元或者策略，然后修改。


# 回测引擎：初始化函数，只执行一次
def m19_initialize_bigquant_run(context):
    # 影线或实体比较长时的隶属函数(参数为0.35,0.5)
    def left_linear(x,a,b):
        if x < a:
            return 0
        elif x>=a and x<b:
            return (x-a)/(b-a)
        else:
            return 1
    # 影线或实体中等长度或短(参数为0.15,0.25,0.35,0.5)
    def trapezoid(x,a,b,c,d):
        if x<a:
            return 0
        elif x>= a and x<b:
            return (x-a)/(b-a)
        elif x>= b and x<c:
            return 1
        elif x>=c and x<d:
            return (d-x)/(d-c)
        else:
            return 0
    # 影线或实体极短（或为0,参数为0.05）
    def right_linear(x,a,b):
        if x<a:
            return 1 
        elif x>=a and x<b:
            return (b-x)/(b-a)
        else:
            return 0
    # 实体颜色判断
    def color(x):
        if x['open']>x['close']:
            return 'green'
        elif x['open']<x['close']:
            return 'red'
        else:
            return 'cross'
    # 三角
    def triangle(x,a,b,c):
        if x < a:
            return 0
        elif x>=a and x<b:
            return (x-a)/(b-a)
        elif x>=b and x<c:
            return (c-x)/(c-b)
        else:
            return 0
    def judge_length(x):
        equal_result = right_linear(x,0,0.05)
        short_result = trapezoid(x,0,0.05,0.15,0.25)
        middle_result = trapezoid(x,0.15,0.25,0.35,0.5)
        long_result = left_linear(x,0.35,0.5)
        if max(equal_result,short_result,middle_result,long_result) == equal_result:
            return 'equal'
        elif max(equal_result,short_result,middle_result,long_result) == short_result:
            return 'short'
        elif max(equal_result,short_result,middle_result,long_result) == middle_result:
            return 'middle'
        else:
            return 'long'
    def judge_open_position(x):
        pre_close = x['pre_close']
        pre_open = x['pre_open']
        pre_low = x['pre_low']
        pre_high = x['pre_high']
        x1 = x['open']
        open_low_result = right_linear(x1,pre_low,min(pre_close,pre_open))
        open_equal_low_result = triangle(x1,pre_low,min(pre_close,pre_open),(pre_open+pre_close)/2)
        open_equal_result = triangle(x1,min(pre_close,pre_open),(pre_open+pre_close)/2,max(pre_close,pre_open))
        open_equal_high_result = triangle(x1,(pre_open+pre_close)/2,max(pre_close,pre_open),pre_high)
        open_high_result =left_linear(x1,max(pre_close,pre_open),pre_high)
        if max(open_low_result,open_equal_low_result,open_equal_result,open_equal_high_result,open_high_result) == open_low_result:
            return 'open_low_result'
        elif max(open_low_result,open_equal_low_result,open_equal_result,open_equal_high_result,open_high_result) == open_equal_low_result:
            return 'open_equal_low_result'
        elif max(open_low_result,open_equal_low_result,open_equal_result,open_equal_high_result,open_high_result) == open_equal_result:
            return 'open_equal_result'
        elif max(open_low_result,open_equal_low_result,open_equal_result,open_equal_high_result,open_high_result) == open_equal_high_result:
            return 'open_equal_high_result'
        else:
            return 'open_high_result'
    def judge_close_position(x):
        pre_close = x['pre_close']
        pre_open = x['pre_open']
        pre_low = x['pre_low']
        pre_high = x['pre_high']
        x1 = x['close']
        close_low_result = right_linear(x1,pre_low,min(pre_close,pre_open))
        close_equal_low_result = triangle(x1,pre_low,min(pre_close,pre_open),(pre_open+pre_close)/2)
        close_equal_result = triangle(x1,min(pre_close,pre_open),(pre_open+pre_close)/2,max(pre_close,pre_open))
        close_equal_high_result = triangle(x1,(pre_open+pre_close)/2,max(pre_close,pre_open),pre_high)
        close_high_result =left_linear(x1,max(pre_close,pre_open),pre_high)
        if max(close_low_result,close_equal_low_result,close_equal_result,close_equal_high_result,close_high_result) == close_low_result:
            return 'close_low_result'
        elif max(close_low_result,close_equal_low_result,close_equal_result,close_equal_high_result,close_high_result) == close_equal_low_result:
            return 'close_equal_low_result'
        elif max(close_low_result,close_equal_low_result,close_equal_result,close_equal_high_result,close_high_result) == close_equal_result:
            return 'close_equal_result'
        elif max(close_low_result,close_equal_low_result,close_equal_result,close_equal_high_result,close_high_result) == close_equal_high_result:
            return 'close_equal_high_result'
        else:
            return 'close_high_result'
   
    # 加载预测数据
    # context.ranker_prediction = context.options['data'].read_df()

    # 系统已经设置了默认的交易手续费和滑点，要修改手续费可使用如下函数
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
    # 加载原始数据
    stock_raw_data = D.history_data(context.instruments, '2010-01-01', context.end_date, ['open','close','low','high'])
    # 目标1：获取每一日的蜡烛图的模糊信号
    stock_raw_data['lupper'] = (stock_raw_data['high']-stock_raw_data[["open", "close"]].max(axis=1))/stock_raw_data['open']*10
    stock_raw_data['llower'] = (stock_raw_data[["open", "close"]].min(axis=1)-stock_raw_data['low'])/stock_raw_data['open']*10
    stock_raw_data['lbody'] = (stock_raw_data[["open", "close"]].max(axis=1)-stock_raw_data[["open", "close"]].min(axis=1))/stock_raw_data['open']*10
    stock_raw_data['lupper_result'] = stock_raw_data['lupper'].apply(lambda x: judge_length(x))
    stock_raw_data['llower_result'] = stock_raw_data['llower'].apply(lambda x: judge_length(x))
    stock_raw_data['lbody_result'] = stock_raw_data['lbody'].apply(lambda x: judge_length(x))
    
    #目标2：获取每一日相对前一个交易日位置的模糊信号
    stock_raw_data_shift = stock_raw_data.shift(axis=0)
    stock_raw_data['pre_open'] = stock_raw_data_shift['open']
    stock_raw_data['pre_close'] = stock_raw_data_shift['close']
    stock_raw_data['pre_low'] = stock_raw_data_shift['low']
    stock_raw_data['pre_high'] = stock_raw_data_shift['high']
    stock_raw_data['open_position'] = stock_raw_data.apply(lambda x:judge_open_position(x),axis=1)
    stock_raw_data['close_position'] = stock_raw_data.apply(lambda x:judge_close_position(x),axis=1)

    #目标3：获取每一日的涨跌颜色
    stock_raw_data['color'] = stock_raw_data.apply(lambda x: color(x),axis=1)
    
    #目标4：获得每一日后5天收益率作为y
    stock_raw_data_shift2 = stock_raw_data.shift(periods=-5, axis=0)
    stock_raw_data['target_day_close'] = stock_raw_data_shift2['close']
    stock_raw_data['return'] = (stock_raw_data['target_day_close'] - stock_raw_data['close'])/stock_raw_data['close']
        
    # map 一下各个str信号
    stock_raw_data['lupper_result_map'] = stock_raw_data['lupper_result'].map({'equal':0,'short':1,'middle':2,'long':3})
    stock_raw_data['llower_result_map'] = stock_raw_data['llower_result'].map({'equal':0,'short':1,'middle':2,'long':3})
    stock_raw_data['lbody_result_map'] = stock_raw_data['lbody_result'].map({'equal':0,'short':1,'middle':2,'long':3})
    stock_raw_data['open_position_map'] = stock_raw_data['open_position'].map({'open_low_result':0,'open_equal_low_result':1,'open_equal_result':2,'open_equal_high_result':3,'open_high_result':4})
    stock_raw_data['close_position_map'] = stock_raw_data['close_position'].map({'close_low_result':0,'close_equal_low_result':1,'close_equal_result':2,'close_equal_high_result':3,'close_high_result':4})
    stock_raw_data['color_map'] = stock_raw_data['color'].map({'green':0,'cross':1,'red':2})
    
    context.stock_raw_data = stock_raw_data
    print(stock_raw_data)
# 回测引擎：每日数据处理函数，每天执行一次

def m19_handle_data_bigquant_run(context, data):
    # 按日期过滤得到今日的预测数据
    from scipy import stats
    
    def check_normality(testData):
        p_value= stats.kstest(testData,'norm')[1]
        if p_value<0.05:
            print ("data are not normal distributed")
            return  False
        else:
            print ("data are normal distributed")
            return True
        
    def max_min_norm(x):
        """
        args: x:np.array
        """
        max_ = x.max()
        min_ = x.min()
        new_val = (x-min_)/(max_-min_)
        return new_val
           
    def data_Transform_boxcox(df, cols):
        df_n = df.copy()
        min_num = df_n[cols].min()
        if df_n[cols].min()<=0:
            df_n[cols] = df_n[cols]-min_num+1
        xt, _ = stats.boxcox(df_n[cols])
        df_n[cols+'_Boxcox'] = xt
        return df_n[cols+'_Boxcox']
    
    def dtw_distance(ts_a, ts_b):
        """Computes dtw distance between two time series
        Args:
            ts_a: time series a
            ts_b: time series b
            d: distance function 此处更改为了abs(x-y)
            mww: max warping window, int, optional (default = infinity)
        Returns:
            dtw distance
        """
        # Create cost matrix via broadcasting with large int
        M, N = len(ts_a), len(ts_b)
        cost = np.ones((M, N))
        # Initialize the first row and column
        cost[0, 0] = abs(ts_a[0] - ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i-1, 0] + abs(ts_a[i] - ts_b[0])
        for j in range(1, N):
            cost[0, j] = cost[0, j-1] + abs(ts_a[0] - ts_b[j])
        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(1, N):
                choices = cost[i-1, j-1], cost[i, j-1], cost[i-1, j]
                cost[i, j] = min(choices) + abs(ts_a[i] - ts_b[j])
        # Return DTW distance given window 
        return cost[-1, -1]

    
    stock_data = context.stock_raw_data
    k = stock_data.instrument[0]
    sid = context.symbol(k)

    data_current_index = stock_data[stock_data['date']==data.current_dt.strftime('%Y-%m-%d')].index.tolist()
    data_current = stock_data.loc[data_current_index[0]-63:data_current_index[0],'lupper_result_map':'color_map']
    data_current_numpy_line = np.array(data_current).ravel()
    data_current_close_line = np.array(stock_data.loc[data_current_index[0]-63:data_current_index[0],'close']).ravel()
    total_line = stock_data[stock_data['date']<data.current_dt.strftime('%Y-%m-%d')].shape[0]
    
    # 先保存每一次计算的结果，求相似度分布
    cal_result_list = []
    # 减去部分数据是因为不能包括“自己”的部分，不然存在自己到自己情况
    for i in range(63,total_line-63):
        previous_data = stock_data.loc[i-63:i,'lupper_result_map':'color_map']
        previous_data_close_numpy = np.array(stock_data.loc[i-63:i,'close']).ravel()
        # previous_data = stock_data.iloc[i-63:i,:][['lupper_result_map','llower_result_map','lbody_result_map','open_position_map','close_position_map','color_map']]
        fore_pred = stock_data.loc[i,'return']
        previous_data_numpy = np.array(previous_data)
        previous_data_line = previous_data_numpy.ravel()
        difference = data_current_numpy_line - previous_data_line
        
        # 未来需要优化的重点部分，即difference ratio的计算
        # difference ratio的计算
        # 方法1-1
        # difference_ratio = np.sum(abs(difference))/difference.size
        # 方法1-2
        # difference_ratio = np.sum(difference*difference)/difference.size
        # 方法1-3
        difference_ratio = 1 - (np.sum(difference == 0)+np.sum(difference == 1))/difference.size
        # K线序列的相似度DTW计算
        distance = dtw_distance(max_min_norm(data_current_close_line), max_min_norm(previous_data_close_numpy))
        
        cal_result_list.append([i,difference_ratio,fore_pred,distance])
    cal_result_numpy = np.array(cal_result_list)
    cal_result_df = pd.DataFrame(cal_result_list,columns=['index','difference_ratio','fore_pred','dtw_distance'])
    
    # 检查difference分布，利用两标准差选取合适的difference ratio对应行
    normality_dif = check_normality(cal_result_numpy[:,1])
    normality_dtw = check_normality(cal_result_numpy[:,3])
    # 我们认为如果在 mean-2standard 之外（95%概率），属于及其相似
    if normality_dif:
        mean = np.mean(cal_result_numpy[:,1])
        standard = np.std(cal_result_numpy[:,1])
        target_difference_standard = mean - 2*standard
        cal_result_df['judge_dif'] = cal_result_df['difference_ratio'].apply(lambda x: 1 if x<target_difference_standard else 0)
    else:
        cal_result_df['norm_trans_dif'] = data_Transform_boxcox(cal_result_df,'difference_ratio')
        mean = cal_result_df['norm_trans_dif'].mean()
        standard = cal_result_df['norm_trans_dif'].std()
        target_difference_standard = mean - 2*standard
        cal_result_df['judge_dif'] = cal_result_df['norm_trans_dif'].apply(lambda x: 1 if x<target_difference_standard else 0)  
    if normality_dtw:
        mean = np.mean(cal_result_numpy[:,3])
        standard = np.std(cal_result_numpy[:,3])
        target_difference_standard = mean - 2*standard
        cal_result_df['judge_dtw'] = cal_result_df['dtw_distance'].apply(lambda x: 1 if x<target_difference_standard else 0)
    else:
        cal_result_df['norm_trans_dtw'] = data_Transform_boxcox(cal_result_df,'dtw_distance')
        mean = cal_result_df['norm_trans_dtw'].mean()
        standard = cal_result_df['norm_trans_dtw'].std()
        target_difference_standard = mean - 2*standard
        cal_result_df['judge_dtw'] = cal_result_df['norm_trans_dtw'].apply(lambda x: 1 if x<target_difference_standard else 0)  
    # trading 逻辑
    cal_fore_set = cal_result_df[(cal_result_df['judge_dtw']==1) | (cal_result_df['judge_dif']==1)]

#     cal_fore_set = cal_result_df[cal_result_df['judge_dif']==1].append(cal_result_df[cal_result_df['judge_dtw']==1 & cal_result_df['judge_dif']==0])
#     cal_fore_set = cal_result_df[cal_result_df['judge_dtw']==1]
    print(cal_fore_set)

    print(len(cal_fore_set))
    # 找到相似度高的匹配天数后如何确定对应y和买入条件
    # 直观的感受是：此种模式未来满足右偏分布（即买入后获利概率较大，其他方式有待尝试）时可以买入
    skewness = cal_fore_set['fore_pred'].skew()
    fore_mean = cal_fore_set['fore_pred'].mean()
    if (skewness > 0) & (fore_mean>0):
        # 交易函数，买入+持有 逻辑
        order_target_percent(sid,1)
    else:
        order_target_percent(sid,0)

# 回测引擎：准备数据，只执行一次
def m19_prepare_bigquant_run(context):
    pass


m1 = M.instruments.v2(
    start_date=T.live_run_param('trading_date', '2020-09-30'),
    end_date=T.live_run_param('trading_date', '2021-05-30'),
    market='CN_STOCK_A',
    instrument_list='000858.SZA',
    max_count=0
)

m19 = M.trade.v4(
    instruments=m1.data,
    start_date='',
    end_date='',
    initialize=m19_initialize_bigquant_run,
    handle_data=m19_handle_data_bigquant_run,
    prepare=m19_prepare_bigquant_run,
    volume_limit=0.025,
    order_price_field_buy='open',
    order_price_field_sell='open',
    capital_base=1000000,
    auto_cancel_non_tradable_orders=True,
    data_frequency='daily',
    price_type='真实价格',
    product_type='股票',
    plot_charts=True,
    backtest_only=False,
    benchmark='000858.SZA'
)