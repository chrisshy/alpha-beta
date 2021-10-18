# 本代码由可视化策略环境自动生成 2021年10月18日 12:56
# 本代码单元只能在可视化模式下编辑。您也可以拷贝代码，粘贴到新建的代码单元或者策略，然后修改。


# Python 代码入口函数，input_1/2/3 对应三个输入端，data_1/2/3 对应三个输出端
def m6_run_bigquant_run(input_1, input_2, input_3):
    # 示例代码如下。在这里编写您的代码
    df = DataSource.read_df(input_1)
    columns = ['close_0','close_1','date','instrument','return_7','ta_sma_30_0','deviation','cross'] 
    df.columns = columns
    data_1 = DataSource.write_df(df)
#     data_2 = DataSource.write_pickle(df)
    return Outputs(data_1=data_1, data_2=None, data_3=None)

# 后处理函数，可选。输入是主函数的输出，可以在这里对数据做处理，或者返回更友好的outputs数据格式。此函数输出不会被缓存。
def m6_post_run_bigquant_run(outputs):
    return outputs

# 回测引擎：初始化函数，只执行一次
def m2_initialize_bigquant_run(context):
    # 加载预测数据
    context.ranker_prediction = context.options['data'].read_df()
    # 系统已经设置了默认的交易手续费和滑点，要修改手续费可使用如下函数
#     print(context.ranker_prediction)
    # 建立初始仓位
    stock_list = context.instruments
    stock_percent_check_dict={
        '600519.SHA':0.04,
        '000858.SZA':0.04,
        '603259.SHA':0.04,
        '600276.SHA':0.04,
        '300015.SZA':0.04,
        '300896.SZA':0.04,
        '300285.SZA':0.04,
        '002138.SZA':0.04,
        '002241.SZA':0.04,
        '600893.SHA':0.04,
        '603501.SHA':0.04,
        '002049.SZA':0.04,
        '688012.SHA':0.04,
        '002371.SZA':0.04,
        '300655.SZA':0.04,
        '600584.SHA':0.04,
        '600460.SHA':0.04,
        '603290.SHA':0.04,
        '300274.SZA':0.04,
        '601012.SHA':0.04,
        '300750.SZA':0.04,
        '002812.SZA':0.01,
        '002709.SZA':0.01,
        '002460.SZA':0.01,
        '300073.SZA':0.01,
        '300450.SZA':0.01,
        '300724.SZA':0.01,
        '601318.SHA':0.03,
        '600036.SHA':0.03,
        '002271.SZA':0.03,
        '600111.SHA':0.03
    }
    
    context.check_dict = stock_percent_check_dict
    context.count = 0
    
    # 构建买卖逻辑，创造每只股票每天的买卖选择
#     context.ranker_prediction['target_percent'] = 0
    # 1. cross为-2时卖出，为2时买入
#     context.ranker_prediction[]
    
    
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
    # 预测数据，通过options传入进来，使用 read_df 函数，加载到内存 (DataFrame)
    # 设置买入的股票数量，这里买入预测股票列表排名靠前的
    
    
# 回测引擎：每日数据处理函数，每天执行一次
def m2_handle_data_bigquant_run(context, data):
    # 按日期过滤得到今日的预测数据
    # 初始建仓
    if context.count == 0 :
        for key in context.check_dict:
            sid = context.symbol(key) 
            order_target_percent(sid,context.check_dict[key])
        context.count += 1
    else:
        context.count += 1
#         print(context.count)
#         print(DataSource('basic_info_CN_FUTURE').read())
#         price_history = DataSource('bar1d_CN_FUTURE').read(instrument=['IF2103.SHF'])
    
#         print(price_history)


        data = context.ranker_prediction[
            context.ranker_prediction.date == data.current_dt.strftime('%Y-%m-%d')]
        data = data.fillna(1)
        positions_hold_dict = {}
        for stock in context.portfolio.positions:
            value = context.portfolio.positions[stock].amount * context.portfolio.positions[stock].last_sale_price
            position = value/context.portfolio.portfolio_value
            positions_hold_dict[stock] = position
#         print('****')
#         print(positions_hold_dict)
        data['sid'] = data['instrument'].apply(lambda x : context.symbol(x))
        data.index = data['sid']
        data['new_target_percent'] = 0
        for stock in positions_hold_dict:
            data.loc[stock,'new_target_percent'] = positions_hold_dict[stock]

        data['new_target_percent'] = data.apply(lambda x: x['new_target_percent']-0.003 if x['return_7']>1.4 else x['new_target_percent'],axis=1)
        data['new_target_percent'] = data.apply(lambda x: x['new_target_percent']+0.005 if x['return_7']<0.8 else x['new_target_percent'],axis=1)

        data['new_target_percent'] = data.apply(lambda x: x['new_target_percent']-0.003 if x['deviation']>1.5 else x['new_target_percent'],axis=1)
        data['new_target_percent'] = data.apply(lambda x: x['new_target_percent']+0.005 if x['deviation']<0.7 else x['new_target_percent'],axis=1)

        data['new_target_percent'] = data.apply(lambda x: context.check_dict[x['instrument']] if x['cross']==2 else x['new_target_percent'],axis=1)
        data['new_target_percent'] = data.apply(lambda x: 0 if x['cross']==-2 else x['new_target_percent'],axis=1)

        # 至少保证每个股票有一定底仓，且单个不超过总持仓的10%
        data['new_target_percent'] = data.apply(lambda x: 0.002 if x['new_target_percent']<=0 else x['new_target_percent'],axis=1)
        data['new_target_percent'] = data.apply(lambda x: 0.1 if x['new_target_percent']>=0.1 else x['new_target_percent'],axis=1)

        # 总持仓比例检查，若超过1，则降低个股降低比例
        adjust_before = data['new_target_percent'].sum()
        while adjust_before > 0.99:
            data['new_target_percent'] = data['new_target_percent']-0.001
            # 确保每个有底仓
            data['new_target_percent'] = data.apply(lambda x: 0.002 if x['new_target_percent']<=0 else x['new_target_percent'],axis=1)
            adjust_before =  data['new_target_percent'].sum()
#         print(data['new_target_percent'])
        
        #解决停牌股票维持原有仓位的问题
        data = data.dropna()
        
        # 进行每日的买卖
        for i in range(0,len(data)):
            order_target_percent(data['sid'].iloc[i],data['new_target_percent'].iloc[i])
        


# 回测引擎：准备数据，只执行一次
def m2_prepare_bigquant_run(context):
    pass

# 回测引擎：每个单位时间开始前调用一次，即每日开盘前调用一次。
def m2_before_trading_start_bigquant_run(context, data):
    pass


m1 = M.instruments.v2(
    start_date=T.live_run_param('trading_date', '2018-01-01'),
    end_date=T.live_run_param('trading_date', '2021-07-11'),
    market='CN_STOCK_A',
    instrument_list="""#白酒（4%+4%=8%）
#贵州茅台，五粮液
600519.SHA
000858.SZA

#医药+医美+CRO（4%+4%+4%+4%=16%）
#药明康德，恒瑞医药，爱尔眼科，爱美客
603259.SHA
600276.SHA
300015.SZA
300896.SZA

#电子+军工（4%+4%+4%+4%=16%）
#国瓷材料，顺络电子，歌尔股份，航发动力
300285.SZA
002138.SZA
002241.SZA
600893.SHA

#半导体（4%+4%+4%+4%+4%+4%+4%+4%=24%）
#韦尔股份，紫光国微，中微公司，北方华创，晶瑞股份，长电科技，士兰微，斯达半导
603501.SHA
002049.SZA
688012.SHA
002371.SZA
300655.SZA
600584.SHA
600460.SHA
603290.SHA


#新能源汽车+光伏（4%+4%+4%+1%+1%+1%+1%+1%+1%+1%=18%）
#阳光电源，隆基股份，宁德时代，恩捷股份，天赐材料，赣锋锂业，当升科技，先导智能，捷佳伟创
300274.SZA
601012.SHA
300750.SZA
002812.SZA
002709.SZA
002460.SZA
300073.SZA
300450.SZA
300724.SZA


#金融+白马+稀土(3%+3%+3%+3%=12%)
#中国平安，招商银行，东方雨虹，北方稀土
601318.SHA
600036.SHA
002271.SZA
600111.SHA

#鸿蒙生态链——等待业绩龙头出现
#以上共94%,31个股票

#其他(1%+1%+1%+1%)
#北方稀土，中科创达，顺丰控股，中国中免
# 600111.SHA
# 300496.SZA
# 002352.SZA
# 601888.SHA



""",
    max_count=0
)

m3 = M.input_features.v1(
    features="""
# #号开始的表示注释，注释需单独一行
# 多个特征，每行一个，可以包含基础特征和衍生特征，特征须为本平台特征
return_7
close_0/ta_sma_30_0
sign(close_0-ta_sma_30_0)-sign(close_1-ta_sma(close_1, 30))
# 上穿为2，不变为0，下穿为-2

"""
)

m4 = M.general_feature_extractor.v7(
    instruments=m1.data,
    features=m3.data,
    start_date='',
    end_date='',
    before_start_days=60
)

m5 = M.derived_feature_extractor.v3(
    input_data=m4.data,
    features=m3.data,
    date_col='date',
    instrument_col='instrument',
    drop_na=False,
    remove_extra_columns=False,
    user_functions={}
)

m6 = M.cached.v3(
    input_1=m5.data,
    run=m6_run_bigquant_run,
    post_run=m6_post_run_bigquant_run,
    input_ports='',
    params='{}',
    output_ports=''
)

m2 = M.trade.v4(
    instruments=m1.data,
    options_data=m6.data_1,
    start_date='2019-01-01',
    end_date='2021-07-11',
    initialize=m2_initialize_bigquant_run,
    handle_data=m2_handle_data_bigquant_run,
    prepare=m2_prepare_bigquant_run,
    before_trading_start=m2_before_trading_start_bigquant_run,
    volume_limit=0.05,
    order_price_field_buy='close',
    order_price_field_sell='close',
    capital_base=100000000,
    auto_cancel_non_tradable_orders=True,
    data_frequency='daily',
    price_type='真实价格',
    product_type='股票',
    plot_charts=True,
    backtest_only=False,
    benchmark=''
)