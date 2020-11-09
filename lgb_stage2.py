import pandas as pd
import numpy as np
import lightgbm as lgb

in_df_stage1 = pd.read_excel('data/入库流量数据.xlsx')
env_df_stage1 = pd.read_excel('data/环境表.xlsx')
rain_df_stage1 = pd.read_excel('data/遥测站降雨数据.xlsx')
rain_fore_df_stage1 = pd.read_excel('data/降雨预报数据.xlsx')

in_df_stage2 = pd.read_excel('data_stage2/入库流量数据.xlsx')
env_df_stage2 = pd.read_excel('data_stage2/环境表.xlsx')
rain_df_stage2 = pd.read_excel('data_stage2/遥测站降雨数据.xlsx')
rain_fore_df_stage2 = pd.read_excel('data_stage2/降雨预报数据.xlsx')

sub = pd.read_csv('data/submission.csv')

# 补充测试集时间
test_add_df = pd.DataFrame()
test_add_df['year'] = 2019*np.ones(56*5, dtype=int)
m = np.ones(56*5, dtype=int)
test_month = [2, 4, 6, 8, 11]
for i in range(5):
    print(i)
    m[i*56:(i+1)*56] = test_month[i]
test_add_df['month'] = m
for i in range(5):
    for j in range(7):
        for k in range(8):
            m[i*56+j*8+k] = j+1
test_add_df['day'] = m
for i in range(5*7):
    for j in range(8):
        m[i*8+j] = 3*j+2
test_add_df['hour'] = m

# 填充空值
for df in [in_df_stage1, env_df_stage1, rain_df_stage1, rain_fore_df_stage1,
           in_df_stage2, env_df_stage2, rain_df_stage2, rain_fore_df_stage2]:
    df = df.fillna(method='ffill')

# 将初赛复赛数据组合在一起
in_df = pd.concat([in_df_stage1, in_df_stage2]).reset_index(drop=True)
env_df = pd.concat([env_df_stage1, env_df_stage2]).reset_index(drop=True)
rain_df = pd.concat([rain_df_stage1, rain_df_stage2]).reset_index(drop=True)
rain_fore_df = pd.concat([rain_fore_df_stage1, rain_fore_df_stage2]).reset_index(drop=True)

# 数据集提取
# 读取流量数据
time_list = in_df.TimeStample.map(
    lambda x: [x.year, x.month, x.day, x.hour])
in_df['year'] = time_list.map(lambda x: int(x[0]))
in_df['month'] = time_list.map(lambda x: int(x[1]))
in_df['day'] = time_list.map(lambda x: int(x[2]))
in_df['hour'] = time_list.map(lambda x: int(x[3]))
in_df.drop(['TimeStample'], axis=1, inplace=True)
in_df = pd.concat([in_df, test_add_df])
in_df = in_df.sort_values(by=['year', 'month', 'day', 'hour']).reset_index(drop=True)

# 读取降雨量数据
time_list = rain_fore_df.TimeStample.map(
    lambda x: [x.year, x.month, x.day])
rain_fore_df['year'] = time_list.map(lambda x: int(x[0]))
rain_fore_df['month'] = time_list.map(lambda x: int(x[1]))
rain_fore_df['day'] = time_list.map(lambda x: int(x[2]))
rain_fore_df.drop(['TimeStample'], axis=1, inplace=True)
for name in ['D1', 'D2', 'D3', 'D4', 'D5']:
    rain_fore_df[name+'_mean_month'] = rain_fore_df.groupby(['month', 'year']).transform('mean')[name]

# 读取环境数据
time_list = env_df.TimeStample.map(lambda x: x.split('-'))
env_df['year'] = time_list.map(lambda x: int(x[0]))
env_df['month'] = time_list.map(lambda x: int(x[1]))
env_df['day'] = time_list.map(lambda x: int(x[2]))
env_df.drop(['TimeStample'], axis=1, inplace=True)

df_dummies = pd.get_dummies(env_df["wd"]) # onehot编码
df_dummies.columns = ['wd_'+str(x+1) for x in range(16)]
df_dummies.loc[:] = df_dummies.values
env_df = pd.concat([env_df.drop(['wd'], axis=1), df_dummies], axis=1)

# 读取天气预报
time_list = rain_df.TimeStample.map(
        lambda x: [x.year, x.month, x.day, x.hour])
rain_df['year'] = time_list.map(lambda x: int(x[0]))
rain_df['month'] = time_list.map(lambda x: int(x[1]))
rain_df['day'] = time_list.map(lambda x: int(x[2]))
rain_df['hour'] = time_list.map(lambda x: int(x[3]))
rain_df.drop(['TimeStample'], axis=1, inplace=True)

feats_R = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16',
                 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R27', 'R28', 'R29', 'R30',
                 'R31', 'R32', 'R33', 'R34', 'R35', 'R36', 'R37', 'R38', 'R39']

# 每3小时求和
tmp = rain_df.copy()
tmp.index = pd.date_range('1/1/2013', periods=49776, freq='H')
idx = rain_df[rain_df['hour']%3==2].index
rain_df.loc[idx, feats_R] = tmp[feats_R].resample('3H').sum().values

for df in [in_df, env_df, rain_df, rain_fore_df]:
    df = df.fillna(method='ffill')

# 合表
data_df = pd.merge(in_df, rain_df, on=['year', 'month', 'day', 'hour'], how='left')
data_df = pd.merge(data_df, env_df, on=['year', 'month', 'day'], how='left')
data_df = pd.merge(data_df, rain_fore_df, on=['year', 'month', 'day'], how='left')
fill_name = list(data_df.columns)

print(list(data_df.columns))

# 训练集测试集划分
train0 = data_df[(data_df['year'] <= 2017)]
print('------------train0')
print(train0.shape)
train = [data_df[(data_df['year'] == 2019) & (data_df['month'] == 1)],
         data_df[(data_df['year'] == 2019) & (data_df['month'] == 3)],
         data_df[(data_df['year'] == 2019) & (data_df['month'] == 5)],
         data_df[(data_df['year'] == 2019) & (data_df['month'] == 7)],
         data_df[(data_df['year'] == 2019) & (data_df['month'] == 10)]]

test = [data_df[(data_df['year'] == 2019) & (data_df['month'] == 2) & (data_df['day'] >= 1) & (data_df['day'] <= 7)],
        data_df[(data_df['year'] == 2019) & (data_df['month'] == 4) & (data_df['day'] >= 1) & (data_df['day'] <= 7)],
        data_df[(data_df['year'] == 2019) & (data_df['month'] == 6) & (data_df['day'] >= 1) & (data_df['day'] <= 7)],
        data_df[(data_df['year'] == 2019) & (data_df['month'] == 8) & (data_df['day'] >= 1) & (data_df['day'] <= 7)],
        data_df[(data_df['year'] == 2019) & (data_df['month'] == 11) & (data_df['day'] >= 1) & (data_df['day'] <= 7)]]

# 时间特征值
_feats = ['month', 'day', 'hour']

# 5段测试集
for i in range(5):
    print('=================================================================%d'%i)
    ans = np.zeros(57)
    ans[0] = i+1
    # 按天建模 每天为不同的取不同的特征  体现在不同长度的同期历史值
    for ddd in range(1, 8):
        print('------------------------------------------------------------%d'%ddd)
        feats = _feats.copy()
        train_data = pd.concat([train0, train[i]]).reset_index(drop=True)
        test_data = test[i].reset_index(drop=True).copy()

        df_all = pd.concat([train_data, test_data]).reset_index(drop=True)
        print(df_all.shape)

        # 对天气预报值加权
        for day in range(2,6):
            df_all['D%d_%d'%(day, 8*(day-1))] = df_all['D%d'%day].shift(8*(day-1))
        df_all['D_mean'] = df_all['D1']*0.4+df_all['D2_8']*0.3+df_all['D3_16']*0.2+df_all['D4_24']*0.05+df_all['D5_32']*0.05
        feats = feats + ['D_mean']

        # 将天气预报变换到同一天并加权平均
        for day in range(1,6):
            df_all['D%d_yesterday'%day] = df_all['D%d'%day].shift(8*day)

        df_all['D_mean_yesterday'] = df_all['D1_yesterday']*0.4+df_all['D2_yesterday']*0.3+df_all['D3_yesterday']*0.2+\
                                     df_all['D4_yesterday']*0.05+df_all['D5_yesterday']*0.05

        df_all['D_mean_yesterday1'] = df_all['D1_yesterday']*0.3+df_all['D2_yesterday']*0.3+df_all['D3_yesterday']*0.3+\
                                     df_all['D4_yesterday']*0.05+df_all['D5_yesterday']*0.05

        feats = feats + ['D1_yesterday', 'D2_yesterday', 'D_mean_yesterday', 'D_mean_yesterday1']

        # 计算遥测降雨总量和重要度较大的几个的均值
        df_all['R_all'] = df_all[feats_R].sum(axis=1)
        df_all['R_mean'] = df_all[['R39', 'R20', 'R1', 'R23', 'R10', 'R24']].mean(axis=1)

        # 入库流量的移动平均
        df_all['Qi_rolling'] = df_all['Qi'].rolling(window=5, min_periods=1, center=True).mean()

        # 求各种特征的历史值
        for day in [1, 3, 5, 7]:
            df_all['D_mean_yesterday_%d' % day] = df_all['D_mean_yesterday'].shift(8 * day)
            feats.append('D_mean_yesterday_%d' % day)

        for day in range(ddd, 7):
            df_all['Qi_%d' % day] = df_all['Qi_rolling'].shift(8 * day)
            feats.append('Qi_%d' % day)
            for name in ['R_all', 'R_mean', 'T', 'w']:
                df_all[name + '_%d' % day] = df_all[name].shift(8 * day)
                feats.append(name + '_%d' % day)
            for wd in range(16):
                df_all['wd_%d_%d' % (wd + 1, day)] = df_all['wd_%d' % (wd + 1)].shift(8 * day)
                feats.append('wd_%d_%d' % (wd + 1, day))

        for day in [7, 10, 14, 20, 30]:
            df_all['Qi_%d'%day] = df_all['Qi_rolling'].shift(8*day)
            feats.append('Qi_%d'%day)
            for name in ['R_all', 'R_mean', 'T', 'w']:
                df_all[name + '_%d' % day] = df_all[name].shift(8 * day)
                feats.append(name + '_%d' % day)
            for wd in range(16):
                df_all['wd_%d_%d' % (wd + 1, day)] = df_all['wd_%d' % (wd + 1)].shift(8 * day)
                feats.append('wd_%d_%d' % (wd + 1, day))

        df_all['Qi_sub_1'] = df_all['Qi_7'] - df_all['Qi_10']
        df_all['Qi_sub_2'] = df_all['Qi_10'] - df_all['Qi_14']
        df_all['Qi_sub_3'] = df_all['Qi_14'] - df_all['Qi_20']
        df_all['Qi_sub_4'] = df_all['Qi_20'] - df_all['Qi_30']
        feats = feats + ['Qi_sub_1', 'Qi_sub_2', 'Qi_sub_3', 'Qi_sub_4']

        print(feats)

        # 取训练集和测试集
        train_data = df_all[:-56]
        test_data = df_all[-56:]
        train_x = train_data[feats]
        train_y = train_data['Qi']
        test_x = test_data[feats]
        print(train_x.shape, test_x.shape)
        train_matrix = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
        val_matrix = lgb.Dataset(test_x, label=test_data['Qi'])

        test_matrix = lgb.Dataset(test_x)
        y = 0
        # 多随机种子
        for seed in [2, 222, 22222]:
            params = {
                'boosting_type': 'gbdt',
                'max_depth': 12,
                'objective': 'mse',
                'min_child_weight': 32,
                'feature_fraction': 0.5,
                'bagging_fraction': 0.5,
                'bagging_freq': 1,
                'learning_rate': 0.01,
                'seed': seed,
                'verbose': -1
            }
            model = lgb.train(params, train_matrix, num_boost_round=600)
            this_y = model.predict(test_x)
            y += this_y/3

        ans[1+(ddd-1)*8:ddd*8+1] = y[(ddd-1)*8:ddd*8]

    sub.loc[i] = ans
print(sub)

sub.to_csv('ans_stage2/ans.csv', index=False)
