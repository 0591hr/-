import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ===================== 1. 读取所有数据文件 =====================
print("="*60)
print("稳健版模型 - 步骤1: 读取数据文件")
print("="*60)

data_path = r"C:\Users\何红儒\Downloads\Purchase Redemption Data\Purchase Redemption Data"

df_balance = pd.read_csv(f"{data_path}/user_balance_table.csv")
df_yield = pd.read_csv(f"{data_path}/mfd_day_share_interest.csv")
df_shibor = pd.read_csv(f"{data_path}/mfd_bank_shibor.csv")

print(f"用户申购赎回表: {df_balance.shape}")
print(f"收益率表: {df_yield.shape}")
print(f"Shibor利率表: {df_shibor.shape}")

# 日期格式转换
def trans_date(s):
    return pd.to_datetime(str(s), format="%Y%m%d")

df_balance["report_date"] = df_balance["report_date"].apply(trans_date)
df_yield["mfd_date"] = df_yield["mfd_date"].apply(trans_date)
df_shibor["mfd_date"] = df_shibor["mfd_date"].apply(trans_date)

# ===================== 2. 聚合每日总申购、总赎回 =====================
print("\n" + "="*60)
print("稳健版模型 - 步骤2: 聚合每日总申购赎回标签")
print("="*60)

df_daily = df_balance.groupby("report_date").agg(
    y_purchase=("total_purchase_amt", "sum"),
    y_redeem=("total_redeem_amt", "sum"),
    user_count=("user_id", "nunique"),
).reset_index()

print(f"聚合后每日数据: {df_daily.shape}")
print(f"日期范围: {df_daily['report_date'].min()} ~ {df_daily['report_date'].max()}")

# ===================== 3. 合并外部特征数据 =====================
print("\n" + "="*60)
print("稳健版模型 - 步骤3: 合并收益率和Shibor数据")
print("="*60)

df_merge = pd.merge(df_daily, df_yield, left_on="report_date", right_on="mfd_date", how="left")
df_merge = pd.merge(df_merge, df_shibor, left_on="report_date", right_on="mfd_date", how="left")
df_merge = df_merge.drop(["mfd_date_x","mfd_date_y"], axis=1, errors='ignore')

# 前向填充缺失的利率数据
df_merge = df_merge.sort_values("report_date").reset_index(drop=True)
rate_cols = ["Interest_O_N","Interest_1_W","Interest_2_W","Interest_1_M",
             "Interest_3_M","Interest_6_M","Interest_9_M","Interest_1_Y",
             "mfd_daily_yield","mfd_7daily_yield"]
for col in rate_cols:
    df_merge[col] = df_merge[col].ffill().bfill()

# ===================== 4. 精简特征工程（核心特征） =====================
print("\n" + "="*60)
print("稳健版模型 - 步骤4: 精简特征工程（核心特征）")
print("="*60)

def build_lag_roll_feat(df, target_col, lag_list, roll_win_list):
    for lag in lag_list:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    for win in roll_win_list:
        df[f"{target_col}_roll{win}_mean"] = df[target_col].rolling(window=win).mean()
        df[f"{target_col}_roll{win}_std"] = df[target_col].rolling(window=win).std()
        df[f"{target_col}_roll{win}_max"] = df[target_col].rolling(window=win).max()
        df[f"{target_col}_roll{win}_min"] = df[target_col].rolling(window=win).min()
    return df

# 申购核心特征（精简）
df_merge = build_lag_roll_feat(df_merge, "y_purchase", 
                               [1,2,3,4,5,6,7,14,28], 
                               [7,14,28])

# 赎回核心特征
df_merge = build_lag_roll_feat(df_merge, "y_redeem", 
                               [1,2,3,4,5,6,7,14,28], 
                               [7,14,28])

# 用户数特征
df_merge = build_lag_roll_feat(df_merge, "user_count", 
                               [1,7], 
                               [7,14])

# ========== 日历特征 ==========
df_merge["weekday"] = df_merge["report_date"].dt.weekday
df_merge["day"] = df_merge["report_date"].dt.day
df_merge["is_weekend"] = (df_merge["weekday"] >= 5).astype(int)
df_merge["is_month_start"] = df_merge["report_date"].dt.is_month_start.astype(int)
df_merge["is_month_end"] = df_merge["report_date"].dt.is_month_end.astype(int)
df_merge["week_of_month"] = (df_merge["day"] - 1) // 7 + 1

# 月初月末附近
df_merge["near_month_start"] = (df_merge["day"] <= 5).astype(int)
df_merge["near_month_end"] = (df_merge["day"] >= 25).astype(int)

# 发薪日
df_merge["is_salary_day"] = ((df_merge["day"] == 10) | (df_merge["day"] == 20)).astype(int)

# ========== 节假日特征 ==========
holidays_2014 = [
    "2014-01-01",
    "2014-01-31", "2014-02-01", "2014-02-02", "2014-02-03", "2014-02-04", "2014-02-05", "2014-02-06",
    "2014-04-05", "2014-04-06", "2014-04-07",
    "2014-05-01", "2014-05-02", "2014-05-03",
    "2014-05-31", "2014-06-01", "2014-06-02",
    "2014-09-06", "2014-09-07", "2014-09-08",
    "2014-10-01", "2014-10-02", "2014-10-03", "2014-10-04", "2014-10-05", "2014-10-06", "2014-10-07",
]

holidays_2013 = [
    "2013-01-01", "2013-01-02", "2013-01-03",
    "2013-02-09", "2013-02-10", "2013-02-11", "2013-02-12", "2013-02-13", "2013-02-14", "2013-02-15",
    "2013-04-04", "2013-04-05", "2013-04-06",
    "2013-04-29", "2013-04-30", "2013-05-01",
    "2013-06-10", "2013-06-11", "2013-06-12",
    "2013-09-19", "2013-09-20", "2013-09-21",
    "2013-10-01", "2013-10-02", "2013-10-03", "2013-10-04", "2013-10-05", "2013-10-06", "2013-10-07",
]

work_weekends = [
    "2014-01-26", "2014-02-08",
    "2014-05-04",
    "2014-09-28", "2014-10-11",
    "2013-01-05", "2013-01-06",
    "2013-02-16", "2013-02-17",
    "2013-04-07",
    "2013-04-27", "2013-04-28",
    "2013-06-08", "2013-06-09",
    "2013-09-22",
    "2013-09-29", "2013-10-12",
]

all_holidays = set(pd.to_datetime(holidays_2013 + holidays_2014))
all_work_weekends = set(pd.to_datetime(work_weekends))

def is_holiday(date):
    if date in all_holidays:
        return 1
    if date.weekday() >= 5 and date not in all_work_weekends:
        return 1
    return 0

df_merge["is_holiday"] = df_merge["report_date"].apply(is_holiday)

# 节假日前一天、后一天
df_merge["holiday_next_1d"] = df_merge["is_holiday"].shift(-1).fillna(0).astype(int)
df_merge["holiday_prev_1d"] = df_merge["is_holiday"].shift(1).fillna(0).astype(int)

# ========== 利率特征 ==========
df_merge["yield_change_1d"] = df_merge["mfd_daily_yield"].diff(1)
df_merge["yield_change_7d"] = df_merge["mfd_daily_yield"].diff(7)
df_merge["shibor_on_change_1d"] = df_merge["Interest_O_N"].diff(1)
df_merge["shibor_1m_change_7d"] = df_merge["Interest_1_M"].diff(7)

# ========== 比例特征 ==========
df_merge["purchase_redeem_ratio_lag1"] = df_merge["y_purchase_lag1"] / (df_merge["y_redeem_lag1"] + 1)

# 周内同天均值
df_merge["weekday_purchase_mean"] = df_merge.groupby("weekday")["y_purchase_lag1"].transform("mean")
df_merge["weekday_redeem_mean"] = df_merge.groupby("weekday")["y_redeem_lag1"].transform("mean")

print(f"特征工程后数据形状: {df_merge.shape}")
print(f"特征总数: {len(df_merge.columns) - 2}")

# 剔除空值
df_merge_clean = df_merge.dropna().reset_index(drop=True)
print(f"剔除缺失值后数据形状: {df_merge_clean.shape}")

# ===================== 5. 时序划分训练验证集 =====================
print("\n" + "="*60)
print("稳健版模型 - 步骤5: 时序划分数据集")
print("="*60)

split_train_end = pd.to_datetime("2014-06-30")
split_val_end = pd.to_datetime("2014-08-31")

train = df_merge_clean[df_merge_clean["report_date"] <= split_train_end].copy()
val = df_merge_clean[(df_merge_clean["report_date"] > split_train_end) & 
                     (df_merge_clean["report_date"] <= split_val_end)].copy()

print(f"训练集: {train['report_date'].min()} ~ {train['report_date'].max()}, 共 {len(train)} 天")
print(f"验证集: {val['report_date'].min()} ~ {val['report_date'].max()}, 共 {len(val)} 天")

# 特征列
exclude_cols = ["report_date", "y_purchase", "y_redeem", "user_count"]
feat_cols = [col for col in df_merge_clean.columns if col not in exclude_cols]
print(f"\n使用特征数: {len(feat_cols)}")

# 申购模型数据集（不用对数）
X_train_p, y_train_p = train[feat_cols], train["y_purchase"]
X_val_p, y_val_p = val[feat_cols], val["y_purchase"]

# 赎回模型数据集
X_train_r, y_train_r = train[feat_cols], train["y_redeem"]
X_val_r, y_val_r = val[feat_cols], val["y_redeem"]

# ===================== 6. 训练LightGBM双模型（强正则化） =====================
print("\n" + "="*60)
print("稳健版模型 - 步骤6: 训练LightGBM双模型（强正则化）")
print("="*60)

def train_lgb_robust(X_tr, y_tr, X_val, y_val, target_name):
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    # 强正则化参数，防止过拟合
    params = {
        "objective": "regression_l2",
        "metric": "mape",  # 直接用MAPE作为优化目标
        "learning_rate": 0.01,  # 更小的学习率
        "max_depth": 5,  # 更浅的树
        "num_leaves": 15,  # 更少的叶子
        "subsample": 0.9,  # 更高的采样（更保守）
        "colsample_bytree": 0.8,
        "min_child_samples": 10,  # 更多的最小样本数
        "min_gain_to_split": 0.1,
        "reg_alpha": 1.0,  # 更强的L1正则
        "reg_lambda": 2.0,  # 更强的L2正则
        "random_state": 42,
        "verbose": -1
    }
    
    model = lgb.train(
        params, 
        lgb_train, 
        num_boost_round=5000,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(stopping_rounds=200), 
                   lgb.log_evaluation(period=500)]
    )
    
    # 验证集预测
    y_pred = model.predict(X_val)
    y_pred = np.maximum(y_pred, 0)
    
    mae = np.mean(np.abs(y_val - y_pred))
    mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1))) * 100
    
    print(f"\n{target_name}模型验证结果:")
    print(f"  MAE: {mae:,.0f} 分 ({mae/100:,.2f} 元)")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  最优迭代次数: {model.best_iteration}")
    
    # 特征重要性
    importance = pd.DataFrame({
        "feature": feat_cols,
        "importance": model.feature_importance(importance_type="gain")
    }).sort_values("importance", ascending=False)
    
    print(f"\n{target_name} Top10重要特征:")
    for i, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.0f}")
    
    return model, importance, mape

model_purchase, imp_purchase, mape_p = train_lgb_robust(
    X_train_p, y_train_p, X_val_p, y_val_p, "申购")
model_redeem, imp_redeem, mape_r = train_lgb_robust(
    X_train_r, y_train_r, X_val_r, y_val_r, "赎回")

# 估算得分
def estimate_score(mape):
    score_per_day = max(0, 10 * (1 - mape / 30))
    return score_per_day * 30

score_p = estimate_score(mape_p)
score_r = estimate_score(mape_r)
total_score_est = score_p * 0.45 + score_r * 0.55

print(f"\n" + "="*60)
print(f"验证集估算得分:")
print(f"  申购得分: {score_p:.2f} / 300")
print(f"  赎回得分: {score_r:.2f} / 300")
print(f"  总估算得分: {total_score_est:.2f} / 300")
print("="*60)

# ===================== 7. 9月滚动预测（稳定版） =====================
print("\n" + "="*60)
print("稳健版模型 - 步骤7: 2014年9月逐日滚动预测（稳定版）")
print("="*60)

pred_start = pd.to_datetime("2014-09-01")
pred_end = pd.to_datetime("2014-09-30")
pred_dates = pd.date_range(pred_start, pred_end, freq="D")

# 用全部训练数据重新训练
print("使用全部历史数据重新训练模型...")

train_all = df_merge_clean.copy()
X_all_p, y_all_p = train_all[feat_cols], train_all["y_purchase"]
X_all_r, y_all_r = train_all[feat_cols], train_all["y_redeem"]

def train_lgb_final(X_tr, y_tr, best_iter):
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    params = {
        "objective": "regression_l2",
        "metric": "mape",
        "learning_rate": 0.01,
        "max_depth": 5,
        "num_leaves": 15,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_samples": 10,
        "min_gain_to_split": 0.1,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "random_state": 42,
        "verbose": -1
    }
    model = lgb.train(params, lgb_train, num_boost_round=int(best_iter * 1.1))
    return model

model_purchase_final = train_lgb_final(X_all_p, y_all_p, model_purchase.best_iteration)
model_redeem_final = train_lgb_final(X_all_r, y_all_r, model_redeem.best_iteration)

# 初始化历史数据
history_df = df_merge.copy()
history_df = history_df.sort_values("report_date").reset_index(drop=True)

# 计算历史均值，用于平滑预测
hist_purchase_mean = history_df["y_purchase"].tail(30).mean()
hist_redeem_mean = history_df["y_redeem"].tail(30).mean()
hist_purchase_std = history_df["y_purchase"].tail(30).std()
hist_redeem_std = history_df["y_redeem"].tail(30).std()

res_list = []

print(f"\n开始预测 2014-09-01 至 2014-09-30 共 {len(pred_dates)} 天...")

for i, dt in enumerate(pred_dates):
    # 构造当日特征
    row_feat = {}
    row_feat["report_date"] = dt
    row_feat["weekday"] = dt.weekday()
    row_feat["day"] = dt.day
    row_feat["is_weekend"] = 1 if dt.weekday() >= 5 else 0
    row_feat["is_month_start"] = 1 if dt.is_month_start else 0
    row_feat["is_month_end"] = 1 if dt.is_month_end else 0
    row_feat["week_of_month"] = (dt.day - 1) // 7 + 1
    row_feat["near_month_start"] = 1 if dt.day <= 5 else 0
    row_feat["near_month_end"] = 1 if dt.day >= 25 else 0
    row_feat["is_salary_day"] = 1 if dt.day == 10 or dt.day == 20 else 0
    
    # 节假日特征
    row_feat["is_holiday"] = is_holiday(dt)
    
    # 检查前一天后一天是否是假期
    prev_date = dt - pd.Timedelta(days=1)
    next_date = dt + pd.Timedelta(days=1)
    row_feat["holiday_prev_1d"] = 1 if is_holiday(prev_date) else 0
    row_feat["holiday_next_1d"] = 1 if is_holiday(next_date) else 0
    
    # 填充收益率、Shibor
    last_yield = df_yield.iloc[-1]
    last_shibor = df_shibor.iloc[-1]
    
    row_feat["mfd_daily_yield"] = last_yield["mfd_daily_yield"]
    row_feat["mfd_7daily_yield"] = last_yield["mfd_7daily_yield"]
    
    for c in ["Interest_O_N","Interest_1_W","Interest_2_W","Interest_1_M",
              "Interest_3_M","Interest_6_M","Interest_9_M","Interest_1_Y"]:
        row_feat[c] = last_shibor[c]
    
    # 利率变化
    row_feat["yield_change_1d"] = 0
    row_feat["yield_change_7d"] = 0
    row_feat["shibor_on_change_1d"] = 0
    row_feat["shibor_1m_change_7d"] = 0
    
    # 用户数特征
    avg_user_count = history_df["user_count"].tail(30).mean()
    row_feat["user_count"] = avg_user_count
    
    # 用户数滞后和滚动
    for lag in [1, 7]:
        if len(history_df) >= lag:
            row_feat[f"user_count_lag{lag}"] = history_df["user_count"].iloc[-lag]
        else:
            row_feat[f"user_count_lag{lag}"] = avg_user_count
    
    for win in [7, 14]:
        if len(history_df) >= win:
            recent_u = history_df["user_count"].tail(win)
            row_feat[f"user_count_roll{win}_mean"] = recent_u.mean()
            row_feat[f"user_count_roll{win}_std"] = recent_u.std()
            row_feat[f"user_count_roll{win}_max"] = recent_u.max()
            row_feat[f"user_count_roll{win}_min"] = recent_u.min()
        else:
            row_feat[f"user_count_roll{win}_mean"] = avg_user_count
            row_feat[f"user_count_roll{win}_std"] = 0
            row_feat[f"user_count_roll{win}_max"] = avg_user_count
            row_feat[f"user_count_roll{win}_min"] = avg_user_count
    
    # 申购滞后和滚动特征
    for lag in [1,2,3,4,5,6,7,14,28]:
        if len(history_df) >= lag:
            row_feat[f"y_purchase_lag{lag}"] = history_df["y_purchase"].iloc[-lag]
            row_feat[f"y_redeem_lag{lag}"] = history_df["y_redeem"].iloc[-lag]
        else:
            row_feat[f"y_purchase_lag{lag}"] = hist_purchase_mean
            row_feat[f"y_redeem_lag{lag}"] = hist_redeem_mean
    
    # 滚动统计特征
    for win in [7, 14, 28]:
        if len(history_df) >= win:
            recent_p = history_df["y_purchase"].tail(win)
            recent_r = history_df["y_redeem"].tail(win)
            row_feat[f"y_purchase_roll{win}_mean"] = recent_p.mean()
            row_feat[f"y_purchase_roll{win}_std"] = recent_p.std()
            row_feat[f"y_purchase_roll{win}_max"] = recent_p.max()
            row_feat[f"y_purchase_roll{win}_min"] = recent_p.min()
            row_feat[f"y_redeem_roll{win}_mean"] = recent_r.mean()
            row_feat[f"y_redeem_roll{win}_std"] = recent_r.std()
            row_feat[f"y_redeem_roll{win}_max"] = recent_r.max()
            row_feat[f"y_redeem_roll{win}_min"] = recent_r.min()
        else:
            row_feat[f"y_purchase_roll{win}_mean"] = hist_purchase_mean
            row_feat[f"y_purchase_roll{win}_std"] = hist_purchase_std
            row_feat[f"y_purchase_roll{win}_max"] = hist_purchase_mean + hist_purchase_std
            row_feat[f"y_purchase_roll{win}_min"] = hist_purchase_mean - hist_purchase_std
            row_feat[f"y_redeem_roll{win}_mean"] = hist_redeem_mean
            row_feat[f"y_redeem_roll{win}_std"] = hist_redeem_std
            row_feat[f"y_redeem_roll{win}_max"] = hist_redeem_mean + hist_redeem_std
            row_feat[f"y_redeem_roll{win}_min"] = hist_redeem_mean - hist_redeem_std
    
    # 比例特征
    row_feat["purchase_redeem_ratio_lag1"] = row_feat["y_purchase_lag1"] / (row_feat["y_redeem_lag1"] + 1)
    
    # 周内同天均值
    weekday_data = history_df[history_df["weekday"] == dt.weekday()]
    if len(weekday_data) > 0:
        row_feat["weekday_purchase_mean"] = weekday_data["y_purchase"].tail(20).mean()
        row_feat["weekday_redeem_mean"] = weekday_data["y_redeem"].tail(20).mean()
    else:
        row_feat["weekday_purchase_mean"] = hist_purchase_mean
        row_feat["weekday_redeem_mean"] = hist_redeem_mean
    
    # 转换为DataFrame
    tmp_df = pd.DataFrame([row_feat])
    for c in feat_cols:
        if c not in tmp_df.columns:
            tmp_df[c] = train_all[c].mean()
    
    # 预测
    pred_p = model_purchase_final.predict(tmp_df[feat_cols])[0]
    pred_r = model_redeem_final.predict(tmp_df[feat_cols])[0]
    
    # 平滑约束：预测值不能偏离历史均值太多（防止异常值）
    # 限制在历史均值的0.3倍到3倍之间
    pred_p = max(pred_p, hist_purchase_mean * 0.3)
    pred_p = min(pred_p, hist_purchase_mean * 3.0)
    pred_r = max(pred_r, hist_redeem_mean * 0.3)
    pred_r = min(pred_r, hist_redeem_mean * 3.0)
    
    # 确保非负且为整数
    pred_p = max(int(round(pred_p)), 0)
    pred_r = max(int(round(pred_r)), 0)
    
    res_list.append({
        "report_date": int(dt.strftime("%Y%m%d")),
        "purchase": pred_p,
        "redeem": pred_r
    })
    
    # 追加预测值到历史
    new_row = row_feat.copy()
    new_row["y_purchase"] = pred_p
    new_row["y_redeem"] = pred_r
    new_row["user_count"] = avg_user_count
    history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
    
    if (i + 1) % 10 == 0:
        print(f"  已完成 {i+1}/{len(pred_dates)} 天预测")

print(f"\n预测完成！共 {len(res_list)} 天数据")

# ===================== 8. 输出提交文件 =====================
print("\n" + "="*60)
print("稳健版模型 - 步骤8: 生成提交文件")
print("="*60)

submit = pd.DataFrame(res_list)
submit = submit[["report_date", "purchase", "redeem"]]

output_path = f"{data_path}/tc_comp_predict_table.csv"
submit.to_csv(output_path, header=False, index=False)

print(f"提交文件已保存至: {output_path}")
print(f"\n预测结果预览 (前10天):")
print(submit.head(10).to_string(index=False))

print(f"\n预测结果统计:")
print(f"申购总额: {submit['purchase'].sum():,.0f} 分 ({submit['purchase'].sum()/100:,.2f} 元)")
print(f"赎回总额: {submit['redeem'].sum():,.0f} 分 ({submit['redeem'].sum()/100:,.2f} 元)")
print(f"日均申购: {submit['purchase'].mean():,.0f} 分 ({submit['purchase'].mean()/100:,.2f} 元)")
print(f"日均赎回: {submit['redeem'].mean():,.0f} 分 ({submit['redeem'].mean()/100:,.2f} 元)")
print(f"申购最小值: {submit['purchase'].min():,.0f} 分 ({submit['purchase'].min()/100:,.2f} 元)")
print(f"申购最大值: {submit['purchase'].max():,.0f} 分 ({submit['purchase'].max()/100:,.2f} 元)")
print(f"赎回最小值: {submit['redeem'].min():,.0f} 分 ({submit['redeem'].min()/100:,.2f} 元)")
print(f"赎回最大值: {submit['redeem'].max():,.0f} 分 ({submit['redeem'].max()/100:,.2f} 元)")

# 中秋假期
print(f"\n中秋假期(9.6-9.8)预测:")
mid_autumn = submit[submit["report_date"].isin([20140906, 20140907, 20140908])]
print(mid_autumn.to_string(index=False))

print("\n" + "="*60)
print("稳健版Baseline建模完成！")
print("="*60)
