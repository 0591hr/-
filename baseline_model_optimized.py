import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ===================== 1. 读取所有数据文件 =====================
print("="*60)
print("优化版模型 - 步骤1: 读取数据文件")
print("="*60)

data_path = r"C:\Users\何红儒\Downloads\Purchase Redemption Data\Purchase Redemption Data"

df_user = pd.read_csv(f"{data_path}/user_profile_table.csv")
df_balance = pd.read_csv(f"{data_path}/user_balance_table.csv")
df_yield = pd.read_csv(f"{data_path}/mfd_day_share_interest.csv")
df_shibor = pd.read_csv(f"{data_path}/mfd_bank_shibor.csv")

print(f"用户画像表: {df_user.shape}")
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
print("优化版模型 - 步骤2: 聚合每日总申购赎回标签")
print("="*60)

df_daily = df_balance.groupby("report_date").agg(
    y_purchase=("total_purchase_amt", "sum"),
    y_redeem=("total_redeem_amt", "sum"),
    user_count=("user_id", "nunique"),
    total_consume=("consume_amt", "sum"),
    total_transfer=("transfer_amt", "sum"),
    total_direct_purchase=("direct_purchase_amt", "sum"),
    total_share=("share_amt", "sum")
).reset_index()

print(f"聚合后每日数据: {df_daily.shape}")
print(f"日期范围: {df_daily['report_date'].min()} ~ {df_daily['report_date'].max()}")

# ===================== 3. 合并外部特征数据 =====================
print("\n" + "="*60)
print("优化版模型 - 步骤3: 合并收益率和Shibor数据")
print("="*60)

df_merge = pd.merge(df_daily, df_yield, left_on="report_date", right_on="mfd_date", how="left")
df_merge = pd.merge(df_merge, df_shibor, left_on="report_date", right_on="mfd_date", how="left")
df_merge = df_merge.drop(["mfd_date_x","mfd_date_y"], axis=1, errors='ignore')

# 前向填充缺失的利率数据（周末和节假日）
df_merge = df_merge.sort_values("report_date").reset_index(drop=True)
rate_cols = ["Interest_O_N","Interest_1_W","Interest_2_W","Interest_1_M",
             "Interest_3_M","Interest_6_M","Interest_9_M","Interest_1_Y",
             "mfd_daily_yield","mfd_7daily_yield"]
for col in rate_cols:
    df_merge[col] = df_merge[col].ffill().bfill()

# ===================== 4. 增强特征工程 =====================
print("\n" + "="*60)
print("优化版模型 - 步骤4: 增强特征工程")
print("="*60)

def build_lag_roll_feat(df, target_col, lag_list, roll_win_list):
    for lag in lag_list:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    for win in roll_win_list:
        df[f"{target_col}_roll{win}_mean"] = df[target_col].rolling(window=win).mean()
        df[f"{target_col}_roll{win}_std"] = df[target_col].rolling(window=win).std()
        df[f"{target_col}_roll{win}_max"] = df[target_col].rolling(window=win).max()
        df[f"{target_col}_roll{win}_min"] = df[target_col].rolling(window=win).min()
        df[f"{target_col}_roll{win}_median"] = df[target_col].rolling(window=win).median()
    return df

# 申购滞后和滚动特征（增加更多滞后阶数）
df_merge = build_lag_roll_feat(df_merge, "y_purchase", 
                               [1,2,3,4,5,6,7,8,9,10,14,21,30,60], 
                               [7,14,21,30,60])

# 赎回滞后和滚动特征
df_merge = build_lag_roll_feat(df_merge, "y_redeem", 
                               [1,2,3,4,5,6,7,8,9,10,14,21,30,60], 
                               [7,14,21,30,60])

# 用户数滞后特征
df_merge = build_lag_roll_feat(df_merge, "user_count", 
                               [1,2,3,7,14,30], 
                               [7,14,30])

# 收益金额特征
df_merge = build_lag_roll_feat(df_merge, "total_share", 
                               [1,2,3,7], 
                               [7,14,30])

# ========== 日历特征增强 ==========
df_merge["weekday"] = df_merge["report_date"].dt.weekday
df_merge["month"] = df_merge["report_date"].dt.month
df_merge["day"] = df_merge["report_date"].dt.day
df_merge["dayofyear"] = df_merge["report_date"].dt.dayofyear
df_merge["weekofyear"] = df_merge["report_date"].dt.isocalendar().week.astype(int)
df_merge["is_weekend"] = (df_merge["weekday"] >= 5).astype(int)
df_merge["is_month_start"] = df_merge["report_date"].dt.is_month_start.astype(int)
df_merge["is_month_end"] = df_merge["report_date"].dt.is_month_end.astype(int)
df_merge["is_quarter_start"] = df_merge["report_date"].dt.is_quarter_start.astype(int)
df_merge["is_quarter_end"] = df_merge["report_date"].dt.is_quarter_end.astype(int)
df_merge["week_of_month"] = (df_merge["day"] - 1) // 7 + 1

# 月初月末附近特征
df_merge["near_month_start"] = (df_merge["day"] <= 5).astype(int)
df_merge["near_month_end"] = (df_merge["day"] >= 25).astype(int)
df_merge["mid_month"] = ((df_merge["day"] >= 10) & (df_merge["day"] <= 20)).astype(int)

# 发薪日特征（假设每月10号、20号发薪）
df_merge["is_salary_day"] = ((df_merge["day"] == 10) | (df_merge["day"] == 20)).astype(int)
df_merge["after_salary_3d"] = ((df_merge["day"] >= 10) & (df_merge["day"] <= 13) | 
                               (df_merge["day"] >= 20) & (df_merge["day"] <= 23)).astype(int)

# ========== 节假日特征（2013-2014年） ==========
# 2014年节假日
holidays_2014 = [
    # 元旦
    "2014-01-01",
    # 春节
    "2014-01-31", "2014-02-01", "2014-02-02", "2014-02-03", "2014-02-04", "2014-02-05", "2014-02-06",
    # 清明节
    "2014-04-05", "2014-04-06", "2014-04-07",
    # 劳动节
    "2014-05-01", "2014-05-02", "2014-05-03",
    # 端午节
    "2014-05-31", "2014-06-01", "2014-06-02",
    # 中秋节
    "2014-09-06", "2014-09-07", "2014-09-08",
    # 国庆节
    "2014-10-01", "2014-10-02", "2014-10-03", "2014-10-04", "2014-10-05", "2014-10-06", "2014-10-07",
]

# 2013年节假日
holidays_2013 = [
    # 元旦
    "2013-01-01", "2013-01-02", "2013-01-03",
    # 春节
    "2013-02-09", "2013-02-10", "2013-02-11", "2013-02-12", "2013-02-13", "2013-02-14", "2013-02-15",
    # 清明节
    "2013-04-04", "2013-04-05", "2013-04-06",
    # 劳动节
    "2013-04-29", "2013-04-30", "2013-05-01",
    # 端午节
    "2013-06-10", "2013-06-11", "2013-06-12",
    # 中秋节
    "2013-09-19", "2013-09-20", "2013-09-21",
    # 国庆节
    "2013-10-01", "2013-10-02", "2013-10-03", "2013-10-04", "2013-10-05", "2013-10-06", "2013-10-07",
]

# 调休上班的周末
work_weekends = [
    # 2014年
    "2014-01-26", "2014-02-08",  # 春节调休
    "2014-04-06",  # 清明调休（实际是周一，这里可能需要调整）
    "2014-05-04",  # 劳动节调休
    "2014-09-28", "2014-10-11",  # 国庆调休
    # 2013年
    "2013-01-05", "2013-01-06",  # 元旦调休
    "2013-02-16", "2013-02-17",  # 春节调休
    "2013-04-07",  # 清明调休
    "2013-04-27", "2013-04-28",  # 劳动节调休
    "2013-06-08", "2013-06-09",  # 端午调休
    "2013-09-22",  # 中秋调休
    "2013-09-29", "2013-10-12",  # 国庆调休
]

all_holidays = set(pd.to_datetime(holidays_2013 + holidays_2014))
all_work_weekends = set(pd.to_datetime(work_weekends))

def is_holiday(date):
    if date in all_holidays:
        return 1
    # 周末但不是调休上班
    if date.weekday() >= 5 and date not in all_work_weekends:
        return 1
    return 0

def is_workday(date):
    if date in all_work_weekends:
        return 1
    if date.weekday() < 5 and date not in all_holidays:
        return 1
    return 0

df_merge["is_holiday"] = df_merge["report_date"].apply(is_holiday)
df_merge["is_workday"] = df_merge["report_date"].apply(is_workday)

# 节假日前一天、后一天特征
df_merge["holiday_next_1d"] = df_merge["is_holiday"].shift(-1).fillna(0).astype(int)
df_merge["holiday_prev_1d"] = df_merge["is_holiday"].shift(1).fillna(0).astype(int)
df_merge["holiday_next_3d"] = df_merge["is_holiday"].shift(-1).fillna(0).rolling(3, min_periods=1).max().shift(-2).fillna(0).astype(int)

# 中秋节前后特征（9月最重要的节假日）
def is_mid_autumn_near(date):
    # 2014年中秋是9月8日
    if date.year == 2014 and date.month == 9:
        day = date.day
        if 6 <= day <= 8:  # 中秋假期
            return 1
        if day == 5 or day == 9:  # 假期前后一天
            return 2
    # 2013年中秋是9月19日
    if date.year == 2013 and date.month == 9:
        day = date.day
        if 19 <= day <= 21:
            return 1
        if day == 18 or day == 22:
            return 2
    return 0

df_merge["mid_autumn_near"] = df_merge["report_date"].apply(is_mid_autumn_near)

# ========== 利率衍生特征增强 ==========
df_merge["yield_change_1d"] = df_merge["mfd_daily_yield"].diff(1)
df_merge["yield_change_3d"] = df_merge["mfd_daily_yield"].diff(3)
df_merge["yield_change_7d"] = df_merge["mfd_daily_yield"].diff(7)
df_merge["yield_change_14d"] = df_merge["mfd_daily_yield"].diff(14)
df_merge["yield_roll7_mean"] = df_merge["mfd_daily_yield"].rolling(7).mean()
df_merge["yield_roll14_mean"] = df_merge["mfd_daily_yield"].rolling(14).mean()

df_merge["shibor_on_change_1d"] = df_merge["Interest_O_N"].diff(1)
df_merge["shibor_on_change_7d"] = df_merge["Interest_O_N"].diff(7)
df_merge["shibor_1m_change_7d"] = df_merge["Interest_1_M"].diff(7)
df_merge["shibor_3m_change_7d"] = df_merge["Interest_3_M"].diff(7)

# 利率期限利差
df_merge["shibor_term_spread_1m_on"] = df_merge["Interest_1_M"] - df_merge["Interest_O_N"]
df_merge["shibor_term_spread_3m_1m"] = df_merge["Interest_3_M"] - df_merge["Interest_1_M"]

# ========== 比例和增长率特征 ==========
# 申购赎回比例特征（滞后）
df_merge["purchase_redeem_ratio_lag1"] = df_merge["y_purchase_lag1"] / (df_merge["y_redeem_lag1"] + 1)
df_merge["purchase_redeem_ratio_lag7"] = df_merge["y_purchase_lag7"] / (df_merge["y_redeem_lag7"] + 1)

# 环比增长率
df_merge["purchase_growth_rate_lag1"] = (df_merge["y_purchase_lag1"] - df_merge["y_purchase_lag2"]) / (df_merge["y_purchase_lag2"] + 1)
df_merge["redeem_growth_rate_lag1"] = (df_merge["y_redeem_lag1"] - df_merge["y_redeem_lag2"]) / (df_merge["y_redeem_lag2"] + 1)

# 7日环比
df_merge["purchase_growth_rate_7d"] = (df_merge["y_purchase_lag1"] - df_merge["y_purchase_lag7"]) / (df_merge["y_purchase_lag7"] + 1)
df_merge["redeem_growth_rate_7d"] = (df_merge["y_redeem_lag1"] - df_merge["y_redeem_lag7"]) / (df_merge["y_redeem_lag7"] + 1)

# 偏离均值程度（z-score）
df_merge["purchase_zscore_7d"] = (df_merge["y_purchase_lag1"] - df_merge["y_purchase_roll7_mean"]) / (df_merge["y_purchase_roll7_std"] + 1)
df_merge["redeem_zscore_7d"] = (df_merge["y_redeem_lag1"] - df_merge["y_redeem_roll7_mean"]) / (df_merge["y_redeem_roll7_std"] + 1)

# 周内同天的历史均值（捕捉周度周期）
df_merge["weekday_purchase_mean"] = df_merge.groupby("weekday")["y_purchase_lag1"].transform("mean")
df_merge["weekday_redeem_mean"] = df_merge.groupby("weekday")["y_redeem_lag1"].transform("mean")

print(f"特征工程后数据形状: {df_merge.shape}")
print(f"特征总数: {len(df_merge.columns) - 2}")

# 剔除空值（滚动特征前N行缺失）
df_merge_clean = df_merge.dropna().reset_index(drop=True)
print(f"剔除缺失值后数据形状: {df_merge_clean.shape}")

# ===================== 5. 时序划分训练验证集 =====================
print("\n" + "="*60)
print("优化版模型 - 步骤5: 时序划分数据集")
print("="*60)

split_train_end = pd.to_datetime("2014-06-30")
split_val_end = pd.to_datetime("2014-08-31")

train = df_merge_clean[df_merge_clean["report_date"] <= split_train_end].copy()
val = df_merge_clean[(df_merge_clean["report_date"] > split_train_end) & 
                     (df_merge_clean["report_date"] <= split_val_end)].copy()

print(f"训练集: {train['report_date'].min()} ~ {train['report_date'].max()}, 共 {len(train)} 天")
print(f"验证集: {val['report_date'].min()} ~ {val['report_date'].max()}, 共 {len(val)} 天")

# 特征列
exclude_cols = ["report_date", "y_purchase", "y_redeem", "user_count", 
                "total_consume", "total_transfer", "total_direct_purchase", "total_share"]
feat_cols = [col for col in df_merge_clean.columns if col not in exclude_cols]
print(f"\n使用特征数: {len(feat_cols)}")

# 申购模型数据集（对数变换）
X_train_p, y_train_p = train[feat_cols], np.log1p(train["y_purchase"])
X_val_p, y_val_p = val[feat_cols], np.log1p(val["y_purchase"])

# 赎回模型数据集（对数变换）
X_train_r, y_train_r = train[feat_cols], np.log1p(train["y_redeem"])
X_val_r, y_val_r = val[feat_cols], np.log1p(val["y_redeem"])

# ===================== 6. 训练LightGBM双模型（对数目标） =====================
print("\n" + "="*60)
print("优化版模型 - 步骤6: 训练LightGBM双模型（对数目标）")
print("="*60)

def train_lgb_log(X_tr, y_tr, X_val, y_val_orig, target_name):
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    
    # 验证集用原始值计算MAPE
    y_val_log = np.log1p(y_val_orig)
    lgb_valid = lgb.Dataset(X_val, label=y_val_log, reference=lgb_train)
    
    params = {
        "objective": "regression_l2",
        "metric": "mae",
        "learning_rate": 0.02,
        "max_depth": 7,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_samples": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.2,
        "random_state": 42,
        "verbose": -1
    }
    
    model = lgb.train(
        params, 
        lgb_train, 
        num_boost_round=3000,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(stopping_rounds=150), 
                   lgb.log_evaluation(period=300)]
    )
    
    # 验证集预测（还原对数）
    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0)  # 确保非负
    
    mae = np.mean(np.abs(y_val_orig - y_pred))
    mape = np.mean(np.abs((y_val_orig - y_pred) / (y_val_orig + 1))) * 100
    
    print(f"\n{target_name}模型验证结果:")
    print(f"  MAE: {mae:,.0f} 分 ({mae/100:,.2f} 元)")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  最优迭代次数: {model.best_iteration}")
    
    # 特征重要性
    importance = pd.DataFrame({
        "feature": feat_cols,
        "importance": model.feature_importance(importance_type="gain")
    }).sort_values("importance", ascending=False)
    
    print(f"\n{target_name} Top15重要特征:")
    for i, row in importance.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.0f}")
    
    return model, importance, mape

model_purchase, imp_purchase, mape_p = train_lgb_log(
    X_train_p, y_train_p, X_val_p, val["y_purchase"].values, "申购")
model_redeem, imp_redeem, mape_r = train_lgb_log(
    X_train_r, y_train_r, X_val_r, val["y_redeem"].values, "赎回")

# 估算得分（根据评分规则估算）
# 假设误差=0得10分，误差=0.3得0分，线性递减的话
# 得分 = max(0, 10 * (1 - mape/30))
def estimate_score(mape):
    # 简化估算：线性递减
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
print(f"  (申购权重45%, 赎回权重55%)")
print("="*60)

# ===================== 7. 9月滚动预测 =====================
print("\n" + "="*60)
print("优化版模型 - 步骤7: 2014年9月逐日滚动预测")
print("="*60)

pred_start = pd.to_datetime("2014-09-01")
pred_end = pd.to_datetime("2014-09-30")
pred_dates = pd.date_range(pred_start, pred_end, freq="D")

# 用全部训练数据重新训练模型（使用到8月31日的所有数据）
print("使用全部历史数据重新训练模型...")

train_all = df_merge_clean.copy()
X_all_p, y_all_p = train_all[feat_cols], np.log1p(train_all["y_purchase"])
X_all_r, y_all_r = train_all[feat_cols], np.log1p(train_all["y_redeem"])

def train_lgb_final_log(X_tr, y_tr, best_iter):
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    params = {
        "objective": "regression_l2",
        "metric": "mae",
        "learning_rate": 0.02,
        "max_depth": 7,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_samples": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.2,
        "random_state": 42,
        "verbose": -1
    }
    model = lgb.train(params, lgb_train, num_boost_round=int(best_iter * 1.2))
    return model

model_purchase_final = train_lgb_final_log(X_all_p, y_all_p, model_purchase.best_iteration)
model_redeem_final = train_lgb_final_log(X_all_r, y_all_r, model_redeem.best_iteration)

# 初始化历史数据
history_df = df_merge.copy()
history_df = history_df.sort_values("report_date").reset_index(drop=True)

res_list = []

print(f"\n开始预测 2014-09-01 至 2014-09-30 共 {len(pred_dates)} 天...")

for i, dt in enumerate(pred_dates):
    # 构造当日特征
    row_feat = {}
    row_feat["report_date"] = dt
    row_feat["weekday"] = dt.weekday()
    row_feat["month"] = dt.month
    row_feat["day"] = dt.day
    row_feat["dayofyear"] = dt.dayofyear
    row_feat["weekofyear"] = dt.isocalendar()[1]
    row_feat["is_weekend"] = 1 if dt.weekday() >= 5 else 0
    row_feat["is_month_start"] = 1 if dt.is_month_start else 0
    row_feat["is_month_end"] = 1 if dt.is_month_end else 0
    row_feat["is_quarter_start"] = 1 if dt.is_quarter_start else 0
    row_feat["is_quarter_end"] = 1 if dt.is_quarter_end else 0
    row_feat["week_of_month"] = (dt.day - 1) // 7 + 1
    row_feat["near_month_start"] = 1 if dt.day <= 5 else 0
    row_feat["near_month_end"] = 1 if dt.day >= 25 else 0
    row_feat["mid_month"] = 1 if 10 <= dt.day <= 20 else 0
    row_feat["is_salary_day"] = 1 if dt.day == 10 or dt.day == 20 else 0
    row_feat["after_salary_3d"] = 1 if (10 <= dt.day <= 13) or (20 <= dt.day <= 23) else 0
    
    # 节假日特征
    row_feat["is_holiday"] = is_holiday(dt)
    row_feat["is_workday"] = is_workday(dt)
    row_feat["mid_autumn_near"] = is_mid_autumn_near(dt)
    
    # 节假日前一天后一天（简化处理）
    row_feat["holiday_prev_1d"] = 0
    row_feat["holiday_next_1d"] = 0
    row_feat["holiday_next_3d"] = 0
    # 检查未来几天是否有假期
    for d in range(1, 4):
        future_date = dt + pd.Timedelta(days=d)
        if is_holiday(future_date):
            if d == 1:
                row_feat["holiday_next_1d"] = 1
            row_feat["holiday_next_3d"] = 1
    # 检查前一天
    prev_date = dt - pd.Timedelta(days=1)
    if is_holiday(prev_date):
        row_feat["holiday_prev_1d"] = 1
    
    # 填充收益率、Shibor（用最后已知值前向填充）
    last_yield = df_yield.iloc[-1]
    last_shibor = df_shibor.iloc[-1]
    
    row_feat["mfd_daily_yield"] = last_yield["mfd_daily_yield"]
    row_feat["mfd_7daily_yield"] = last_yield["mfd_7daily_yield"]
    
    for c in ["Interest_O_N","Interest_1_W","Interest_2_W","Interest_1_M",
              "Interest_3_M","Interest_6_M","Interest_9_M","Interest_1_Y"]:
        row_feat[c] = last_shibor[c]
    
    # 利率变化（假设不变）
    row_feat["yield_change_1d"] = 0
    row_feat["yield_change_3d"] = 0
    row_feat["yield_change_7d"] = 0
    row_feat["yield_change_14d"] = 0
    row_feat["yield_roll7_mean"] = last_yield["mfd_daily_yield"]
    row_feat["yield_roll14_mean"] = last_yield["mfd_daily_yield"]
    row_feat["shibor_on_change_1d"] = 0
    row_feat["shibor_on_change_7d"] = 0
    row_feat["shibor_1m_change_7d"] = 0
    row_feat["shibor_3m_change_7d"] = 0
    row_feat["shibor_term_spread_1m_on"] = last_shibor["Interest_1_M"] - last_shibor["Interest_O_N"]
    row_feat["shibor_term_spread_3m_1m"] = last_shibor["Interest_3_M"] - last_shibor["Interest_1_M"]
    
    # 用户数和其他统计特征（用历史均值近似）
    avg_user_count = history_df["user_count"].tail(30).mean()
    row_feat["user_count"] = avg_user_count
    row_feat["total_consume"] = history_df["total_consume"].tail(30).mean()
    row_feat["total_transfer"] = history_df["total_transfer"].tail(30).mean()
    row_feat["total_direct_purchase"] = history_df["total_direct_purchase"].tail(30).mean()
    row_feat["total_share"] = history_df["total_share"].tail(30).mean()
    
    # lag特征从历史数据取
    for lag in [1,2,3,4,5,6,7,8,9,10,14,21,30,60]:
        if len(history_df) >= lag:
            row_feat[f"y_purchase_lag{lag}"] = history_df["y_purchase"].iloc[-lag]
            row_feat[f"y_redeem_lag{lag}"] = history_df["y_redeem"].iloc[-lag]
            if lag <= 30:
                row_feat[f"user_count_lag{lag}"] = history_df["user_count"].iloc[-lag] if lag <= 30 else np.nan
            if lag <= 7:
                row_feat[f"total_share_lag{lag}"] = history_df["total_share"].iloc[-lag]
        else:
            row_feat[f"y_purchase_lag{lag}"] = np.nan
            row_feat[f"y_redeem_lag{lag}"] = np.nan
            row_feat[f"user_count_lag{lag}"] = np.nan
            row_feat[f"total_share_lag{lag}"] = np.nan
    
    # 滚动统计特征
    for win in [7, 14, 21, 30, 60]:
        if len(history_df) >= win:
            recent_p = history_df["y_purchase"].tail(win)
            recent_r = history_df["y_redeem"].tail(win)
            row_feat[f"y_purchase_roll{win}_mean"] = recent_p.mean()
            row_feat[f"y_purchase_roll{win}_std"] = recent_p.std()
            row_feat[f"y_purchase_roll{win}_max"] = recent_p.max()
            row_feat[f"y_purchase_roll{win}_min"] = recent_p.min()
            row_feat[f"y_purchase_roll{win}_median"] = recent_p.median()
            row_feat[f"y_redeem_roll{win}_mean"] = recent_r.mean()
            row_feat[f"y_redeem_roll{win}_std"] = recent_r.std()
            row_feat[f"y_redeem_roll{win}_max"] = recent_r.max()
            row_feat[f"y_redeem_roll{win}_min"] = recent_r.min()
            row_feat[f"y_redeem_roll{win}_median"] = recent_r.median()
            
            if win <= 30:
                recent_u = history_df["user_count"].tail(win)
                row_feat[f"user_count_roll{win}_mean"] = recent_u.mean()
                row_feat[f"user_count_roll{win}_std"] = recent_u.std()
                row_feat[f"user_count_roll{win}_max"] = recent_u.max()
                row_feat[f"user_count_roll{win}_min"] = recent_u.min()
                row_feat[f"user_count_roll{win}_median"] = recent_u.median()
            
            if win <= 30:
                recent_s = history_df["total_share"].tail(win)
                row_feat[f"total_share_roll{win}_mean"] = recent_s.mean()
                row_feat[f"total_share_roll{win}_std"] = recent_s.std()
                row_feat[f"total_share_roll{win}_max"] = recent_s.max()
                row_feat[f"total_share_roll{win}_min"] = recent_s.min()
                row_feat[f"total_share_roll{win}_median"] = recent_s.median()
        else:
            for target in ["y_purchase", "y_redeem", "user_count", "total_share"]:
                for stat in ["mean", "std", "max", "min", "median"]:
                    col = f"{target}_roll{win}_{stat}"
                    if col in feat_cols:
                        row_feat[col] = np.nan
    
    # 比例和增长率特征
    if "y_purchase_lag1" in row_feat and "y_redeem_lag1" in row_feat:
        row_feat["purchase_redeem_ratio_lag1"] = row_feat["y_purchase_lag1"] / (row_feat["y_redeem_lag1"] + 1)
    if "y_purchase_lag7" in row_feat and "y_redeem_lag7" in row_feat:
        row_feat["purchase_redeem_ratio_lag7"] = row_feat["y_purchase_lag7"] / (row_feat["y_redeem_lag7"] + 1)
    
    # 环比增长率
    if "y_purchase_lag1" in row_feat and "y_purchase_lag2" in row_feat:
        row_feat["purchase_growth_rate_lag1"] = (row_feat["y_purchase_lag1"] - row_feat["y_purchase_lag2"]) / (row_feat["y_purchase_lag2"] + 1)
    if "y_redeem_lag1" in row_feat and "y_redeem_lag2" in row_feat:
        row_feat["redeem_growth_rate_lag1"] = (row_feat["y_redeem_lag1"] - row_feat["y_redeem_lag2"]) / (row_feat["y_redeem_lag2"] + 1)
    
    # 7日增长率
    if "y_purchase_lag1" in row_feat and "y_purchase_lag7" in row_feat:
        row_feat["purchase_growth_rate_7d"] = (row_feat["y_purchase_lag1"] - row_feat["y_purchase_lag7"]) / (row_feat["y_purchase_lag7"] + 1)
    if "y_redeem_lag1" in row_feat and "y_redeem_lag7" in row_feat:
        row_feat["redeem_growth_rate_7d"] = (row_feat["y_redeem_lag1"] - row_feat["y_redeem_lag7"]) / (row_feat["y_redeem_lag7"] + 1)
    
    # z-score
    if "y_purchase_lag1" in row_feat and "y_purchase_roll7_mean" in row_feat and "y_purchase_roll7_std" in row_feat:
        row_feat["purchase_zscore_7d"] = (row_feat["y_purchase_lag1"] - row_feat["y_purchase_roll7_mean"]) / (row_feat["y_purchase_roll7_std"] + 1)
    if "y_redeem_lag1" in row_feat and "y_redeem_roll7_mean" in row_feat and "y_redeem_roll7_std" in row_feat:
        row_feat["redeem_zscore_7d"] = (row_feat["y_redeem_lag1"] - row_feat["y_redeem_roll7_mean"]) / (row_feat["y_redeem_roll7_std"] + 1)
    
    # 周内同天均值（用历史数据计算）
    weekday_data = history_df[history_df["weekday"] == dt.weekday()]
    if len(weekday_data) > 0:
        row_feat["weekday_purchase_mean"] = weekday_data["y_purchase"].tail(20).mean()
        row_feat["weekday_redeem_mean"] = weekday_data["y_redeem"].tail(20).mean()
    else:
        row_feat["weekday_purchase_mean"] = history_df["y_purchase"].tail(30).mean()
        row_feat["weekday_redeem_mean"] = history_df["y_redeem"].tail(30).mean()
    
    # 转换为DataFrame并填充缺失值
    tmp_df = pd.DataFrame([row_feat])
    for c in feat_cols:
        if c not in tmp_df.columns:
            tmp_df[c] = np.nan
    
    # 用训练集均值填充缺失值
    for c in feat_cols:
        if pd.isna(tmp_df[c].iloc[0]):
            tmp_df[c] = train_all[c].mean()
    
    # 预测（对数还原）
    pred_p_log = model_purchase_final.predict(tmp_df[feat_cols])[0]
    pred_r_log = model_redeem_final.predict(tmp_df[feat_cols])[0]
    pred_p = np.expm1(pred_p_log)
    pred_r = np.expm1(pred_r_log)
    
    # 确保非负且为整数
    pred_p = max(int(round(pred_p)), 0)
    pred_r = max(int(round(pred_r)), 0)
    
    res_list.append({
        "report_date": int(dt.strftime("%Y%m%d")),
        "purchase": pred_p,
        "redeem": pred_r
    })
    
    # 追加预测值到历史，用于下一日lag构造
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
print("优化版模型 - 步骤8: 生成提交文件")
print("="*60)

submit = pd.DataFrame(res_list)
submit = submit[["report_date", "purchase", "redeem"]]  # 确保列顺序正确

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

# 9月特殊日期分析
print(f"\n9月关键日期预测:")
mid_autumn_dates = submit[submit["report_date"].isin([20140906, 20140907, 20140908])]
print(f"  中秋假期(9.6-9.8)日均申购: {mid_autumn_dates['purchase'].mean()/100:,.2f} 元")
print(f"  中秋假期(9.6-9.8)日均赎回: {mid_autumn_dates['redeem'].mean()/100:,.2f} 元")

month_end_dates = submit[submit["report_date"] >= 20140925]
print(f"  月末(9.25-9.30)日均申购: {month_end_dates['purchase'].mean()/100:,.2f} 元")
print(f"  月末(9.25-9.30)日均赎回: {month_end_dates['redeem'].mean()/100:,.2f} 元")

print("\n" + "="*60)
print("优化版Baseline建模完成！")
print("="*60)
