# -
资金流入流出预测
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ===================== 1. 读取所有数据文件 =====================
print("="*50)
print("步骤1: 读取数据文件")
print("="*50)

data_path = r"C:\Users\何红儒\Downloads\Purchase Redemption Data\Purchase Redemption Data"

df_user = pd.read_csv(f"{data_path}/user_profile_table.csv")
df_balance = pd.read_csv(f"{data_path}/user_balance_table.csv")
df_yield = pd.read_csv(f"{data_path}/mfd_day_share_interest.csv")
df_shibor = pd.read_csv(f"{data_path}/mfd_bank_shibor.csv")
df_submit_sample = pd.read_csv(f"{data_path}/comp_predict_table.csv")

print(f"用户画像表: {df_user.shape}")
print(f"用户申购赎回表: {df_balance.shape}")
print(f"收益率表: {df_yield.shape}")
print(f"Shibor利率表: {df_shibor.shape}")
print(f"提交样例表: {df_submit_sample.shape}")

# 日期格式转换
def trans_date(s):
    return pd.to_datetime(str(s), format="%Y%m%d")

df_balance["report_date"] = df_balance["report_date"].apply(trans_date)
df_yield["mfd_date"] = df_yield["mfd_date"].apply(trans_date)
df_shibor["mfd_date"] = df_shibor["mfd_date"].apply(trans_date)

print("\n数据时间范围:")
print(f"申购赎回数据: {df_balance['report_date'].min()} ~ {df_balance['report_date'].max()}")
print(f"收益率数据: {df_yield['mfd_date'].min()} ~ {df_yield['mfd_date'].max()}")
print(f"Shibor数据: {df_shibor['mfd_date'].min()} ~ {df_shibor['mfd_date'].max()}")

# ===================== 2. 聚合每日总申购、总赎回 =====================
print("\n" + "="*50)
print("步骤2: 聚合每日总申购赎回标签")
print("="*50)

df_daily = df_balance.groupby("report_date").agg(
    y_purchase=("total_purchase_amt", "sum"),
    y_redeem=("total_redeem_amt", "sum"),
    user_count=("user_id", "nunique"),
    total_consume=("consume_amt", "sum"),
    total_transfer=("transfer_amt", "sum")
).reset_index()

print(f"聚合后每日数据: {df_daily.shape}")
print(f"日期范围: {df_daily['report_date'].min()} ~ {df_daily['report_date'].max()}")
print(f"\n申购统计 (单位: 分):")
print(f"  均值: {df_daily['y_purchase'].mean():,.0f}")
print(f"  最大值: {df_daily['y_purchase'].max():,.0f}")
print(f"  最小值: {df_daily['y_purchase'].min():,.0f}")
print(f"\n赎回统计 (单位: 分):")
print(f"  均值: {df_daily['y_redeem'].mean():,.0f}")
print(f"  最大值: {df_daily['y_redeem'].max():,.0f}")
print(f"  最小值: {df_daily['y_redeem'].min():,.0f}")

# ===================== 3. 合并外部特征数据 =====================
print("\n" + "="*50)
print("步骤3: 合并收益率和Shibor数据")
print("="*50)

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

print(f"合并后数据形状: {df_merge.shape}")
print(f"缺失值检查:")
print(df_merge.isnull().sum())

# ===================== 4. 特征工程 =====================
print("\n" + "="*50)
print("步骤4: 特征工程")
print("="*50)

def build_lag_roll_feat(df, target_col, lag_list, roll_win_list):
    for lag in lag_list:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    for win in roll_win_list:
        df[f"{target_col}_roll{win}_mean"] = df[target_col].rolling(window=win).mean()
        df[f"{target_col}_roll{win}_std"] = df[target_col].rolling(window=win).std()
        df[f"{target_col}_roll{win}_max"] = df[target_col].rolling(window=win).max()
        df[f"{target_col}_roll{win}_min"] = df[target_col].rolling(window=win).min()
    return df

# 申购滞后和滚动特征
df_merge = build_lag_roll_feat(df_merge, "y_purchase", 
                               [1,2,3,4,5,6,7,14,30], 
                               [7,14,30])

# 赎回滞后和滚动特征
df_merge = build_lag_roll_feat(df_merge, "y_redeem", 
                               [1,2,3,4,5,6,7,14,30], 
                               [7,14,30])

# 用户数滞后特征
df_merge = build_lag_roll_feat(df_merge, "user_count", 
                               [1,2,3,7], 
                               [7,14])

# 日历特征
df_merge["weekday"] = df_merge["report_date"].dt.weekday
df_merge["month"] = df_merge["report_date"].dt.month
df_merge["day"] = df_merge["report_date"].dt.day
df_merge["dayofyear"] = df_merge["report_date"].dt.dayofyear
df_merge["is_weekend"] = (df_merge["weekday"] >= 5).astype(int)
df_merge["is_month_start"] = df_merge["report_date"].dt.is_month_start.astype(int)
df_merge["is_month_end"] = df_merge["report_date"].dt.is_month_end.astype(int)
df_merge["week_of_month"] = (df_merge["day"] - 1) // 7 + 1

# 月初月末附近特征
df_merge["near_month_start"] = (df_merge["day"] <= 5).astype(int)
df_merge["near_month_end"] = (df_merge["day"] >= 25).astype(int)

# 利率衍生特征
df_merge["yield_change_1d"] = df_merge["mfd_daily_yield"].diff(1)
df_merge["yield_change_7d"] = df_merge["mfd_daily_yield"].diff(7)
df_merge["shibor_on_change_1d"] = df_merge["Interest_O_N"].diff(1)
df_merge["shibor_1m_change_7d"] = df_merge["Interest_1_M"].diff(7)

# 申购赎回比例特征（滞后）
df_merge["purchase_redeem_ratio_lag1"] = df_merge["y_purchase_lag1"] / (df_merge["y_redeem_lag1"] + 1)

# 同比特征（去年同月同日，这里数据只有14个月，用30天前近似）
df_merge["purchase_yoy_lag30"] = df_merge["y_purchase_lag30"]
df_merge["redeem_yoy_lag30"] = df_merge["y_redeem_lag30"]

print(f"特征工程后数据形状: {df_merge.shape}")
print(f"特征总数: {len(df_merge.columns) - 2}")  # 减去日期和两个标签

# 剔除空值（滚动特征前N行缺失）
df_merge_clean = df_merge.dropna().reset_index(drop=True)
print(f"剔除缺失值后数据形状: {df_merge_clean.shape}")

# ===================== 5. 时序划分训练验证集 =====================
print("\n" + "="*50)
print("步骤5: 时序划分数据集")
print("="*50)

split_train_end = pd.to_datetime("2014-06-30")
split_val_end = pd.to_datetime("2014-08-31")

train = df_merge_clean[df_merge_clean["report_date"] <= split_train_end].copy()
val = df_merge_clean[(df_merge_clean["report_date"] > split_train_end) & 
                     (df_merge_clean["report_date"] <= split_val_end)].copy()

print(f"训练集: {train['report_date'].min()} ~ {train['report_date'].max()}, 共 {len(train)} 天")
print(f"验证集: {val['report_date'].min()} ~ {val['report_date'].max()}, 共 {len(val)} 天")

# 特征列
exclude_cols = ["report_date", "y_purchase", "y_redeem", "user_count", 
                "total_consume", "total_transfer"]
feat_cols = [col for col in df_merge_clean.columns if col not in exclude_cols]
print(f"\n使用特征数: {len(feat_cols)}")

# 申购模型数据集
X_train_p, y_train_p = train[feat_cols], train["y_purchase"]
X_val_p, y_val_p = val[feat_cols], val["y_purchase"]

# 赎回模型数据集
X_train_r, y_train_r = train[feat_cols], train["y_redeem"]
X_val_r, y_val_r = val[feat_cols], val["y_redeem"]

# ===================== 6. 训练LightGBM双模型 =====================
print("\n" + "="*50)
print("步骤6: 训练LightGBM双模型")
print("="*50)

def train_lgb(X_tr, y_tr, X_val, y_val, target_name):
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    params = {
        "objective": "regression_l2",
        "metric": "mae",
        "learning_rate": 0.03,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1
    }
    
    model = lgb.train(
        params, 
        lgb_train, 
        num_boost_round=2000,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(stopping_rounds=100), 
                   lgb.log_evaluation(period=200)]
    )
    
    # 验证集预测
    y_pred = model.predict(X_val)
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
    
    return model, importance

model_purchase, imp_purchase = train_lgb(X_train_p, y_train_p, X_val_p, y_val_p, "申购")
model_redeem, imp_redeem = train_lgb(X_train_r, y_train_r, X_val_r, y_val_r, "赎回")

# ===================== 7. 9月滚动预测 =====================
print("\n" + "="*50)
print("步骤7: 2014年9月逐日滚动预测")
print("="*50)

pred_start = pd.to_datetime("2014-09-01")
pred_end = pd.to_datetime("2014-09-30")
pred_dates = pd.date_range(pred_start, pred_end, freq="D")

# 用全部训练数据重新训练模型（使用到8月31日的所有数据）
print("使用全部历史数据重新训练模型...")

# 准备全量训练数据
train_all = df_merge_clean.copy()
X_all_p, y_all_p = train_all[feat_cols], train_all["y_purchase"]
X_all_r, y_all_r = train_all[feat_cols], train_all["y_redeem"]

def train_lgb_final(X_tr, y_tr):
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    params = {
        "objective": "regression_l2",
        "metric": "mae",
        "learning_rate": 0.03,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1
    }
    model = lgb.train(params, lgb_train, num_boost_round=500)
    return model

model_purchase_final = train_lgb_final(X_all_p, y_all_p)
model_redeem_final = train_lgb_final(X_all_r, y_all_r)

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
    row_feat["is_weekend"] = 1 if dt.weekday() >= 5 else 0
    row_feat["is_month_start"] = 1 if dt.is_month_start else 0
    row_feat["is_month_end"] = 1 if dt.is_month_end else 0
    row_feat["week_of_month"] = (dt.day - 1) // 7 + 1
    row_feat["near_month_start"] = 1 if dt.day <= 5 else 0
    row_feat["near_month_end"] = 1 if dt.day >= 25 else 0
    
    # 填充收益率、Shibor（用最后已知值前向填充）
    last_yield = df_yield.iloc[-1]
    last_shibor = df_shibor.iloc[-1]
    
    row_feat["mfd_daily_yield"] = last_yield["mfd_daily_yield"]
    row_feat["mfd_7daily_yield"] = last_yield["mfd_7daily_yield"]
    
    for c in ["Interest_O_N","Interest_1_W","Interest_2_W","Interest_1_M",
              "Interest_3_M","Interest_6_M","Interest_9_M","Interest_1_Y"]:
        row_feat[c] = last_shibor[c]
    
    # 利率变化特征（假设9月利率保持不变）
    row_feat["yield_change_1d"] = 0
    row_feat["yield_change_7d"] = 0
    row_feat["shibor_on_change_1d"] = 0
    row_feat["shibor_1m_change_7d"] = 0
    
    # 用户数特征（用历史均值近似）
    avg_user_count = history_df["user_count"].tail(30).mean()
    row_feat["user_count"] = avg_user_count
    row_feat["total_consume"] = history_df["total_consume"].tail(30).mean()
    row_feat["total_transfer"] = history_df["total_transfer"].tail(30).mean()
    
    # lag特征从历史数据取
    for lag in [1,2,3,4,5,6,7,14,30]:
        if len(history_df) >= lag:
            row_feat[f"y_purchase_lag{lag}"] = history_df["y_purchase"].iloc[-lag]
            row_feat[f"y_redeem_lag{lag}"] = history_df["y_redeem"].iloc[-lag]
            row_feat[f"user_count_lag{lag}"] = history_df["user_count"].iloc[-lag] if lag <= 7 else np.nan
        else:
            row_feat[f"y_purchase_lag{lag}"] = np.nan
            row_feat[f"y_redeem_lag{lag}"] = np.nan
            row_feat[f"user_count_lag{lag}"] = np.nan
    
    # 滚动统计特征
    for win in [7, 14, 30]:
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
            
            if win <= 14:
                recent_u = history_df["user_count"].tail(win)
                row_feat[f"user_count_roll{win}_mean"] = recent_u.mean()
                row_feat[f"user_count_roll{win}_std"] = recent_u.std()
                row_feat[f"user_count_roll{win}_max"] = recent_u.max()
                row_feat[f"user_count_roll{win}_min"] = recent_u.min()
        else:
            row_feat[f"y_purchase_roll{win}_mean"] = np.nan
            row_feat[f"y_purchase_roll{win}_std"] = np.nan
            row_feat[f"y_purchase_roll{win}_max"] = np.nan
            row_feat[f"y_purchase_roll{win}_min"] = np.nan
            row_feat[f"y_redeem_roll{win}_mean"] = np.nan
            row_feat[f"y_redeem_roll{win}_std"] = np.nan
            row_feat[f"y_redeem_roll{win}_max"] = np.nan
            row_feat[f"y_redeem_roll{win}_min"] = np.nan
            if win <= 14:
                row_feat[f"user_count_roll{win}_mean"] = np.nan
                row_feat[f"user_count_roll{win}_std"] = np.nan
                row_feat[f"user_count_roll{win}_max"] = np.nan
                row_feat[f"user_count_roll{win}_min"] = np.nan
    
    # 申购赎回比例特征
    if "y_purchase_lag1" in row_feat and "y_redeem_lag1" in row_feat:
        row_feat["purchase_redeem_ratio_lag1"] = row_feat["y_purchase_lag1"] / (row_feat["y_redeem_lag1"] + 1)
    
    # 同比特征
    if "y_purchase_lag30" in row_feat:
        row_feat["purchase_yoy_lag30"] = row_feat["y_purchase_lag30"]
        row_feat["redeem_yoy_lag30"] = row_feat["y_redeem_lag30"]
    
    # 转换为DataFrame并填充缺失值
    tmp_df = pd.DataFrame([row_feat])
    for c in feat_cols:
        if c not in tmp_df.columns:
            tmp_df[c] = np.nan
    
    # 用训练集均值填充缺失值
    for c in feat_cols:
        if pd.isna(tmp_df[c].iloc[0]):
            tmp_df[c] = train_all[c].mean()
    
    # 预测
    pred_p = model_purchase_final.predict(tmp_df[feat_cols])[0]
    pred_r = model_redeem_final.predict(tmp_df[feat_cols])[0]
    
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
print("\n" + "="*50)
print("步骤8: 生成提交文件")
print("="*50)

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

print("\n" + "="*50)
print("Baseline建模完成！")
print("="*50)
