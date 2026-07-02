# 余额宝资金流入流出预测竞赛

## 项目简介

本项目是参加"资金流入流出预测"竞赛的代码仓库，目标是预测2014年9月每天的申购总额和赎回总额。

## 赛题说明

- **预测目标**：2014年9月1日-30日，共30天的每日申购总额(purchase)和赎回总额(redeem)
- **单位**：分（精确到分）
- **提交格式**：CSV文件，无表头，三列：report_date(日期数字), purchase(申购), redeem(赎回)
- **评分规则**：
  - 每天误差用相对误差计算
  - 误差=0时得10分/天，误差>0.3时得0分，单调递减
  - 总积分 = 申购预测得分 * 45% + 赎回预测得分 * 55%
  - 满分300分

## 项目结构

```
.
├── README.md                    # 项目说明文档
├── .gitignore                   # Git忽略文件
├── requirements.txt             # Python依赖
├── baseline_model.py            # 初始Baseline版本（81特征）
├── baseline_model_optimized.py  # 优化版（172特征，对数变换）
├── baseline_model_robust.py     # 稳健版（81特征，强正则化）
└── Purchase Redemption Data/    # 数据目录（不提交到Git）
    ├── user_profile_table.csv
    ├── user_balance_table.csv
    ├── mfd_day_share_interest.csv
    ├── mfd_bank_shibor.csv
    └── comp_predict_table.csv
```

## 数据说明

所有数据文件位于 `Purchase Redemption Data/` 目录下：

1. **user_profile_table.csv** - 用户画像表（28041用户）
2. **user_balance_table.csv** - 用户申购赎回数据表（284万行，核心数据）
3. **mfd_day_share_interest.csv** - 收益率表（427天）
4. **mfd_bank_shibor.csv** - Shibor利率表（294天）
5. **comp_predict_table.csv** - 提交样例

数据时间范围：2013-07-01 ~ 2014-08-31

## 模型版本说明

### Baseline版 (baseline_model.py)
- 81个特征
- 原始目标（直接预测金额）
- LightGBM模型
- 实际得分：69.2433分

### 优化版 (baseline_model_optimized.py)
- 172个特征
- 对数目标变换（log1p/expm1）
- 增加节假日特征、发薪日、增长率等
- 实际得分：68.6684分（特征过多导致过拟合）

### 稳健版 (baseline_model_robust.py)
- 81个核心特征
- 强正则化（learning_rate=0.01, max_depth=5, num_leaves=15）
- 直接优化MAPE目标
- 预测值平滑约束（防止滚动预测误差累积）
- 保留节假日特征

## 特征工程

### 核心有效特征
1. **时序滞后特征**：lag1-7, lag14, lag28
2. **滚动统计特征**：7天、14天、28天的均值、标准差、最大值、最小值
3. **日历特征**：星期几、是否周末、月初月末、发薪日
4. **节假日特征**：是否节假日、节假日前一天后一天
5. **利率特征**：万份收益、七日年化、Shibor各期限利率
6. **用户数特征**：活跃用户数及滞后滚动

## 验证策略

- **训练集**：2013-07-01 ~ 2014-06-30
- **验证集**：2014-07-01 ~ 2014-08-31（共62天）
- **预测集**：2014-09-01 ~ 2014-09-30（共30天）
- 严格时序划分，禁止随机划分，防止数据泄露

## 使用方法

### 环境要求
- Python 3.8+
- lightgbm
- pandas
- numpy
- scipy

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行模型
```bash
python baseline_model_robust.py
```

### 输出文件
运行后会在 `Purchase Redemption Data/` 目录下生成 `tc_comp_predict_table.csv` 提交文件。

## 2014年9月重要节假日

- 中秋节：9月6日、9月7日、9月8日（三天假期）
- 9月28日：周日调休上班

## 后续优化方向

1. 多模型融合（LightGBM + XGBoost + 线性回归）
2. 特征选择（基于重要性筛选Top特征）
3. 业务规则后处理（对节假日、月末等特殊日期做规则修正）
4. 针对赎回（权重55%）做专门优化

## 注意事项

1. 提交文件必须30行，不能有表头
2. 金额单位是分，不是元
3. 预测值不能为负数，必须是整数
4. 评分用相对误差，关注MAPE而不是MAE
5. 赎回权重更高（55%），优化赎回的收益更大
6. 数据文件较大（150MB+），不提交到Git仓库
