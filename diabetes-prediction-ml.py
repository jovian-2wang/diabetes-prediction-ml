# 如果没有安装seaborn，取消注释以下行来安装
# !pip install seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 尝试导入seaborn，如果失败则安装
try:
    import seaborn as sns
except ImportError:
    # 如果没有seaborn，则安装
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns

# 其余代码不变...
# 直接从网络加载数据，无需下载
import pandas as pd
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 1. 加载数据
print("=== 糖尿病预测项目 ===")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age',
                'Outcome']
df = pd.read_csv(url, names=column_names)

print(f"数据集形状: {df.shape}")
print(f"患糖尿病比例: {df['Outcome'].mean():.2%}")

# 2. 数据预处理
# 处理异常值（医疗数据中0值可能表示缺失）
columns_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_to_clean:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# 3. 探索性数据分析 - 生成图表1：相关性热力图
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('特征相关性热力图')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 生成图表2：特征分布箱线图
plt.figure(figsize=(12, 8))
features_to_plot = ['Glucose', 'BloodPressure', 'BMI', 'Age']
df_melted = df.melt(id_vars=['Outcome'], value_vars=features_to_plot)

plt.subplot(2, 2, 1)
sns.boxplot(x='Outcome', y='Glucose', data=df)
plt.title('血糖水平分布')

plt.subplot(2, 2, 2)
sns.boxplot(x='Outcome', y='BloodPressure', data=df)
plt.title('血压分布')

plt.subplot(2, 2, 3)
sns.boxplot(x='Outcome', y='BMI', data=df)
plt.title('BMI分布')

plt.subplot(2, 2, 4)
sns.boxplot(x='Outcome', y='Age', data=df)
plt.title('年龄分布')

plt.tight_layout()
plt.savefig('feature_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 特征工程
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 6. 模型训练与比较
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True)
}

results = {}
print("\n=== 模型性能比较 ===")
for name, model in models.items():
    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    # 训练模型
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)

    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'accuracy': accuracy,
        'auc': auc_score,
        'model': model
    }

    print(f"{name}:")
    print(f"  交叉验证准确率: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    print(f"  测试集准确率: {accuracy:.3f}")
    print(f"  AUC Score: {auc_score:.3f}")

# 7. 生成图表3：特征重要性（随机森林）
best_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': column_names[:-1],
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('特征重要性')
plt.title('随机森林特征重要性排序')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 生成图表4：ROC曲线
plt.figure(figsize=(8, 6))
for name, result in results.items():
    model = result['model']
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('ROC曲线比较')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. 生成图表5：混淆矩阵
best_rf_model = results['Random Forest']['model']
y_pred = best_rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('随机森林混淆矩阵')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. 最终结果总结
print("\n=== 项目总结 ===")
best_result = results['Random Forest']
print(f"最佳模型: Random Forest")
print(f"测试集准确率: {best_result['accuracy']:.3f}")
print(f"AUC Score: {best_result['auc']:.3f}")
print(f"最重要的特征: {feature_importance.iloc[-1]['feature']}")

print("\n所有图表已保存为PNG文件，可直接用于作品集！")
