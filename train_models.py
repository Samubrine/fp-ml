#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

sns.set_style('darkgrid')
print('Libraries loaded.')

# ======== 1. Load & Prepare ========
train = pd.read_csv('avgo_train.csv', index_col=0)
val = pd.read_csv('avgo_val.csv', index_col=0)
test = pd.read_csv('avgo_test.csv', index_col=0)
df = pd.concat([train, val, test])
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df = df.dropna(subset=['target'])
df['target'] = df['target'].astype(int)

print(f'Data: {df.shape}, Target balance: {df["target"].value_counts(normalize=True).mul(100).round(1).to_dict()}')

exclude = ['open', 'high', 'low', 'close', 'volume', 'target']
feature_cols = [c for c in df.columns if c not in exclude]
X = df[feature_cols].values
y = df['target'].values

n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)
X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]
print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')

results = []

def evaluate(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    
    results.append({'Model': name, 'Train Acc': f'{train_acc:.4f}', 'Val Acc': f'{val_acc:.4f}',
                    'Test Acc': f'{acc:.4f}', 'F1': f'{f1:.4f}', 'Precision': f'{prec:.4f}', 'Recall': f'{rec:.4f}'})
    
    print(f'\n=== {name} ===')
    print(f'Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {acc:.4f}')
    print(f'F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}')
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['DOWN','UP'], yticklabels=['DOWN','UP'])
    plt.title(f'{name} Confusion Matrix')
    plt.tight_layout()
    fname = f'cm_{name.lower().split("(")[0].strip().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=100)
    print(f'  Saved: {fname}')

# ======== 2. KNN ========
print('\n--------- KNN TUNING ---------')
best_k, best_val = 1, 0
for k in [3, 5, 7, 9, 11, 15, 21, 31]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    va = accuracy_score(y_val, knn.predict(X_val))
    print(f'  k={k:2d} -> Val: {va:.4f}')
    if va > best_val:
        best_val, best_k = va, k
print(f'Best: k={best_k}')

knn = KNeighborsClassifier(n_neighbors=best_k)
evaluate(f'KNN (k={best_k})', knn)

# ======== 3. Naive Bayes ========
print('\n--------- NAIVE BAYES ---------')
evaluate('Gaussian Naive Bayes', GaussianNB())

# ======== 4. SVM ========
print('\n--------- SVM TUNING ---------')
best_model, best_name, best_val = None, '', 0
for kernel in ['linear', 'rbf']:
    for C in [0.1, 1, 10]:
        svm = SVC(kernel=kernel, C=C, random_state=42)
        svm.fit(X_train, y_train)
        va = accuracy_score(y_val, svm.predict(X_val))
        print(f'  kernel={kernel:6s} C={C:4} -> Val: {va:.4f}')
        if va > best_val:
            best_val, best_model, best_name = va, svm, f'SVM ({kernel}, C={C})'
print(f'Best: {best_name}')
evaluate(best_name, best_model)

# ======== 5. Comparison ========
print('\n' + '='*70)
print('MODEL COMPARISON')
print('='*70)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print('='*70)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i, metric in enumerate(['Test Acc', 'F1', 'Precision']):
    ax = axes[i]
    vals = [float(r[metric]) for r in results]
    names = [r['Model'][:20] for r in results]
    bars = ax.bar(names, vals, color=['#2ecc71','#3498db','#e74c3c'])
    ax.set_title(metric, fontweight='bold')
    ax.set_ylim(0.4, 0.7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=120)
print('Saved: model_comparison.png')

print('\n' + '='*60)
print('FINAL SUMMARY')
print('='*60)
best = max(results, key=lambda r: float(r['F1']))
print(f'Best: {best["Model"]} (F1={best["F1"]}, Acc={best["Test Acc"]})')
for r in results:
    print(f'  {r["Model"]:30s} Test Acc: {r["Test Acc"]}  F1: {r["F1"]}')
