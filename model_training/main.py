

# ─────────────────────────────────────────────
# Import
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

# ═══════════════════════════════════════════════════════════════
# 1. LOAD DỮ LIỆU
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("1. LOAD DỮ LIỆU")
print("=" * 60)

DATA_PATH = "data/raw/bank_churn_dataset.csv"   
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Phân phối nhãn (exit):\n{df['exit'].value_counts()}\n")

# ═══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING & TIỀN XỬ LÝ
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("2. FEATURE ENGINEERING & TIỀN XỬ LÝ")
print("=" * 60)

DROP_COLS = ["id", "full_name", "address", "last_active_date", "created_date"]

TARGET = "exit"


meta_df   = df[["id", "full_name"]].copy()
actual_df = df[TARGET].astype(int).rename("actual_churn")

X = df.drop(columns=DROP_COLS + [TARGET])
y = df[TARGET].astype(int)   

cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
print(f"Cột categorical cần encode : {cat_cols}")

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"Số features sau khi xử lý : {X.shape[1]}\n")

# ═══════════════════════════════════════════════════════════════
# 3. CHIA DỮ LIỆU & LƯU TỪNG TẬP RA CSV
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("3. CHIA DỮ LIỆU TRAIN / VALIDATION / TEST (6-2-2)")
print("=" * 60)

RANDOM_STATE = 42

X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
    X, y, meta_df.index,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X_temp, y_temp, idx_temp,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

print(f"Train      : {X_train.shape[0]:>6} mẫu  ({X_train.shape[0] / len(df) * 100:.0f} %)")
print(f"Validation : {X_val.shape[0]:>6} mẫu  ({X_val.shape[0]   / len(df) * 100:.0f} %)")
print(f"Test       : {X_test.shape[0]:>6} mẫu  ({X_test.shape[0]  / len(df) * 100:.0f} %)\n")

meta_train   = meta_df.loc[idx_train].reset_index(drop=True)
meta_val     = meta_df.loc[idx_val].reset_index(drop=True)
meta_test    = meta_df.loc[idx_test].reset_index(drop=True)

actual_train = actual_df.loc[idx_train].reset_index(drop=True)
actual_val   = actual_df.loc[idx_val].reset_index(drop=True)
actual_test  = actual_df.loc[idx_test].reset_index(drop=True)

print("Lưu các tập dữ liệu ra file CSV...")

train_csv = pd.concat(
    [meta_train, X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
)
val_csv = pd.concat(
    [meta_val, X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1
)
test_csv = pd.concat(
    [meta_test, X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
)

train_csv.to_csv("data_train.csv",      index=False)
val_csv.to_csv("data_validation.csv",   index=False)
test_csv.to_csv("data_test.csv",        index=False)

print(f"  data_train.csv      → {len(train_csv):,} dòng")
print(f"  data_validation.csv → {len(val_csv):,} dòng")
print(f"  data_test.csv       → {len(test_csv):,} dòng\n")

# ═══════════════════════════════════════════════════════════════
# 4. SCALE FEATURES
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("4. SCALE FEATURES (StandardScaler)")
print("=" * 60)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)

print("   Fit scaler trên tập Train, transform Train & Validation\n")

# ═══════════════════════════════════════════════════════════════
# 5. TRAIN MÔ HÌNH
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("5. TRAIN MÔ HÌNH (chỉ dùng Train + Validation)")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=50,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=30,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
}

trained_models = {}   
val_results    = {}   
val_preds      = {}   
val_probs      = {}  

for name, model in models.items():
    print(f"\n▶ {name}")
    print("-" * 40)

    
    use_scaled = "Logistic" in name

    # ── Fit ────────────────────────────────────────────────────
    model.fit(X_train_sc if use_scaled else X_train, y_train)

    # ── Predict trên Validation ─────────────────────────────────
    val_pred      = model.predict(X_val_sc       if use_scaled else X_val)
    val_pred_prob = model.predict_proba(X_val_sc  if use_scaled else X_val)[:, 1]

    # ── Predict trên Train (để phát hiện overfitting) ───────────
    train_pred      = model.predict(X_train_sc       if use_scaled else X_train)
    train_pred_prob = model.predict_proba(X_train_sc  if use_scaled else X_train)[:, 1]

    # ── Metrics Train ───────────────────────────────────────────
    tr_acc  = accuracy_score(y_train,  train_pred)
    tr_prec = precision_score(y_train, train_pred,      zero_division=0)
    tr_rec  = recall_score(y_train,    train_pred,      zero_division=0)
    tr_f1   = f1_score(y_train,        train_pred,      zero_division=0)
    tr_auc  = roc_auc_score(y_train,   train_pred_prob)

    # ── Metrics Validation ──────────────────────────────────────
    val_acc  = accuracy_score(y_val,  val_pred)
    val_prec = precision_score(y_val, val_pred,      zero_division=0)
    val_rec  = recall_score(y_val,    val_pred,      zero_division=0)
    val_f1   = f1_score(y_val,        val_pred,      zero_division=0)
    val_auc  = roc_auc_score(y_val,   val_pred_prob)

    # ── In bảng so sánh Train vs Validation ─────────────────────
    print(f"  {'Metric':<12} {'Train':>10} {'Validation':>12}")
    print(f"  {'-'*36}")
    print(f"  {'Accuracy':<12} {tr_acc:>10.4f} {val_acc:>12.4f}")
    print(f"  {'Precision':<12} {tr_prec:>10.4f} {val_prec:>12.4f}")
    print(f"  {'Recall':<12} {tr_rec:>10.4f} {val_rec:>12.4f}")
    print(f"  {'F1 Score':<12} {tr_f1:>10.4f} {val_f1:>12.4f}")
    print(f"  {'ROC-AUC':<12} {tr_auc:>10.4f} {val_auc:>12.4f}")

    # ── Confusion Matrix (Validation) ───────────────────────────
    cm = confusion_matrix(y_val, val_pred)
    print(f"\n  Confusion Matrix (Validation):")
    print(f"  {'':>16} Pred: Ở lại  Pred: Rời đi")
    print(f"  {'True: Ở lại':>16}   {cm[0, 0]:>9}    {cm[0, 1]:>10}")
    print(f"  {'True: Rời đi':>16}   {cm[1, 0]:>9}    {cm[1, 1]:>10}")

    # ── Lưu kết quả ─────────────────────────────────────────────
    trained_models[name] = model
    val_preds[name]      = val_pred
    val_probs[name]      = val_pred_prob
    val_results[name]    = {
        "Accuracy" : val_acc,
        "Precision": val_prec,
        "Recall"   : val_rec,
        "F1 Score" : val_f1,
        "ROC-AUC"  : val_auc,
    }

# ═══════════════════════════════════════════════════════════════
# 6. SO SÁNH MÔ HÌNH TRÊN TẬP VALIDATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. SO SÁNH MÔ HÌNH TRÊN TẬP VALIDATION")
print("=" * 60)

comparison_df = pd.DataFrame(val_results).T.sort_values("ROC-AUC", ascending=False)
print(comparison_df.to_string(float_format=lambda x: f"{x:.4f}"))

best_model_name = comparison_df.index[0]
best_model      = trained_models[best_model_name]
print(f"\n→ Mô hình tốt nhất (theo ROC-AUC): {best_model_name}")

# ═══════════════════════════════════════════════════════════════
# 7. LƯU MÔ HÌNH 
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. LƯU MÔ HÌNH (.pkl)")
print("=" * 60)

for name, model in trained_models.items():
    safe_name  = name.lower().replace(" ", "_")
    model_path = f"model_{safe_name}.pkl"
    joblib.dump(model, model_path)
    print(f"  ✓ {name:25s} → {model_path}")

joblib.dump(scaler, "scaler.pkl")
print(f"  ✓ {'StandardScaler':25s} → scaler.pkl")

# ═══════════════════════════════════════════════════════════════
# 8. VẼ BIỂU ĐỒ ROC-AUC 
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. VẼ BIỂU ĐỒ ROC-AUC")
print("=" * 60)

COLORS = {
    "Logistic Regression": "#2563EB",
    "Decision Tree"      : "#16A34A",
    "Random Forest"      : "#DC2626",
}

fig, ax = plt.subplots(figsize=(8, 6))

for name, probs in val_probs.items():
    fpr, tpr, _ = roc_curve(y_val, probs)
    auc_score   = val_results[name]["ROC-AUC"]
    ax.plot(fpr, tpr, color=COLORS[name], linewidth=2,
            label=f"{name}  (AUC = {auc_score:.4f})")

ax.plot([0, 1], [0, 1], color="#9CA3AF", linewidth=1.2,
        linestyle="--", label="Random Baseline (AUC = 0.5000)")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — So sánh các mô hình (Validation Set)",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
roc_path = "roc_auc_comparison.png"
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"  ✓ Biểu đồ ROC-AUC đã được lưu → {roc_path}")

# ═══════════════════════════════════════════════════════════════
# 9. PREDICT TRÊN TẬP VALIDATION & LƯU KẾT QUẢ CSV
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("9. PREDICT TRÊN TẬP VALIDATION & LƯU KẾT QUẢ")
print("=" * 60)

for name in trained_models:
    safe_name = name.lower().replace(" ", "_")
    out_path  = f"predictions_{safe_name}.csv"

    output_df = meta_val.copy()
    output_df["actual_churn"]    = actual_val.values   # Nhãn thực tế
    output_df["predicted_churn"] = val_preds[name]     # Nhãn dự đoán (tái dùng từ bước 5)

    output_df[["id", "full_name", "actual_churn", "predicted_churn"]].to_csv(
        out_path, index=False
    )
    n_churn = int(val_preds[name].sum())
    print(f"  ✓ {name:25s} → {out_path}  ({n_churn:,} khách dự báo rời đi)")

print("\n Tất cả file đầu ra đã được lưu.")
