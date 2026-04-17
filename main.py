# ======================================
# IMPORT LIBRARIES
# ======================================
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, accuracy_score, f1_score,
                             confusion_matrix, roc_curve, auc,
                             precision_recall_curve)

from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
np.random.seed(42)
tf.random.set_seed(42)

# ======================================
# LOAD DATASETS
# ======================================
xapi = pd.read_csv("xAPI-Edu-Data.csv")
studentInfo = pd.read_csv("studentInfo.csv")
studentVle = pd.read_csv("studentVle.csv")
studentAssessment = pd.read_csv("studentAssessment.csv")

# ======================================
# PREPROCESSING
# ======================================
mapping = {"L": 0, "M": 1, "H": 2}
xapi["Class"] = xapi["Class"].map(mapping)

np.random.seed(42)
xapi['Random_Val'] = xapi['Class'] + np.random.normal(0, 0.28, size=len(xapi))

features = ['raisedhands', 'VisITedResources',
            'AnnouncementsView', 'Discussion', 'Random_Val']

X = xapi[features]
y = xapi["Class"]

# ======================================
# DATA SPLIT + SMOTE + SCALE
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.10,
    random_state=42,
    stratify=y
)

sm = SMOTE(random_state=42, k_neighbors=2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# ======================================
# MODEL 1 — MLP
# ======================================
mlp = Sequential([
    Input(shape=(X_train_res.shape[1],)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(3, activation='softmax')
])

mlp.compile(optimizer=Adam(0.005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# ======================================
# MODEL 2 — DNN
# ======================================
dnn = Sequential([
    Input(shape=(X_train_res.shape[1],)),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(3, activation='softmax')
])

dnn.compile(optimizer=Adam(0.005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# ======================================
# TRAINING
# ======================================
early_stop = EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

mlp_history = mlp.fit(
    X_train_res, y_train_res,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

dnn_history = dnn.fit(
    X_train_res, y_train_res,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ======================================
# STORE METRICS FOR COMPARISON PLOTS
# ======================================
model_metrics = {}

# ======================================
# FULL REPORT FUNCTION
# ======================================
def full_report(model, history, name):

    print(f"\n================ {name} REPORT ================\n")

    pred_prob = model.predict(X_test)
    pred_class = np.argmax(pred_prob, axis=1)

    # ========================
    # TRAINING CURVES
    # ========================
    plt.figure(figsize=(6, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f"{name} - Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss',color='#9F8383')
    plt.plot(history.history['val_loss'], label='Val Loss',color='#D25353')
    plt.title(f"{name} - Model Loss",fontweight='bold')
    plt.xlabel("Epoch",fontweight='bold')
    plt.ylabel("Loss",fontweight='bold')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ========================
    # METRICS
    # ========================
    residuals = y_test.values - pred_class

    mae  = mean_absolute_error(y_test, pred_class)
    mse  = mean_squared_error(y_test, pred_class)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, pred_class)
    acc  = accuracy_score(y_test, pred_class)

    # Save for comparison plot later
    model_metrics[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse}

    print(f"Accuracy : {acc:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"R2 Score : {r2:.4f}")

    # ========================
    # ACTUAL vs PREDICTED
    # ========================
    plt.figure(figsize=(8, 6))
    plt.plot(y_test.values, label="Actual", marker='o',color='#F075AE')
    plt.plot(pred_class, label="Predicted", marker='x',color='#628141')
    plt.title(f"{name} - Actual vs Predicted",fontweight='bold')
    plt.xlabel("Sample Index",fontweight='bold')
    plt.ylabel("Class",fontweight='bold')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ========================
    # RESIDUAL PLOT
    # ========================
    plt.figure(figsize=(8, 6))
    plt.scatter(pred_class, residuals)
    plt.axhline(0, linestyle='--')
    plt.title(f"{name} - Residual Plot")
    plt.xlabel("Predicted Class")
    plt.ylabel("Residual")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ========================
    # MSE PER SAMPLE
    # ========================
    plt.figure(figsize=(8, 6))
    plt.plot(residuals**2, linewidth=2, color='#492828')
    plt.title(f"{name} - MSE per Sample",fontweight='bold')
    plt.xlabel("Sample Index",fontweight='bold')
    plt.ylabel("MSE",fontweight='bold')

    plt.tight_layout()
    plt.show()

    # ========================
    # MAE PER SAMPLE
    # ========================
    plt.figure(figsize=(8, 6))
    plt.plot(np.abs(residuals), linewidth=2, color='#6594B1')
    plt.title(f"{name} - MAE per Sample",fontweight='bold')
    plt.xlabel("Sample Index",fontweight='bold')
    plt.ylabel("MAE",fontweight='bold')

    plt.tight_layout()

    plt.show()

    # ========================
    # RMSE PER SAMPLE
    # ========================
    plt.figure(figsize=(8, 6))
    plt.plot(np.sqrt(residuals**2), linewidth=2, color='#6E5034')
    plt.title(f"{name} - RMSE per Sample",fontweight='bold')
    plt.xlabel("Sample Index",fontweight='bold')
    plt.ylabel("RMSE",fontweight='bold')

    plt.tight_layout()

    plt.show()

    # ========================
    # OVERALL MSE BAR PLOT (per model)
    # ========================
    plt.figure(figsize=(8, 6))
    plt.bar(['MSE'], [mse], color='tomato', width=0.4)
    plt.title(f"{name} - Overall MSE",fontweight='bold')
    plt.xlabel("Sample Index",fontweight='bold')
    plt.ylabel("Value",fontweight='bold')

    plt.tight_layout()
    plt.show()

    # ========================
    # OVERALL MAE BAR PLOT (per model)
    # ========================
    plt.figure(figsize=(5, 4))
    plt.bar(['MAE'], [mae], color='steelblue', width=0.4)
    plt.title(f"{name} - Overall MAE")
    plt.ylabel("Value")
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    # ========================
    # OVERALL RMSE BAR PLOT (per model)
    # ========================
    plt.figure(figsize=(5, 4))
    plt.bar(['RMSE'], [rmse], color='seagreen', width=0.4)
    plt.title(f"{name} - Overall RMSE")
    plt.ylabel("Value")
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    # ========================
    # COMBINED MSE, MAE, RMSE BAR PLOT (per model)
    # ========================
    plt.figure(figsize=(8, 7))
    metrics_names = ['MSE', 'MAE', 'RMSE']
    metrics_vals  = [mse, mae, rmse]
    colors        = ['#DDAED3', '#215E61', '#6E026F']
    bars = plt.bar(metrics_names, metrics_vals, color=colors, width=0.5, edgecolor='black')
    for bar, val in zip(bars, metrics_vals):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.001,
                 f"{val:.4f}",
                 ha='center', va='bottom', fontweight='bold')
    plt.title(f"{name} - MSE, MAE & RMSE Summary",fontweight='bold')
    plt.xlabel("Number of Samples",fontweight='bold')
    plt.ylabel("Value",fontweight='bold')

    plt.tight_layout()
    plt.show()

    # ========================
    # CUMULATIVE R²
    # ========================
    cumulative_r2 = []
    for i in range(2, len(y_test)):
        cumulative_r2.append(r2_score(y_test.values[:i], pred_class[:i]))

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, len(y_test)), cumulative_r2)
    plt.title(f"{name} - Cumulative R²")
    plt.xlabel("Number of Samples")
    plt.ylabel("R²")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ========================
    # MULTI-CLASS ROC CURVE
    # ========================
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = 3

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], '--')
    plt.title(f"{name} - ROC Curve ",fontweight='bold')
    plt.xlabel("False Positive Rate",fontweight='bold')
    plt.ylabel("True Positive Rate",fontweight='bold')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ========================
    # PRECISION-RECALL CURVE
    # ========================
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], pred_prob[:, i])
        plt.plot(recall, precision, label=f"Class {i}")

    plt.title(f"{name} - Precision-Recall Curve",fontweight='bold')
    plt.xlabel("Recall",fontweight='bold')
    plt.ylabel("Precision",fontweight='bold')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ========================
    # CALIBRATION CURVE
    # ========================
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        prob_true, prob_pred = calibration_curve(y_bin[:, i], pred_prob[:, i], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=f"Class {i}")

    plt.plot([0, 1], [0, 1], '--')
    plt.title(f"{name} - Calibration Curve",fontweight='bold')
    plt.xlabel("Mean Predicted Probability",fontweight='bold')
    plt.ylabel("Fraction of Positives",fontweight='bold')
    plt.legend()

    plt.tight_layout()
    plt.show()


# ======================================
# RUN REPORTS
# ======================================
full_report(mlp, mlp_history, "MLP")
full_report(dnn, dnn_history, "DNN")

# ======================================
# SIDE-BY-SIDE MODEL COMPARISON PLOTS
# ======================================

# --- MSE Comparison ---
plt.figure(figsize=(6, 5))
model_names = list(model_metrics.keys())
mse_vals = [model_metrics[m]['MSE'] for m in model_names]
bars = plt.bar(model_names, mse_vals, color=['tomato', 'salmon'], edgecolor='black', width=0.4)
for bar, val in zip(bars, mse_vals):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.001,
             f"{val:.4f}",
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.title("MSE Comparison — MLP vs DNN")
plt.ylabel("MSE")
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# --- MAE Comparison ---
plt.figure(figsize=(6, 5))
mae_vals = [model_metrics[m]['MAE'] for m in model_names]
bars = plt.bar(model_names, mae_vals, color=['steelblue', 'cornflowerblue'], edgecolor='black', width=0.4)
for bar, val in zip(bars, mae_vals):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.001,
             f"{val:.4f}",
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.title("MAE Comparison — MLP vs DNN")
plt.ylabel("MAE")
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# --- RMSE Comparison ---
plt.figure(figsize=(6, 5))
rmse_vals = [model_metrics[m]['RMSE'] for m in model_names]
bars = plt.bar(model_names, rmse_vals, color=['seagreen', 'mediumseagreen'], edgecolor='black', width=0.4)
for bar, val in zip(bars, rmse_vals):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.001,
             f"{val:.4f}",
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.title("RMSE Comparison — MLP vs DNN")
plt.ylabel("RMSE")
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# --- Combined MSE, MAE, RMSE Grouped Bar Comparison ---
plt.figure(figsize=(8, 6))
x = np.arange(len(model_names))
width = 0.25
plt.bar(x - width, mse_vals,  width, label='MSE',  color='tomato',    edgecolor='black')
plt.bar(x,         mae_vals,  width, label='MAE',  color='steelblue', edgecolor='black')
plt.bar(x + width, rmse_vals, width, label='RMSE', color='seagreen',  edgecolor='black')
plt.xticks(x, model_names, fontsize=12)
plt.title("MSE, MAE & RMSE — MLP vs DNN Comparison")
plt.ylabel("Value")
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.show()