import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

%matplotlib inline

# ----- 1. Veri okuma -------------------------------------------------
df = pd.read_csv('creditcard.csv')
print("Shape:", df.shape)

X = df.drop(columns=['Class'])
y = df['Class']

# ----- 2. Ölçekleme ---------------------------------------------------
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----- 3. Model (cont=0.003) -----------------------------------------
best_cont = 0.003      # recall ~0.51
best_pct  = 99.3       # eşik

iso_best = IsolationForest(
    n_estimators=300,
    contamination=best_cont,
    random_state=42,
    n_jobs=-1
).fit(X_train)

# ----- 4. Tahmin & metrikler -----------------------------------------
scores = -iso_best.decision_function(X_test)
thr    = np.percentile(scores, best_pct)
y_pred = (scores > thr).astype(int)

cm  = confusion_matrix(y_test, y_pred, labels=[0,1])
rep = classification_report(y_test, y_pred, digits=4)
print(cm)
print(rep)

# ----- 5. Confusion Matrix ısı haritası -----------------------
plt.figure(figsize=(5,4))
sns.heatmap(cm,
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal (0)', 'Fraud (1)'],
            yticklabels=['Normal (0)', 'Fraud (1)'])
plt.title('Confusion Matrix – IF (cont=0.003, pct=99.3)')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout(); plt.show()

# ----- 6. PCA görselleştirme -----------------------------------------
pca = PCA(n_components=2, random_state=42)
X_vis = pca.fit_transform(X_test)

plt.figure(figsize=(7,5))
plt.scatter(X_vis[:,0], X_vis[:,1],
            c=y_pred, cmap='coolwarm', s=2, alpha=0.6)
plt.title('Isolation Forest – Anomali Dağılımı (pct 99.3)')
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.show()

# ----- 7. Model & scaler kaydet --------------------------------------
joblib.dump(iso_best, 'isolation_forest_credit.pkl', compress=('gzip', 3))
joblib.dump(scaler,   'scaler.pkl')
