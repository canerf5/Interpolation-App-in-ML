############################################################################
# TELCO CUSTOMER CHURN - TEZ PROJESİ: GENİŞLETİLMİŞ İMPUTASYON ANALİZİ
# Hazırlayan: Caner Erenler
# Versiyon: Tüm Metotlar (7 Adet) + Tam Feature Engineering + 10-Fold CV
############################################################################
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.impute import KNNImputer

# Ayarlar
warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

print("############################################################################")
print("PROJE BAŞLATILIYOR: VERİ YÜKLEME VE HAZIRLIK")
print("############################################################################")


def load_and_prep_data():
    try:
        df = pd.read_csv("Telco-Customer-Churn.csv")
    except FileNotFoundError:
        print("HATA: 'Telco-Customer-Churn.csv' dosyası bulunamadı.")
        exit()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    return df


df = load_and_prep_data()

# %10 Yapay Eksiklik Oluşturma
df_with_nans = df.copy()
np.random.seed(123)
missing_indices = np.random.choice(df.index, int(len(df) * 0.10), replace=False)
df_with_nans.loc[missing_indices, "TotalCharges"] = np.nan

############################################################################
# 1. İMPUTASYON VE İNTERPOLASYON YÖNTEMLERİ (7 DATASET)
############################################################################
datasets = {}
datasets["1_Original"] = df.copy()

# A. Temel İmputasyon (Mean)
df_mean = df_with_nans.copy()
df_mean["TotalCharges"].fillna(df_mean["TotalCharges"].mean(), inplace=True)
datasets["2_Mean_Imputed"] = df_mean

# B. İnterpolasyon Grubu (Tenure'a Göre Sıralı)
df_sorted = df_with_nans.sort_values("tenure").copy()

# Linear
df_linear = df_sorted.copy()
df_linear["TotalCharges"] = df_linear["TotalCharges"].interpolate(method='linear').ffill().bfill()
datasets["3_Linear_Interpolation"] = df_linear.sort_index()

# Spline (Order 3)
df_spline = df_sorted.copy().reset_index()
df_spline["TotalCharges"] = df_spline["TotalCharges"].interpolate(method='spline', order=3).ffill().bfill()
datasets["4_Spline_Interpolation"] = df_spline.set_index("index").sort_index()

# Polynomial (Order 2)
df_poly = df_sorted.copy().reset_index()
df_poly["TotalCharges"] = df_poly["TotalCharges"].interpolate(method='polynomial', order=2).ffill().bfill()
datasets["5_Polynomial_Interpolation"] = df_poly.set_index("index").sort_index()

# C. Algoritmik İmputasyon
# KNN
df_knn = df_with_nans.copy()
imputer = KNNImputer(n_neighbors=5)
df_knn[["tenure", "MonthlyCharges", "TotalCharges"]] = imputer.fit_transform(
    df_knn[["tenure", "MonthlyCharges", "TotalCharges"]])
datasets["6_KNN_Imputed"] = df_knn

# Least Squares (Regression Imputation)
df_ls = df_with_nans.copy()
train_ls = df_ls.dropna(subset=["TotalCharges"])
test_ls = df_ls[df_ls["TotalCharges"].isna()]
ls_reg = LinearRegression()
ls_reg.fit(train_ls[["tenure", "MonthlyCharges"]], train_ls["TotalCharges"])
df_ls.loc[df_ls["TotalCharges"].isna(), "TotalCharges"] = ls_reg.predict(test_ls[["tenure", "MonthlyCharges"]])
datasets["7_Least_Squares"] = df_ls

print(f"-> 7 Farklı veri seti başarıyla hazırlandı.")


############################################################################
# 2. FEATURE ENGINEERING PIPELINE 
############################################################################

def feature_engineering_pipeline(dataframe):
    dff = dataframe.copy()

    # Outlier Handling
    def replace_with_thresholds(dataframe, variable):
        q1, q3 = 0.05, 0.95
        quartile1 = dataframe[variable].quantile(q1)
        quartile3 = dataframe[variable].quantile(q3)
        iqr = quartile3 - quartile1
        low, up = quartile1 - 1.5 * iqr, quartile3 + 1.5 * iqr
        dataframe.loc[(dataframe[variable] < low), variable] = low
        dataframe.loc[(dataframe[variable] > up), variable] = up

    num_cols = [col for col in dff.columns if dff[col].dtypes != "O" and col not in ["Churn", "customerID"]]
    for col in num_cols:
        replace_with_thresholds(dff, col)

    # Feature Extraction (Tüm yeni değişkenler eklendi)
    dff.loc[(dff["tenure"] >= 0) & (dff["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
    dff.loc[(dff["tenure"] > 12) & (dff["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
    dff.loc[(dff["tenure"] > 24) & (dff["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
    dff.loc[(dff["tenure"] > 36) & (dff["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
    dff.loc[(dff["tenure"] > 48) & (dff["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
    dff.loc[(dff["tenure"] > 60) & (dff["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

    dff["NEW_Engaged"] = dff["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)
    dff["NEW_noProt"] = dff.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (
                x["TechSupport"] != "Yes") else 0, axis=1)
    dff["NEW_Young_Not_Engaged"] = dff.apply(
        lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)
    dff['NEW_TotalServices'] = (dff[['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                     'DeviceProtection', 'TechSupport', 'StreamingTV',
                                     'StreamingMovies']] == 'Yes').sum(axis=1)
    dff["NEW_FLAG_ANY_STREAMING"] = dff.apply(
        lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)
    dff["NEW_FLAG_AutoPayment"] = dff["PaymentMethod"].apply(
        lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)
    dff["NEW_AVG_Charges"] = dff["TotalCharges"] / (dff["tenure"] + 1)
    dff["NEW_Increase"] = dff["NEW_AVG_Charges"] / dff["MonthlyCharges"]
    dff["NEW_AVG_Service_Fee"] = dff["MonthlyCharges"] / (dff['NEW_TotalServices'] + 1)

    # Encoding
    le = LabelEncoder()
    binary_cols = [col for col in dff.columns if dff[col].dtypes == "O" and dff[col].nunique() == 2]
    for col in binary_cols: dff[col] = le.fit_transform(dff[col])

    cat_cols = [col for col in dff.columns if dff[col].dtypes == "O" and col != "customerID"]
    dff = pd.get_dummies(dff, columns=cat_cols, drop_first=True)

    X = dff.drop(["Churn", "customerID"], axis=1)
    y = dff["Churn"]
    return X, y


############################################################################
# 3. MODELLEME DÖNGÜSÜ (10-FOLD CV)
############################################################################

classifiers = [
    ('LR', LogisticRegression(random_state=17)),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier(random_state=17)),
    ('RF', RandomForestClassifier(random_state=17)),
    ('SVM', SVC(gamma='auto', random_state=17)),
    ('XGB', XGBClassifier(random_state=17)),
    ("LightGBM", LGBMClassifier(random_state=17, verbose=-1)),
    ("CatBoost", CatBoostClassifier(verbose=False, random_state=17))
]

results_data = []

print("\n############################################################################")
print("ANALİZ BAŞLIYOR: 7 DATASET x 8 MODEL x 10-FOLD CV")
print("############################################################################")

for ds_name, df_curr in datasets.items():
    print(f"\n>>> İşleniyor: {ds_name}")
    X, y = feature_engineering_pipeline(df_curr)

    for model_name, model in classifiers:
        cv_res = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

        results_data.append({
            "Imputation_Method": ds_name,
            "Model": model_name,
            "Accuracy": cv_res['test_accuracy'].mean(),
            "F1_Score": cv_res['test_f1'].mean(),
            "ROC_AUC": cv_res['test_roc_auc'].mean()
        })
        print(f"   -> Model: {model_name} Tamamlandı.")

############################################################################
# 4. SONUÇLAR VE GÖRSELLEŞTİRME
############################################################################

results_df = pd.DataFrame(results_data)
print("\n" + "=" * 80)
print("FİNAL BAŞARI SIRALAMASI (ROC_AUC)")
print("=" * 80)
print(results_df.sort_values(by="ROC_AUC", ascending=False).to_string(index=False))

plt.figure(figsize=(16, 8))
sns.barplot(x="Model", y="ROC_AUC", hue="Imputation_Method", data=results_df)
plt.title("İmputasyon Metotlarının Model Başarısına Etkisi (Tüm Metotlar)")
plt.ylim(0.60, 0.90)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# EK: FEATURE IMPORTANCE VE SHAP ANALİZİ 
# -----------------------------------------------------------
import shap

# 1. En iyi modeli tekrar eğit (Örn: LightGBM - Orijinal Veri ile)
print("\n##################################################")
print("DETAYLI ANALİZ: FEATURE IMPORTANCE & SHAP")
print("##################################################")

# Orijinal veri seti üzerinde en iyi modeli kuralım
X_final, y_final = feature_engineering_pipeline(datasets["1_Original"])
final_model = LGBMClassifier(random_state=17, verbose=-1).fit(X_final, y_final)

# A. Feature Importance Grafiği
plt.figure(figsize=(10, 6))
importances = pd.DataFrame({'Feature': X_final.columns, 'Importance': final_model.feature_importances_})
importances = importances.sort_values(by='Importance', ascending=False).head(15)
sns.barplot(x='Importance', y='Feature', data=importances)
plt.title("LightGBM - En Önemli 15 Değişken")
plt.tight_layout()
plt.show()

# B. SHAP Analizi
# (Not: CatBoost/LightGBM/XGBoost için TreeExplainer kullanılır)
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_final)

# KONTROL VE ÇİZİM: SHAP değerlerinin yapısına göre çizim
plt.figure()

# Eğer shap_values bir listeyse (eski versiyonlar veya çok sınıflı modeller için)
if isinstance(shap_values, list):
    print("SHAP values liste formatında geldi, pozitif sınıf seçiliyor...")
    shap.summary_plot(shap_values[1], X_final, plot_type="dot")
# Eğer shap_values tek bir matrisse (senin durumun)
else:
    print("SHAP values matris formatında geldi, direkt çiziliyor...")
    shap.summary_plot(shap_values, X_final, plot_type="dot")

plt.show()
corr_value = df['tenure'].corr(df['TotalCharges'])
print(f"Pearson Korelasyonu: {corr_value:.3f}")
