# 🩺 Diabetes Risk Prediction

A machine learning project that predicts whether a person is at risk of diabetes using a **VotingClassifier (XGBoost + LightGBM)** model and a Streamlit web application.

## ✨ Features

- Ensemble model: **VotingClassifier** (XGBoost + LightGBM, soft voting)
- 18 top features including engineered variables (metabolic score, interaction terms, lipid ratios)
- Streamlit UI with:
  - Real-time diabetes risk score (%)
  - Risk gauge visualization (Plotly)
  - Low / Medium / High risk banner
  - Computed features panel
  - Input summary table

## 📋 Requirements

- Python 3.10+

```bash
pip install -r requirements.txt
```

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tugcesi/Diabetes-Prediction.git
   cd Diabetes-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

```bash
streamlit run app.py
```

## 🗂️ Project Structure

```text
.
├── app.py                              # Streamlit uygulaması
├── diabetespredictionchallenge.ipynb   # EDA + Feature Engineering + Model
├── model.joblib                        # Eğitilmiş VotingClassifier
├── scaler.joblib                       # StandardScaler
├── top_features.joblib                 # Seçilen feature listesi
├── requirements.txt
└── README.md
```

## ⚙️ Technical Details

| Özellik | Detay |
|---------|-------|
| **Model** | `VotingClassifier` — XGBClassifier + LGBMClassifier (soft voting) |
| **Target** | `diagnosed_diabetes` (0 / 1) |
| **Top Features** | 18 feature (ham + engineered) |
| **Encoding** | `pd.get_dummies(drop_first=True)` |
| **Scaling** | `StandardScaler` (`scaler.joblib`) |
| **Engineered** | `age_bmi_interaction`, `metabolic_score`, `trig_hdl_ratio`, `ldl_hdl_ratio`, `bmi_bp_interaction`, `age_activity_interaction` |

## 📊 Engineered Features

| Feature | Formül |
|---------|--------|
| `age_bmi_interaction` | `age × bmi / 100` |
| `age_activity_interaction` | `age × (activity / 100)` |
| `bmi_bp_interaction` | `(bmi / 25) × (systolic_bp / 120)` |
| `metabolic_score` | BMI + WHR normalize kombinasyonu (0–100) |
| `ldl_hdl_ratio` | `ldl / (hdl + 1)` |
| `trig_hdl_ratio` | `triglycerides / (hdl + 1)` |

## ⚕️ Disclaimer

> Bu uygulama yalnızca **bilgilendirme amaçlıdır** ve tıbbi teşhis yerine geçmez. Sağlık durumunuz hakkında her zaman bir doktora danışın.

## 👤 Author

Tugce Basyigit

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
