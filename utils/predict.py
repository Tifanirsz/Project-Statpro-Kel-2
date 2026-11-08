import joblib
import pandas as pd

_model = joblib.load("kmeans_model.pkl")
_scaler = joblib.load("scaler.pkl")
_info = joblib.load("model_info.pkl")
FITUR = _info["fitur"]

def prediksi_senioritas(nilai: dict) -> str:
    """Format input: {'YearsAtCompany': 5, 'YearsInCurrentRole': 3, 'TotalWorkingYears': 10}"""
    x = pd.DataFrame([[nilai[f] for f in FITUR]], columns=FITUR)
    raw = int(_model.predict(_scaler.transform(x))[0])
    return _info["nama_segmen"][_info["peta"][raw]]