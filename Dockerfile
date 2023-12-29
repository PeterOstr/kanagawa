FROM python:3.10-slim
WORKDIR /app
COPY xgb_optuna_model.joblib .
COPY main.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

ENTRYPOINT ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]