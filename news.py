from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import re
import string
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Fake News Detection API",
              description="API that predicts whether a news text is Fake or Real.",
              version="1.0")

# -----------------------------
# Load models and vectorizer
# -----------------------------
with open("vectorizer.pkl", "rb") as f:
    vectorization = pickle.load(f)

with open("LR_model.pkl", "rb") as f:
    LR = pickle.load(f)

with open("DT_model.pkl", "rb") as f:
    DT = pickle.load(f)

with open("GB_model.pkl", "rb") as f:
    GB = pickle.load(f)

with open("RF_model.pkl", "rb") as f:
    RF = pickle.load(f)

# -----------------------------
# Utility Functions
# -----------------------------
def wordopt(text: str) -> str:
    text = re.sub('\[.*?/]','', text)
    text = re.sub('\\W', " ", text)
    text = re.sub("https?://\S+|www\.\S+", '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\w*\d\w*', '', text)
    return text.strip().lower()

def output_label(n: int) -> str:
    return "Fake News" if n == 0 else "Real News"

# -----------------------------
# Request Model
# -----------------------------
class NewsText(BaseModel):
    text: str

# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def home():
    return {"message": "Welcome to Fake News Detection API. Use /predict endpoint with POST request."}

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict_news(news: NewsText):
    try:
        # Step 1: Clean text
        cleaned_text = wordopt(news.text)

        # Step 2: Convert to DataFrame
        df = pd.DataFrame({"text": [cleaned_text]})

        # Step 3: Transform using vectorizer
        new_xv_test = vectorization.transform(df["text"])

        # Step 4: Predict using all models
        pred_LR = LR.predict(new_xv_test)[0]
        pred_DT = DT.predict(new_xv_test)[0]
        pred_GB = GB.predict(new_xv_test)[0]
        pred_RF = RF.predict(new_xv_test)[0]

        # Step 5: Prepare response
        response = {
            "LogisticRegression": output_label(pred_LR),
            "DecisionTree": output_label(pred_DT),
            "GradientBoosting": output_label(pred_GB),
            "RandomForest": output_label(pred_RF)
        }

        return {"input_text": news.text, "predictions": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Run API (only if executed directly)
# -----------------------------
if _name_ == "_main_":
    uvicorn.run(app, host="0.0.0.0", port=8000)