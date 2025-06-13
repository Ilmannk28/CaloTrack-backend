from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from services.predictor import predict_calories
from services.recommendation_system import generate_meal_plan
import shutil
import uuid
from pathlib import Path
import json 



app = FastAPI()

# CORS: supaya frontend bisa akses backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("backend/temp_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    file_ext = file.filename.split('.')[-1]
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}.{file_ext}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = predict_calories(str(file_path))
        return result
    except Exception as e:
        return {"error": str(e)}
    finally:
        file_path.unlink(missing_ok=True)

@app.post("/recommendation")
async def recommend_meal(calorie_goal: int = Form(...)):
    try:
        result = generate_meal_plan(calorie_goal)
        return result
    except Exception as e:
        return {"error": str(e)}
    

@app.get("/recommend")
async def recommend(calories: int = Query(..., ge=1)):
    return generate_meal_plan(calories)

@app.get("/articles")
def get_articles():
    try:
        file_path = Path(__file__).parent / "data" / "articles.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"articles": data}
    except Exception as e:
        return {"error": str(e)}


