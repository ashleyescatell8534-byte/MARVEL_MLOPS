
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.predict import load_champion, predict_dialogue


app = FastAPI(
    title="MARVEL Character Predictor",
    description="API para predecir qué personaje de Marvel dijo un diálogo.",
    version="1.0.0"
)

model = None


@app.on_event("startup")
def startup_event():
    global model
    model = load_champion()


class DialogueRequest(BaseModel):
    dialogue: str = Field(
        min_length=1,
        description="Diálogo que se desea clasificar."
    )


@app.get("/")
def home():
    html_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(html_path)


@app.post("/predict")
def predict(request: DialogueRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo todavía no está disponible."
        )

    dialogue = request.dialogue.strip()

    if not dialogue:
        raise HTTPException(
            status_code=422,
            detail="El diálogo no puede estar vacío."
        )

    try:
        character = predict_dialogue(model, dialogue)

        return {
            "dialogue": dialogue,
            "predicted_character": str(character),
            "model": "MARVEL_Character_Classifier",
            "alias": "champion"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {str(error)}"
        )
