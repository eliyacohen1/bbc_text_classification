from fastapi import FastAPI
import uvicorn
from data_strucure import TrainRequest, PredictionRequest
from utils import load_df, generate_datasets, train_model
from config import config
from bootstraps import wrapper_model_bootstrap
app = FastAPI()


@app.post("/train_bert/")
async def create_item(train_request: TrainRequest):
    df = load_df(train_request.csv_path)
    train_settings = config["train_settings"]

    train_dataloader, val_dataloader, test_dataloader = generate_datasets(df=df,
                                                                          val_size=train_request.validation_size,
                                                                          test_size=train_request.test_size,
                                                                          batch_size=train_settings["batch_size"])
    model = wrapper_model_bootstrap(config)
    train_model(model, train_dataloader, val_dataloader)
    return train_request


@app.post("/prediction_bert/")
async def create_item(prediction_request: PredictionRequest):
    return prediction_request


@app.get("/")
async def health():
    return "Text Classification Is Up, For More Details /docs"


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)  # , workers=1)
