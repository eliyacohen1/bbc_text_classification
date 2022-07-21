import torch
from fastapi import FastAPI
import uvicorn
from src.data_strucure import TrainRequest, PredictionRequest
from src.utils import load_df, split_data, train_model, create_datasets, create_dataloader, evaluate, predict
from configuration.config import config, MODELS_FOLDERS
from bootstraps import wrapper_model_bootstrap, dataset_bootstrap, tokenizer_bootstrap

# Our App
app = FastAPI()

# Configs
TRAIN_SETTINGS = config["train_settings"]
CONFIG_LABELS = config["labels"]

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data(train_request: TrainRequest):
    # Load csv data
    df = load_df(train_request.csv_path)
    print(df.shape)

    # Create data loaders
    train, val, test = split_data(df=df, val_size=train_request.validation_size, test_size=train_request.test_size)
    dataset_obj = dataset_bootstrap(config)
    tokenizer = tokenizer_bootstrap(config)
    ds_train, ds_validation, ds_test = create_datasets(train, val, test, dataset_obj, tokenizer, CONFIG_LABELS)
    return create_dataloader(ds_train, ds_validation, ds_test, TRAIN_SETTINGS["batch_size"])


@app.post("/train/")
async def train(train_request: TrainRequest):
    train_dataloader, val_dataloader, test_dataloader = prepare_data(train_request)
    model = wrapper_model_bootstrap(config)
    model = train_model(model, train_dataloader, val_dataloader,
                        epochs=TRAIN_SETTINGS["epochs"],
                        learning_rate=TRAIN_SETTINGS["lr"],
                        optimizer=TRAIN_SETTINGS["optimizer"],
                        weight_decay=TRAIN_SETTINGS["weight_decay"])
    torch.save(model.state_dict(), MODELS_FOLDERS.format(config["model_name"]))
    results = evaluate(model, test_dataloader, CONFIG_LABELS)
    print(results)
    return {"results": results}


@app.post("/prediction/")
async def prediction(prediction_request: PredictionRequest):
    print("Start Prediction Task")
    model = wrapper_model_bootstrap(config)
    model.load_state_dict(torch.load(MODELS_FOLDERS.format(config["model_name"]), map_location=torch.device(DEVICE)))
    tokenizer = tokenizer_bootstrap(config)
    results = predict(prediction_request.text, model, tokenizer, {value: key for key, value in CONFIG_LABELS.items()},
                      top_k=prediction_request.k)
    print("Done Prediction Task")
    return {"results": results}


@app.get("/")
async def health():
    return "Text Classification Is Up, For More Details Go To /docs"


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)
