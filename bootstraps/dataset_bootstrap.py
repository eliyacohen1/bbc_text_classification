from src.dataset import DatasetForBert


def dataset_bootstrap(config: dict):
    if config["model_name"] == "bert":
        return DatasetForBert
