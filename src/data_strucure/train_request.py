from pydantic import BaseModel


class TrainRequest(BaseModel):
    csv_path: str
    test_size: float = 0.15
    validation_size: float = 0.15
