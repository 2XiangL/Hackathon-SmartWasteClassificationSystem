from ultralytics import YOLO
from openai import OpenAI


class APIKEYNullError(Exception):
    def __init__(self, message="API_KEY is null."):
        super().__init__(message)

def load_yolo_model():

    model = YOLO("./weights/model.pt")
    return model

def init_llm(api_key: str):

    if api_key.strip() == "":
        raise APIKEYNullError

    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    return client
