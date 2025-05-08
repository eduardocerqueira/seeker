#date: 2025-05-08T16:33:20Z
#url: https://api.github.com/gists/4f381d100ca3034514e29ca03d51c246
#owner: https://api.github.com/users/daviddwlee84

from dataclasses import dataclass
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import BaseModel
from fastapi import FastAPI
from starlette.responses import RedirectResponse

app = FastAPI()


# Redirect root URL to the automatic docs
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


# 1. Standard library dataclass
@dataclass
class StandardItem:
    name: str
    price: float


# 2. Pydantic-provided dataclass
@pydantic_dataclass
class PydanticDataclassItem:
    name: str
    price: float


# 3. Pydantic BaseModel
class BaseModelItem(BaseModel):
    name: str
    price: float


# Endpoint using standard dataclass
@app.post("/standard", response_model=StandardItem)
def create_standard(item: StandardItem):
    # FastAPI will internally convert the plain dataclass to a Pydantic model
    # for validation and OpenAPI schema generation
    return item


# Endpoint using @pydantic.dataclasses.dataclass
@app.post("/pydantic-dataclass", response_model=PydanticDataclassItem)
def create_pydantic_dataclass(item: PydanticDataclassItem):
    # Pydantic dataclass performs validation and type coercion on init
    return item


# Endpoint using BaseModel
@app.post("/base-model", response_model=BaseModelItem)
def create_base_model(item: BaseModelItem):
    # Standard Pydantic model with full validation features
    return item


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_dataclass_example:app", host="127.0.0.1", port=8000, reload=True
    )
