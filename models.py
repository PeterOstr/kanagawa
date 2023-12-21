from typing import List
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

class TableData(BaseModel):
    items: List[Item]
