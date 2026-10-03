from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np 

# Don't forget to import your services from here 
# As an example, we import ApplySobel function from the EdgeDetectionService
from Services.EdgeDetectionService import ApplySobel

app = FastAPI()

# A custom request shape. Useful to group all your required inputs for a particular endpoint
# Example Below: A custom input for taking an image and applying an edge detection technique
# It takes a variable named image of type np.array as well as a variable named edge_detection_technique of type string
class EdgeDetectionRequest(BaseModel):
    image: np.array
    edge_detection_technique: str

# Create similar routes with inputs and outputs depending on your goal / task 
@app.get("/")
def read_root():
    return {"Hello": "World"}
