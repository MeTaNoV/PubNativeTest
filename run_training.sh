#!/bin/bash

# POST method predict
curl -d '{
    "v1": "b", 
    "v2": -0.044035228182546376, 
    "v3": 0.2856879380302472, 
    "v7": "bb", 
    "v8": -0.43493150684931514, 
    "v9": "t", 
    "v10": "f", 
    "v11": 0, 
    "v12": "t", 
    "v15": 0,
    "classLabel": "no."
}' -H "Content-Type: application/json" \
   -X POST http://localhost:5000/train #&& \
#echo -e "\n -> predict OK"
