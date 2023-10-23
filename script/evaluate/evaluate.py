import json
import requests
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


def get_response(inputs, max_seq_length_, split_token_):
    # url = "http://219.216.64.231:7032/firefly"
    url = "http://219.216.64.75:7032/firefly"
    payload = json.dumps({
        "inputs": inputs,
        "max_seq_length": max_seq_length_,
        "split_token": split_token_
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    output = response.json()["response"]
    return output
