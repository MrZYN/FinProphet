import pandas as pd
import json
import os

def csv2json(path: str):
    with open(path, 'r') as f:
        raw = json.load(f)
    
    print(len(raw))
    
    max_length = 0
    total_length = 0
    result = []
    for dic in raw:
        max_length = max(max_length, len(dic["instruction"]))
        total_length += len(dic["instruction"])
        dic["system"] = "Given the context, calculate the answer for the financial question. The context is: " + dic["instruction"]
        del dic["instruction"]
        result.append(dic)
    
    print("size: ", len(result))
    print("max_length: ", max_length)
    print("mean: ", total_length//len(result))
    
    # base_name = path.split('/')[-1].split('.')[0]
    with open(os.path.join(f'data/{path}'), 'w') as f:
        json.dump(result, f, ensure_ascii=False)
    

if __name__ == "__main__":
    csv2json('finance_test.json')
    csv2json('finance_train.json')
    # csv2json('cnn_dailymail/validation.csv')
    # csv2json('cnn_dailymail/train.csv')