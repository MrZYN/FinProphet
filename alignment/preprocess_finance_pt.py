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
        if len(dic["text"]) < 20:
            continue
        max_length = max(max_length, len(dic["text"]))
        total_length += len(dic["text"])
        result.append(dic)
    
    print("size: ", len(result))
    print("max_length: ", max_length)
    print("total: ", total_length)
    
    # base_name = path.split('/')[-1].split('.')[0]
    with open(os.path.join('data/result.json'), 'w') as f:
        json.dump(result, f, ensure_ascii=False)
        
    # with open(os.path.join('data/finance_test.json'), 'w') as f:
    #     json.dump(test, f, ensure_ascii=False)
    

if __name__ == "__main__":
    csv2json('data/raw.json')
    # csv2json('cnn_dailymail/validation.csv')
    # csv2json('cnn_dailymail/train.csv')