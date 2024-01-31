import random

from ovq.utils.fileIO import jload, jdump

if __name__ == '__main__':
    obj = jload("../data/alpaca_data.json")
    random.shuffle(obj)
    split_index = int(len(obj) * 0.8)
    train_data = obj[:split_index]
    eval_data = obj[split_index:]
    jdump(train_data, "../data/train_data.json")
    jdump(eval_data, "../data/eval_data.json")
