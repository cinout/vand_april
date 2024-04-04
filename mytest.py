import torch
import random

MULTI_TYPES = {
    "juice_bottle": {
        "orange_juice": [0, 1, 5, 9, 12, 13, 14],
        "cherry_juice": [4, 8, 10, 15, 16],
        "banana_juice": [2, 3, 6, 7, 11],
    },
    "splicing_connectors": {
        "yellow_cable": [0, 2, 5, 6, 8],
        "blue_cable": [3, 7, 11, 15, 16],
        "red_cable": [1, 4, 9, 10, 12, 13, 14],
    },
}
indices = []
k_shot = 4
cls_name = "juice_bottle"
train_imgs = MULTI_TYPES[cls_name]
values = train_imgs.values()
for value in values:
    chosen = random.sample(value, k_shot)
    indices.extend(chosen)

print(indices)
