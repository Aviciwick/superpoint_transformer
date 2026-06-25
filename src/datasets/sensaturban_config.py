import numpy as np


SENSATURBAN_NUM_CLASSES = 13

SENSATURBAN_CITY_SPLIT_COUNTS = {
    "birmingham": {"train": 10, "val": 2, "test": 2},
    "cambridge": {"train": 18, "val": 4, "test": 4},
}

CLASS_NAMES = [
    "ground",
    "vegetation",
    "building",
    "wall",
    "bridge",
    "parking",
    "rail",
    "traffic road",
    "street furniture",
    "car",
    "footpath",
    "bike",
    "water",
    "ignored",
]

CLASS_COLORS = np.asarray([
    [85, 107, 47],
    [0, 255, 0],
    [255, 165, 0],
    [41, 49, 101],
    [0, 0, 255],
    [255, 0, 255],
    [200, 200, 200],
    [255, 0, 0],
    [255, 255, 0],
    [0, 255, 255],
    [255, 105, 180],
    [160, 32, 240],
    [0, 191, 255],
    [0, 0, 0],
], dtype=np.uint8).tolist()

STUFF_CLASSES = list(range(SENSATURBAN_NUM_CLASSES))
ID2TRAINID = np.arange(SENSATURBAN_NUM_CLASSES, dtype=np.int64)
