import numpy as np

TORONTO3D_NUM_CLASSES = 8

CLASS_NAMES = [
    "road",
    "road marking",
    "natural",
    "building",
    "utility line",
    "pole",
    "car",
    "fence",
    "ignored",
]

CLASS_COLORS = np.asarray([
    [128, 64, 128],
    [255, 255, 0],
    [107, 142, 35],
    [70, 70, 70],
    [153, 153, 153],
    [153, 153, 0],
    [0, 0, 142],
    [190, 153, 153],
    [0, 0, 0],
], dtype=np.uint8).tolist()

STUFF_CLASSES = [0, 1, 2, 3, 4, 7]

ID2TRAINID = np.asarray([8, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)

TORONTO3D_SPLITS = {
    "train": ["L001", "L003", "L004"],
    "val": [],
    "test": ["L002"],
}
