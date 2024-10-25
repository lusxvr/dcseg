import numpy as np

class_names_40 = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floor mat",
    "clothes",
    "ceiling",
    "books",
    "refridgerator",
    "television",
    "paper",
    "towel",
    "shower curtain",
    "box",
    "whiteboard",
    "person",
    "nightstand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
    "otherstructure",
    "otherfurniture",
    "otherprop",
]
class_names_20 = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]

# from https://github.com/ScanNet/ScanNet/blob/fcaa1773a9e186b22a4228df632b6999a1633b79/BenchmarkScripts/3d_evaluation/evaluate_semantic_label.py#L43

VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

# If we prompt out segmentation model with the 20+1 classes of scannet, we will indices of 0...20 back indicating the classes. This map mapps these back to the
# nyu40 class ids
MAP_TO_VALID_ID = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 14,
    13: 16,
    14: 24,
    15: 28,
    16: 33,
    17: 34,
    18: 36,
    19: 39,
    20: 40, # Mapping of our '0' to the scannet UNKNOWN_ID = 40
    21: 40
}

# color palette for nyu40 labels
color_palette_40 = [
    (174, 199, 232),  # wall
    (152, 223, 138),  # floor
    (31, 119, 180),  # cabinet
    (255, 187, 120),  # bed
    (188, 189, 34),  # chair
    (140, 86, 75),  # sofa
    (255, 152, 150),  # table
    (214, 39, 40),  # door
    (197, 176, 213),  # window
    (148, 103, 189),  # bookshelf
    (196, 156, 148),  # picture
    (23, 190, 207),  # counter
    (178, 76, 76),
    (247, 182, 210),  # desk
    (66, 188, 102),
    (219, 219, 141),  # curtain
    (140, 57, 197),
    (202, 185, 52),
    (51, 176, 203),
    (200, 54, 131),
    (92, 193, 61),
    (78, 71, 183),
    (172, 114, 82),
    (255, 127, 14),  # refrigerator
    (91, 163, 138),
    (153, 98, 156),
    (140, 153, 101),
    (158, 218, 229),  # shower curtain
    (100, 125, 154),
    (178, 127, 135),
    (120, 185, 128),
    (146, 111, 194),
    (44, 160, 44),  # toilet
    (112, 128, 144),  # sink
    (96, 207, 209),
    (227, 119, 194),  # bathtub
    (213, 92, 176),
    (94, 106, 211),
    (82, 84, 163),  # otherfurn
    (100, 85, 144),
    (0, 0, 0),
]

color_palette_20 = [
    (174, 199, 232),  # wall
    (152, 223, 138),  # floor
    (31, 119, 180),  # cabinet
    (255, 187, 120),  # bed
    (188, 189, 34),  # chair
    (140, 86, 75),  # sofa
    (255, 152, 150),  # table
    (214, 39, 40),  # door
    (197, 176, 213),  # window
    (148, 103, 189),  # bookshelf
    (196, 156, 148),  # picture
    (23, 190, 207),  # counter
    (247, 182, 210),  # desk
    (219, 219, 141),  # curtain
    (255, 127, 14),  # refrigerator
    (158, 218, 229),  # shower curtain
    (44, 160, 44),  # toilet
    (112, 128, 144),  # sink
    (227, 119, 194),  # bathtub
    (82, 84, 163),  # otherfurn
    (0, 0, 0),
]

