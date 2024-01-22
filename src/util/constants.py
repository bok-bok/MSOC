import os

# VISUAL_MODELS_TRAIN = ["160d-trainmaskmkl", "256d-trainmasknocolor", "320d-trainmasknocolor"]
MODEL_WEIGHTS_PATH = os.path.join(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../models/weights",
    )
)

HUBERT_ENCODER_MODEL_STATE_DICT_PATH = os.path.join(
    MODEL_WEIGHTS_PATH,
    "hubert_encoder_state_dict.pth",
)

AST_MODEL_PATH = os.path.join(MODEL_WEIGHTS_PATH, "ast_model.pth")

VIVIT_MODEL_PATH = os.path.join(MODEL_WEIGHTS_PATH, "vivit_model.pth")

MC_TCN_MODEL_PATH = os.path.join(MODEL_WEIGHTS_PATH, "mc_tcn_model.pth")


VISUAL_MODELS_TRAIN = ["160d_train_masknocolor", "256d_train_masknocolor", "320d_train_masknocolor"]

VISUAL_BLENDINGS_TRAIN = ["m1s4", "m2h1", "m1o0", "m4o4e15", "m4s4"]
AUDIO_MODELS_TRAIN = [
    "freevc-pretrained-tuned70k",
    "freevc-smodel-notuning",
    "freevc-pretrained-notuning",
    "freevc-fromscratch-trained65k",
    "freevc-24hz-notuning",
    "freevc-fromscratch-trained50k",
    "freevc-pretrained-tuned109k",
    "freevc-pretrained-tuned40k",
    "freevc-fromscratch-trained109k",
    "freevc-fromscratch-trained109k",
]

FAKE = 1
ORIGINAL = 0


# subject1_subject2
TEST_GROUP = [
    "00001_00006",
    "00006_00001",
    "00003_00039",
    "00039_00003",
    "00003_00034",
    "00034_00003",
    "00034_00030",
    "00030_00034",
    "00017_00020",
    "00020_00017",
]

TRAIN_GROUP = [
    "00026_00024",
    "00035_00016",
    "00004_00002",
    "00004_00014",
    "00021_00022",
    "00036_00042",
    "00018_00050",
    "00051_00052",
    "00055_00048",
    "00002_00015",
    "00002_00004",
    "00027_00031",
    "00027_00011",
    "00010_00047",
    "00005_00016",
    "00042_00036",
    "00022_00060",
    "00022_00021",
    "00028_00025",
    "00052_00051",
    "00052_00054",
    "00049_00040",
    "00031_00027",
    "00012_00013",
    "00024_00026",
    "00058_00043",
    "00040_00049",
    "00040_00033",
    "00032_00033",
    "00029_00013",
    "00029_00047",
    "00060_00022",
    "00047_00029",
    "00047_00010",
    "00054_00052",
    "00015_00002",
    "00048_00019",
    "00048_00055",
    "00033_00032",
    "00033_00040",
    "00043_00058",
    "00016_00035",
    "00016_00005",
    "00050_00018",
    "00025_00028",
    "00019_00048",
    "00011_00027",
    "00013_00012",
    "00013_00029",
    "00014_00004",
]


def get_train_test_subjects():
    train_subjects = set()
    test_subjects = set()
    for pair in TRAIN_GROUP:
        subject1, subject2 = pair.split("_")
        train_subjects.add(subject1)
        train_subjects.add(subject2)

    for pair in TEST_GROUP:
        subject1, subject2 = pair.split("_")
        test_subjects.add(subject1)
        test_subjects.add(subject2)
    return train_subjects, test_subjects
