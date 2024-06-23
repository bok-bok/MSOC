import os

import pandas as pd


def split_new_dataset(df, test_subset="all"):

    train_df_A = df[df["category"] == "A"].sample(350, random_state=42)
    train_df_B = df[df["category"] == "B"].sample(350, random_state=42)
    train_df_C = (
        df[(df["category"] == "C") & (df["method"] != "faceswap")]
        .groupby("method")
        .sample(n=175, random_state=42)
    )
    train_df_D = (
        df[(df["category"] == "D") & (df["method"] != "faceswap-wav2lip")]
        .groupby("method")
        .sample(n=175, random_state=42)
    )

    train_metadata = pd.concat([train_df_A, train_df_B, train_df_C, train_df_D])

    val_df_A = df[df["category"] == "A"].drop(train_df_A.index)[:50]

    val_df_B = df[df["category"] == "B"].drop(train_df_B.index)[:50]
    val_df_C = (
        df[(df["category"] == "C") & (df["method"] != "faceswap")]
        .drop(train_df_C.index)
        .groupby("method")
        .sample(n=25, random_state=42)
    )
    val_df_D = (
        df[(df["category"] == "D") & (df["method"] != "faceswap-wav2lip")]
        .drop(train_df_D.index)
        .groupby("method")
        .sample(n=25, random_state=42)
    )

    val_metadata = pd.concat([val_df_A, val_df_B, val_df_C, val_df_D])

    test_df_A = df[df["category"] == "A"].drop(train_df_A.index).drop(val_df_A.index)
    if test_subset == "all":
        test_df_C = df[(df["category"] == "C") & (df["method"] == "faceswap")].sample(n=100, random_state=42)
        test_df_D = df[(df["category"] == "D") & (df["method"] == "faceswap-wav2lip")].sample(
            n=100, random_state=42
        )
        test_df_E = df[df["category"] == "E"].sample(n=100, random_state=42)
        test_df_F = df[df["category"] == "F"].sample(n=100, random_state=42)

        test_metadata = pd.concat([test_df_A, test_df_C, test_df_D, test_df_E, test_df_F])
        # self.test_metadata = pd.concat([test_df_A, test_df_C, test_df_D])
    else:
        if test_subset == "C":
            test_subset = df[(df["category"] == "C") & (df["method"] == "faceswap")].sample(
                n=100, random_state=42
            )
        elif test_subset == "D":
            test_subset = df[(df["category"] == "D") & (df["method"] == "faceswap-wav2lip")].sample(
                n=100, random_state=42
            )
        else:
            test_subset = df[df["category"] == test_subset].sample(n=100, random_state=42)
        test_metadata = pd.concat([test_df_A, test_subset])

    print(len(train_metadata), len(val_metadata), len(test_metadata))
    return train_metadata, val_metadata, test_metadata


def get_subject(row):
    method_column_mapping = {
        "wav2lip": "source",
        "rtvc": "source",
        # "faceswap": "target1",
        "fsgan": "target1",
        "faceswap-wav2lip": "target1",
        "fsgan-wav2lip": "target1",
        "real": "source",
        "freevc": "source",
        "shifted": "source",
    }
    if row["method"] != "faceswap":
        return row[method_column_mapping[row["method"]]]
    else:
        if "id" in row["target1"]:
            return row["target1"]
        else:
            return row["source"]
