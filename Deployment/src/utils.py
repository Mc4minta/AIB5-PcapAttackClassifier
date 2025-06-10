# define function for dataframe preprocessing

import pandas as pd
import numpy as np
import ast

from collections import Counter

def map_port(port):
    if port == 21:
        return 1  # FTP
    elif port == 22:
        return 2  # SSH
    elif port == 53:
        return 3  # DNS
    elif port == 80:
        return 4  # HTTP
    elif port == 443:
        return 5  # HTTPS
    else:
        return 6  # Other

def preprocess_dataframe(df):
    original_indices = set(df.index)

    # replace space in columns name with underscore
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    # drop objects type columns
    columns_to_drop = [
        'Flow_ID','Src_IP','Dst_IP','Src_Port','Protocol','Timestamp','Label'
    ]

    df = df.drop(columns=columns_to_drop)

    # remove rows with missing and infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()

    # map destination port to 1-6 numbers
    df['Dst_Port'] = df['Dst_Port'].apply(map_port)

    return df

# preprocess_dataframe(df)

def choose_label(labels):
    if all(label == 'Benign' for label in labels):
        return 'Benign'

    # Count non-Benign labels
    non_benign_labels = [label for label in labels if label != 'Benign']
    label_counts = Counter(non_benign_labels)

    # Return the most common non-Benign label
    most_common_label, _ = label_counts.most_common(1)[0]
    return most_common_label