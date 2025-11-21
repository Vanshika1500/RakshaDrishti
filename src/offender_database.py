# src/offender_database.py

import os
import numpy as np
from src.models.face_model import get_embedding_from_path

def load_offender_database(folder=None):
    """
    Reads offenders/<name>/*.jpg and creates dict: {name: avg_embedding}
    """

    # Always load offenders folder from same directory as this file
    if folder is None:
        folder = os.path.join(os.path.dirname(__file__), "offenders")

    db = {}

    if not os.path.exists(folder):
        print(" Offender folder missing!", folder)
        return db

    for person in os.listdir(folder):
        person_path = os.path.join(folder, person)
        if not os.path.isdir(person_path):
            continue

        embeddings = []
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            emb = get_embedding_from_path(img_path)
            if emb is not None:
                embeddings.append(emb)

        if embeddings:
            db[person] = np.mean(embeddings, axis=0)
            print(f" Loaded {person} ({len(embeddings)} images)")

    return db
