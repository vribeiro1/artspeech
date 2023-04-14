import funcy
import numpy as np
import os

from glob import glob
from tqdm import tqdm
from vt_tools import (
    ARYTENOID_CARTILAGE,
    EPIGLOTTIS,
    LOWER_INCISOR,
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE_MIDLINE,
    THYROID_CARTILAGE,
    TONGUE,
    UPPER_INCISOR,
    UPPER_LIP,
    VOCAL_FOLDS
)
from vt_shape_gen.vocal_tract_tube import generate_vocal_tract_tube

from settings import BASE_DIR, DATASET_CONFIG
from vocal_tract_loader import VocalTractShapeLoader

ARTICULATORS = [
    ARYTENOID_CARTILAGE,
    EPIGLOTTIS,
    LOWER_INCISOR,
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE_MIDLINE,
    THYROID_CARTILAGE,
    TONGUE,
    UPPER_INCISOR,
    UPPER_LIP,
    VOCAL_FOLDS
]


def main(database_name, datadir, overwrite=True):
    dataset_config = DATASET_CONFIG[database_name]
    vocal_tract_loader = VocalTractShapeLoader(
        datadir=datadir,
        articulators=ARTICULATORS,
        num_samples=50,
        dataset_config=dataset_config
    )

    sequences_dirs = sorted(filter(os.path.isdir, glob(os.path.join(datadir, "*", "*"))))
    for sequence_dir in sequences_dirs:
        sequence = os.path.basename(sequence_dir)
        subject = os.path.basename(os.path.dirname(sequence_dir))

        all_filepaths = glob(os.path.join(sequence_dir, "inference_contours", "*_*.npy"))
        all_basenames = funcy.lmap(os.path.basename, all_filepaths)
        all_frame_ids = sorted(set(map(lambda s: s.split("_")[0], all_basenames)))

        save_dir = os.path.join(sequence_dir, "air_column")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for frame_id in tqdm(all_frame_ids, desc=f"{subject}/{sequence}"):
            if not overwrite and os.path.isfile(os.path.isfile(os.path.join(save_dir, f"{frame_id}.npy"))):
                continue

            articulators_filepaths = glob(os.path.join(sequence_dir, "inference_contours", f"{frame_id}_*.npy"))
            articulators_dict = {}
            for filepath in articulators_filepaths:
                basename = os.path.basename(filepath)
                filename, _ = basename.split(".")
                _, articulator_name = filename.split("_")
                articulators_dict[articulator_name] = filepath

            if not all([articulator in articulators_dict for articulator in ARTICULATORS]):
                continue

            internal_wall, external_wall = generate_vocal_tract_tube(
                articulators_dict,
                norm_value=dataset_config.RES
            )
            air_column = np.array([internal_wall.T, external_wall.T])
            filepath = os.path.join(save_dir, f"{frame_id}.npy")
            np.save(filepath, air_column)


if __name__ == "__main__":
    database_name = "artspeech2"
    datadir = "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/vsouzaribeiro/datasets/ArtSpeech_Database_2"
    main(database_name, datadir, overwrite=False)