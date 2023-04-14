import numpy as np
import os
import pydicom

from glob import glob
from tqdm import tqdm

datadir = "/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/vsouzaribeiro/datasets/ArtSpeech_Database_2"

dcm_filepaths = sorted(glob(os.path.join(datadir, "*", "S*", "*.dcm")))

for fp in dcm_filepaths:
    dirname = os.path.dirname(fp)
    sequence = os.path.basename(dirname)
    subject = os.path.basename(os.path.dirname(dirname))
    np_dirname = os.path.join(dirname, "NPY_MR")
    if not os.path.exists(np_dirname):
        os.makedirs(np_dirname)

    ds = pydicom.dcmread(fp)
    for i, frame_array in tqdm(
        enumerate(ds.pixel_array, start=1),
        desc=f"{subject}/{sequence}",
        total=len(ds.pixel_array)
    ):
        filepath = os.path.join(np_dirname, "%04d.npy" % i)
        np.save(filepath, frame_array)
