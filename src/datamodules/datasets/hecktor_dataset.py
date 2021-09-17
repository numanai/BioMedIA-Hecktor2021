import os
from typing import Callable, Optional, Tuple
import sys

import SimpleITK as sitk
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from sklearn.preprocessing import scale

from torchmtlr.utils import make_time_bins, encode_survival


def find_centroid(mask: sitk.Image):

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    centroid_coords = stats.GetCentroid(1)
    centroid_idx = mask.TransformPhysicalPointToIndex(centroid_coords)

    return np.asarray(centroid_idx, dtype=np.float64)

class HecktorDataset(Dataset):

    def __init__(self,
                 root_directory:str, 
                 clinical_data_path:str, 
                 cache_dir:str = "data_cropped/data_cache/",
                 transform: Optional[Callable] = None,
                 num_workers: int = 1
    ):

        self.root_directory = root_directory

        self.transform = transform
        self.num_workers = num_workers

        self.clinical_data = self.make_data(clinical_data_path)
        self.time_bins = make_time_bins(times=self.clinical_data["time"], num_bins=14, event = self.clinical_data["event"])
        self.y = encode_survival(self.clinical_data["time"].values, self.clinical_data["event"].values, self.time_bins) # single event

        self.cache_path = os.path.join(cache_dir)



    def make_data(self, path):

        try:
            X = pd.read_csv(path + '/hecktor2021_patient_info_training.csv')
            y = pd.read_csv(path + '/hecktor2021_patient_endpoint_training.csv')
            df = pd.merge(X, y, on="PatientID")
        except:
            df = path

        clinical_data = df
        clinical_data = clinical_data.rename(columns={"Progression": "event", "Progression free survival": "time", "TNM group":"Stage_group", "Gender (1=M,0=F)":"Gender"})

        clinical_data["Age"] = scale(clinical_data["Age"])

        # binarize T stage as T1/2 = 0, T3/4 = 1
        clinical_data["T-stage"] = clinical_data["T-stage"].map(
            lambda x: "T1/2" if x in ["T1", "T2"] else("Tx" if x == "Tx" else "T3/4"), na_action="ignore")

        # use more fine-grained grouping for N stage
        clinical_data["N-stage"] = clinical_data["N-stage"].str.slice(0, 2)

        clinical_data["Stage_group"] = clinical_data["Stage_group"].map(
            lambda x: "I/II" if x in ["I", "II"] else "III/IV", na_action="ignore")

        clinical_data = pd.get_dummies(clinical_data,
                                    columns=["Gender",
                                                "N-stage",
                                                "M-stage",],
                                    drop_first=True)

        cols_to_drop = [
            #"PatientID",
            "Tobacco",
            "Alcohol",
            "Performance status",
            "HPV status (0=-, 1=+)",
            "Estimated weight (kg) for SUV",
            "M-stage_M1",
            "TNM edition"

        ]

        clinical_data = clinical_data.drop(cols_to_drop, axis=1)


        clinical_data = pd.get_dummies(clinical_data,
                                    columns=["T-stage",
                                                "Stage_group",])
        
        return clinical_data


    


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get an input-target pair from the dataset.

        The images are assumed to be preprocessed and cached.

        Parameters
        ----------
        idx
            The index to retrieve (note: this is not the subject ID).

        Returns
        -------
        tuple of torch.Tensor and int
            The input-target pair.
        """
        
        try:      # training data
            # clin_var_data = self.clinical_data.drop(["target_binary", 'time', 'event', 'Study ID'], axis=1) # single event
            clin_var_data = self.clinical_data.drop(['PatientID','time', 'event'], axis=1)
        except:   # test data
            clin_var_data = self.clinical_data.drop(['PatientID'], axis=1)


        clin_var = clin_var_data.iloc[idx].to_numpy(dtype='float32')
        
        target = self.y[idx]
        
        labels = self.clinical_data.iloc[idx].to_dict()
 
        
        subject_id = self.clinical_data.iloc[idx]["PatientID"]
        path = os.path.join(self.cache_path, f"{subject_id}.nii")
#         print('hi:', path)
        
        image = sitk.ReadImage(path)
        if self.transform is not None:
            image = self.transform(image)
    
        return (image, clin_var), target, labels

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.clinical_data)
