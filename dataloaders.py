from datasets.cimat import prepare_dataloaders as prepare_cimat_dataloaders
from datasets.krestenitis import prepare_dataloaders as prepare_krestenitis_dataloaders
from datasets.sos import prepare_dataloaders as prepare_sos_dataloaders
from datasets.chn6_cug import prepare_dataloaders as prepare_chn6_dataloaders


def get_dataloaders(base_dir, name, args=None):
    """
    Available datasets:
    * Cimat (Envisat-Sentinel Oil Spill Detection)
    - Parameters:
      - Dataset: num of dataset (17, 19, 20)
      - Trainset: num of train-val-test (01-30)
      - Channels: channels configuracion (oov, owv)
    * Krestenitis
    * SOS
    * CHN6-CUG
    * CHASE-DB1
    * DRIVE
    * STARE
    """
    if name == "cimat":
        if (args == None) or not (
            ("dataset" in args) and ("trainset" in args) and ("channels" in args)
        ):
            raise Exception("Faltan argumentos para el dataset cimat")
        return prepare_cimat_dataloaders(
            base_dir, args["dataset"], args["trainset"], args["channels"]
        )
    if name == "krestenitis":
        return prepare_krestenitis_dataloaders(base_dir)
    if name == "sos":
        return prepare_sos_dataloaders(base_dir)
    if name == "chn6-cug":
        return prepare_chn6_dataloaders(base_dir)
    # To be implemented
    raise Exception(f"Dataset {name} it's currently not implemented!")
