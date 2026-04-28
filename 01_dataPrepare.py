import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import xarray as xr
from tqdm.autonotebook import tqdm


# nc
def readNc(input_path):
    cn = xr.open_dataset(input_path)
    return cn


def data_path(filename):
    file_path = "{path}/{filename}".format(
        path="H:/paper/resilience/data",
        filename=filename
    )
    return file_path


def read_data(path):
    data = np.load(path)
    print(path, 'read data')
    return data


# Tropics
lat = [180, 240]   # 45S-15N
lon = [120, 1440]  # 150W-180E

features = ["SMsurf", "SMroot"]


if __name__ == '__main__':
    os.chdir(r"H:\paper\01_soilMoistureSensitivity_2023\Data")
    # soil moisture
    smr = readNc("variables/SMroot_1980-2022_GLEAM_v3.8a_YR.nc")   # (43, 1440, 720)
    smf = readNc("variables/SMsurf_1980-2022_GLEAM_v3.8a_YR.nc")   # (43, 1440, 720)
    # tree cover
    tc = readNc("vegetation/VCF_Treecover_1982_2016_annual_0d25tif.nc")   # (33, 1440, 720)


    smr_ = smr.sel(time=slice('1982-12-31','2016-12-31')).where((smr['time.year']!=1994) & (smr['time.year']!=2000), drop=True)
    smf_ = smf.sel(time=slice('1982-12-31','2016-12-31')).where((smr['time.year']!=1994) & (smr['time.year']!=2000), drop=True)

    list = [tc['treecover'].values, smf_['SMsurf'].values, smr_['SMroot'].values]
    Yearly_input_data = np.stack(list, axis=1)    # (33, 3, 720, 1440)


    np.save('MethodUse/TreeCover_SMsurf_SMroot_yearly_1982_2016_0d25', Yearly_input_data)


    print("prepare done")

