import os
import numpy as np
import sys
# Drawing plots is not supported in sensi.yml environmental
import matplotlib
import matplotlib.pyplot as plt
import statsmodels
import cartopy.crs as ccrs
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import shap
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import regressors
import regressors.stats as regressors_stats
import time
import warnings
warnings.filterwarnings('ignore')


#===================function============================================
# lat = 360
# lon = 720
# year1 = 0  # 1982
# year2 = 36  # 2017
# year = 36

# tree cover----------
# year1 = 0  # 1982
# year2 = 33  # 2016
# year = 33

# vod woody cover--------
year1 = 0  # 1993
year2 = 19  # 2012
year = 19

features = ['smf', 'smr']


def data_path(filename):
    file_path = "{path}/{filename}".format(
        path="/data1/usersdir/zyb/02_soilMoistureSensitivity_2023/Data",
        filename=filename
    )
    return file_path


def read_data(path):
    data = np.load(path)
    print(path, 'read data')
    return data


def graph(ax, target, **kwargs):
    lon = np.arange(-180, 180, 0.5)
    lat = np.arange(-90, 90, 0.5)
    ax.coastlines(linewidth=0.2)
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    cs = ax.pcolor(lon, lat, target[::-1, :], transform=ccrs.PlateCarree(), **kwargs)
    cbar = ax.figure.colorbar(cs, ax=ax, shrink=0.4, extend='both')
    return (cs)


# 1440-120=1320
def RF_longterm(block, block1, block2):
    output = np.zeros((3, 12, 1440),
                      dtype=np.float32) * np.nan  # first dimension: results of slope and p_value and R2; second dimension: 4 soil moisture variables; third dimension: longtitude

    # iterate over data across longtitudes
    # tropics
    lon_new = [120, 1440]  # 150W-180E from left top

    for col in range(lon_new[0] + 2, lon_new[1] - 2, 1):

        if np.isnan(block[:, :, col]).any() or np.isinf(block[:, :, col]).any() or np.all(
                block[:, :, col] == 0):
            output[:, :, col] = np.nan
        else:
            # 5x5 grid cells need to be used in one random forest model
            test_data1 = block1[:, :, col - 1]
            test_data2 = block[:, :, col - 1]
            test_data3 = block2[:, :, col - 1]
            test_data4 = block1[:, :, col]
            test_data5 = block[:, :, col]
            test_data6 = block2[:, :, col]
            test_data7 = block1[:, :, col + 1]
            test_data8 = block[:, :, col + 1]
            test_data9 = block2[:, :, col + 1]
            test_data10 = block1[:, :, col - 2]
            test_data11 = block[:, :, col - 2]
            test_data12 = block2[:, :, col - 2]
            test_data13 = block1[:, :, col + 2]
            test_data14 = block[:, :, col + 2]
            test_data15 = block2[:, :, col + 2]
            test_data16 = np.concatenate((test_data1, test_data2, test_data3, test_data4,
                                          test_data5, test_data6, test_data7, test_data8, test_data9,
                                          test_data10,test_data11, test_data12,test_data13,test_data14,
                                          test_data15))
            test_data = test_data16[~np.all(test_data16 == 0, axis=1)]
            test_data = test_data[~np.any(np.isnan(test_data), axis=1)]

            if len(test_data[:,
                   0]) >= 50:  # maximum data points can be 15 grid cells  * 20 years without considering growing seasons and data gaps
                # set common hyperparameters to train a random forest model

                max_component = 6
                mse = []
                n_components_range = range(1, max_component+1)
                for n in n_components_range:
                    pls = PLSRegression(n_components=n)
                    scores = cross_val_score(pls,test_data[:, 1:], test_data[:, 0], cv=5, scoring=make_scorer(mean_squared_error))
                    mse.append(np.mean(scores))
                optimal_n_component = n_components_range[np.argmin(mse)]

                estimator = PLSRegression(n_components=optimal_n_component)
                estimator = estimator.fit(test_data[:, 1:], test_data[:, 0])  # (SMroot)  (Treecover)
                scores = cross_val_score(estimator,test_data[:, 1:], test_data[:, 0], cv=5, scoring='r2')
                # print(np.nanmean(scores))

                for v in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                    print(estimator.coef_.flatten()[v])
                    output[0, v, col] = estimator.coef_.flatten()[v]
                    output[1, v, col] = regressors_stats.coef_pval(estimator, test_data[:, 1:], test_data[:, 0])[v+1]  # p_value
                    output[2, v, col] = np.nanmean(scores)
                    print('oob:', output[2, v, col], 'sensitivity:', output[0, v, col], 'p_value:', output[1, v, col])



                # if True:  # set a basic threshold of model performance
                #     explainer = shap.TreeExplainer(rf)
                #     shap_values = explainer.shap_values(test_data[:, 1:6])
                #
                #     for v in [0, 1, 2, 3, 4]:  # calculate SHAP values of treecover to soil moisture
                #         y = shap_values[:, v]
                #         x = test_data[:, v + 1]
                #         x = x.reshape(-1, 1)
                #         x_new = x[~np.isnan(y)]
                #         y_new = y[~np.isnan(y)]
                #         if len(y_new) >= 5 and ~np.isnan(x_new).any() and ~np.isinf(x_new).any() and ~np.all(
                #                 x_new == 0):
                #             tel = TheilSenRegressor().fit(x_new, y_new)
                #             output[0, v, col] = tel.coef_  # Theil-sen slope
                #             output[1, v, col] = regressors_stats.coef_pval(tel, x_new, y_new)[1]  # p_value
                #             print('oob:', rf.oob_score_, 'sensitivity:', output[0, v, col], 'p_value:',
                #                   output[1, v, col])

    return (output)

def RF_3yblock(block, block1, block2):


    output = np.zeros((year, 12, 1440), dtype=np.float32) * np.nan

    # iterate over data across longtitudes
    # tropics
    lon_new = [120, 1440]  # 150W-180E from left top






    for col in range(lon_new[0] + 2, lon_new[1] - 2, 1):
        window = 5
        for move in range(year1, year2, 1):
            if move+window > year:
                break
            # 5x5 grid cells need to be used in one random forest model
            test_data1 = block1[move:move+window, :, col - 1]
            test_data2 = block[move:move+window, :, col - 1]
            test_data3 = block2[move:move+window, :, col - 1]
            test_data4 = block1[move:move+window, :, col]
            test_data5 = block[move:move+window, :, col]
            test_data6 = block2[move:move+window, :, col]
            test_data7 = block1[move:move+window, :, col + 1]
            test_data8 = block[move:move+window, :, col + 1]
            test_data9 = block2[move:move+window, :, col + 1]
            test_data10 = block1[move:move + window, :, col + 2]
            test_data11 = block[move:move + window, :, col + 2]
            test_data12 = block2[move:move + window, :, col + 2]
            test_data13 = block1[move:move + window, :, col - 2]
            test_data14 = block[move:move + window, :, col - 2]
            test_data15 = block2[move:move + window, :, col - 2]


            test_data16 = np.concatenate((test_data1, test_data2, test_data3, test_data4,
                                          test_data5, test_data6, test_data7, test_data8, test_data9,
                                          test_data10, test_data11, test_data12, test_data13, test_data14,
                                          test_data15))
            test_data = test_data16[~np.all(test_data16 == 0, axis=1)]
            test_data = test_data[~np.any(np.isnan(test_data), axis=1)]

            if np.isnan(test_data).any() or np.isinf(test_data).any() or np.all(test_data == 0):
                output[:, :, col] = np.nan
            else:
                if len(test_data[:,
                       0]) >= 15:  # maximum data points can be 25 grid cells * 3 years without considering growing seasons and data gaps
                    # set common hyperparameters to train a random forest model
                    max_component = 6
                    mse = []
                    n_components_range = range(1, max_component + 1)
                    for n in n_components_range:
                        pls = PLSRegression(n_components=n)
                        scores = cross_val_score(pls, test_data[:, 1:], test_data[:, 0], cv=5,
                                                 scoring=make_scorer(mean_squared_error))
                        mse.append(np.mean(scores))
                    optimal_n_component = n_components_range[np.argmin(mse)]

                    estimator = PLSRegression(n_components=optimal_n_component)
                    estimator = estimator.fit(test_data[:, 1:], test_data[:, 0])  # (SMroot)  (Treecover)
                    scores = cross_val_score(estimator, test_data[:, 1:], test_data[:, 0], cv=5, scoring='r2')
                    scores = np.nanmean(scores)

                    if scores > 0:  # set a basic threshold of model performance

                        for v in [0, 1, 2, 3, 4, 5,6,7,8,9,10,11]:  # only calculate SHAP values of LAI to sub-surface soil moisture
                            output[move, v, col] = estimator.coef_.flatten()[v]
                            pval = regressors_stats.coef_pval(estimator, test_data[:, 1:], test_data[:, 0])[v + 1]  # p_value
                            if pval >= 0.1 or pval < 0:
                                output[move, v, col] = np.nan
                            print('oob:', scores, 'sensitivity:', output[move, v, col])

                            # y = shap_values[:, v]
                            # x = test_data[:, v + 1]
                            # x = x.reshape(-1, 1)
                            # x_new = x[~np.isnan(y)]
                            # y_new = y[~np.isnan(y)]
                            # if len(y_new) >= 5 and ~np.isnan(x_new).any() and ~np.isinf(x_new).any() and ~np.all(
                            #         x_new == 0):
                            #     tel = TheilSenRegressor().fit(x_new, y_new)
                            #     output[move, v, col] = np.round(tel.coef_, 7)
                            #     pvalue = np.round(regressors_stats.coef_pval(tel, x_new, y_new)[1], 7)
                            #     if pvalue >= 0.1 or pvalue < 0:  # significance test
                            #         output[move, v, col] = np.nan
                            # print('oob:', rf.oob_score_, 'sensitivity:', output[move, v, col])

    return (output)

if __name__ == '__main__':

    # treecover
    # input_data = read_data(data_path("3MethodUse/TreeCover_SMsurf_SMroot_yearly_1982_2016_0d25.npy"))
    # vod woody cover
    # input_data = read_data(data_path("3MethodUse/VODwoodyCover_SMsurf_SMroot_yearly_1982_2016_0d25.npy"))

    # vod woody cover
    # input_data = read_data(data_path("3MethodUse/VODwoodyCover_GLDAS_CHIRPS_yearly_1993_2012_0d25_noNormalized.npy"))
    # input_data = read_data(data_path("3MethodUse/woodycover_input_1993_2012.npy"))
    # print(input_data.shape)    # (20,6,720,1440)

    # vod woody cover
    input_data = read_data("/media/User/Expansion/zyb/SM/model5.npy")
    print(input_data.shape)



    '''
    # Long term sensitivity============================================================
    # =================================================================================
    # =================================================================================
    start_time = time.time()
    # set CPU usage
    pool1 = Pool(35)

    # data-preparation for multiprocessing: iterate over data across latitudes for 3 rows (in the end, 3x3 grid cells need to be used in one random forest model.)
    # tropics
    lat_new = [300, 540]  # 45S-15N from left top

    block = [input_data[:, :, row, :] for row in range(lat_new[0] + 1, lat_new[1] - 1, 1)]
    block1 = [input_data[:, :, row, :] for row in range(lat_new[0] + 0, lat_new[1] - 2, 1)]
    block2 = [input_data[:, :, row, :] for row in range(lat_new[0] + 2, lat_new[1], 1)]

    # # multiprocessing of function RF_longterm
    outs = pool1.map(RF_longterm, block, block1, block2)
    pool1.close()
    pool1.join()
    print('shape(outs):', np.shape(outs))

    SHAP_all = np.zeros((3, 12, 720, 1440)) * np.nan
    # upload results across latitudes
    for part in range(1, lat_new[1] - lat_new[0] - 1, 1):
        SHAP_all[:, :, lat_new[0] + part, :] = outs[part - 1]
    # #
    # # # tree cover
    # # # np.save(data_path('4Result/Treecover_SMsurf_SMroot_yearly_1982_2016_0d25_ContributionSensitivity_allYear'), SHAP_all)
    # # # vod woody cover
    # # # np.save(data_path('4Result/VODwoodycover_SMsurf_SMroot_yearly_1993_2012_0d25_ContributionSensitivity_allYear'), SHAP_all)
    # #
    # #     # vod woody cover
    # # # np.save(data_path('4Result/VODwoodycover_GLDAS_CHIRPS_yearly_1993_2012_0d25_ContributionSensitivity_allYear_noNormalized'),
    # # #             SHAP_all)
    # #
    np.save(
        '/data1/usersdir/zyb/02_soilMoistureSensitivity_2023/20241224/woodycover_yearly_1993_2012_0d25_ContributionSensitivity_allYear_normalized_PLS_model5',
            SHAP_all)
    print("35 cpus--- %s seconds ---" % (time.time() - start_time))
    '''

    # per 3-year window================================================================
    # =================================================================================
    # =================================================================================
    start_time = time.time()
    # set CPU usage
    pool = Pool(35)

    # data-preparation for multiprocessing: iterate over data across latitudes for 3 rows (in the end, 3x3 grid cells need to be used in one random forest model.)
    # tropics
    lat_new = [300, 540]  # 45S-15N from left top

    block = [input_data[:, :, row, :] for row in range(lat_new[0] + 1, lat_new[1] - 1, 1)]
    block1 = [input_data[:, :, row, :] for row in range(lat_new[0] + 0, lat_new[1] - 2, 1)]
    block2 = [input_data[:, :, row, :] for row in range(lat_new[0] + 2, lat_new[1], 1)]

    # # multiprocessing of function RF_longterm
    outs = pool.map(RF_3yblock, block, block1, block2)
    pool.close()
    pool.join()
    print('shape(outs):', np.shape(outs))



    SHAP_3Y = np.zeros((year, 12, 720, 1440)) * np.nan
    # upload results across latitudes
    for part in range(1, lat_new[1] - lat_new[0] - 1, 1):
        SHAP_3Y[:, :, lat_new[0] + part, :] = outs[part - 1]


    np.save('/data1/usersdir/zyb/02_soilMoistureSensitivity_2023/20241224/woodycover_yearly_1993_2012_0d25_ContributionSensitivity_5YearWindows_Normalized_PLS_model5',
            SHAP_3Y)
    print("35 cpus--- %s seconds ---" % (time.time() - start_time))

    # python /data1/usersdir/zyb/02_soilMoistureSensitivity_2023/code/02_Tc_SM_sensitivity_PLS.py
    
