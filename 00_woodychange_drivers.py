import numpy as np
import pandas
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from osgeo import gdal
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import statsmodels.api as sm

# 设置随机种子以确保可重复性
np.random.seed(42)


# def prepare_data(data_path):
#     """
#     准备数据,返回特征矩阵X和目标变量y
#     """
#     # 加载数据
#     data = pd.read_csv(data_path)
# 
#     # 定义特征列
#     feature_columns = [
#         'NSR', 'Height', 'Elevation', 'PRE', 'Root_depth',
#         'Elevation_cv', 'Slope', 'Aspect', 'Slope_cv', 'Aspect_cv',
#         'DI', 'Tree_density', 'AWC', 'Soil_sand', 'Soil_clay'
#     ]
# 
#     # 准备特征矩阵和目标变量
#     X = data[feature_columns]
#     y = data['NDVI_change']  # 目标变量
# 
#     return X, y


def train_xgboost_model(X, y, test_size=0.3):
    """
    训练XGBoost模型
    """
    # 划分训练集和测试集
    # n_samples = len(X)
    # indices = np.random.permutation(n_samples)
    # train_size = int((1 - test_size) * n_samples)
    #
    # train_idx = indices[:train_size]
    # test_idx = indices[train_size:]
    #
    # X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    # y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 定义XGBoost模型参数
    params = {
        'max_depth': 10,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'objective': 'reg:squarederror',
        'random_state': 42
    }

    param_grid = {
        'max_depth': [7, 10, 15],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [950, 1000, 1200, 1500],
        'objective': ['reg:squarederror']
    }

    # 训练模型
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    print("feature importance:")
    print(model.feature_importances_)

    # model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # model.fit(X_train, y_train)
    # print("feature importance:")
    # print(model.feature_importances_)


    # model = xgb.XGBRegressor(random_state=42)
    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    # grid_search.fit(X_train, y_train)
    # best_model = grid_search.best_estimator_
    # print("best model", grid_search.best_params_)


    return model, X_train, X_test, y_train, y_test


def calculate_shap_values(model, X):
    """
    计算SHAP值
    """
    # monkey patch
    # booster = model.get_booster()
    # model_bytearray = booster.save_raw()[4:]
    # booster.save_raw = lambda: model_bytearray


    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # print(shap_values[1])
    # shap.summary_plot(shap_values, X)
    plt.figure(figsize=(5, 4))
    shap.dependence_plot('ΔVPD', shap_values, X, interaction_index='Burn area', cmap='Reds', show=False)
    plt.savefig(r'E:\research\01SM_woodycover\00progress\figures\new\FigS13a.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    # plt.axhline(y=0, color='black',linestyle='-.', linewidth=1)
    # ax = plt.gca()
    # ax.spines['top'].set_visible(True)
    # ax.spines['right'].set_visible(True)
    plt.figure(figsize=(5, 4))
    shap.dependence_plot('ΔVPD', shap_values, X, interaction_index='ΔRD', cmap='RdBu', show=False)
    plt.savefig(r'E:\research\01SM_woodycover\00progress\figures\new\FigS13b.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(5, 4))
    shap.dependence_plot('ΔVPD', shap_values, X, interaction_index='Root depth', cmap='Greens', show=False)
    plt.savefig(r'E:\research\01SM_woodycover\00progress\figures\new\FigS13c.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    # shap.dependence_plot('ΔVPD', shap_values, X, interaction_index='Burn area', cmap='Reds')
    plt.figure(figsize=(5, 4))
    shap.dependence_plot('ΔVPD', shap_values, X, interaction_index='AI', cmap='OrRd', show=False)
    plt.savefig(r'E:\research\01SM_woodycover\00progress\figures\new\FigS13d.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    # shap.dependence_plot('ΔVPD', shap_values, X, interaction_index='Dem', cmap='OrRd')
    # shap.dependence_plot('ΔVPD', shap_values, X, interaction_index='ΔSM', cmap='RdBu')
    # shap.dependence_plot('ΔSM', shap_values, X, interaction_index='Root depth', cmap='Greens')
    # shap.dependence_plot('ΔSM', shap_values, X, interaction_index='ΔPI', cmap='RdBu')
    # shap.dependence_plot('ΔSM', shap_values, X, interaction_index='Burn area', cmap='Reds')

    return shap_values, explainer


def create_shap_visualization(bar_values, bar_names, scatter_height, height_shap,
                              scatter_elevation, elevation_shap, scatter_3, shap_3, scatter_4, shap_4
                              , max_feature, second_feature, third_feature, forth_feature):

    # 创建图表和子图
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(2, 3)

    font = {
        'family': 'sans-serif',
        'sans-serif': 'Arial',
        'weight': 'normal',
        'size': 12
    }
    plt.rc('font', **font)  # pass in the font dict as kwargs

    # 左侧柱状图
    ax1 = fig.add_subplot(gs[:, 0])

    # 定义颜色映射
    # colors = ['#4169E1', '#4169E1', '#4169E1', '#4169E1', '#4169E1', '#4169E1', '#4169E1',
    #           '#90EE90', 'red', '#D2691E', '#D2691E', '#D2691E',
    #           '#D2691E', '#808080', '#808080', '#808080', '#808080']

    colors = ['#2B6CB0', '#2B6CB0', '#2B6CB0', '#2B6CB0', '#2B6CB0', '#2B6CB0', '#2B6CB0',
              '#1A9850', '#D73027', '#A6611A', '#A6611A', '#A6611A',
              '#A6611A', '#878787', '#878787', '#878787', '#878787']

    # 对数据进行排序，同时保持颜色对应关系
    sorted_indices = np.argsort(bar_values) # 从大到小排序的索引
    sorted_values = np.array(bar_values)[sorted_indices]
    sorted_names = np.array(bar_names)[sorted_indices]
    sorted_colors = np.array(colors)[sorted_indices]

    # 绘制水平柱状图
    y_pos = np.arange(len(sorted_names))
    ax1.barh(y_pos, sorted_values, color=sorted_colors, edgecolor='black')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sorted_names)
    ax1.set_xlabel('Mean (|SHAP| value)')

    ax1.text(0.6, 8.5, 'R$^2$: 0.844\nMAE: 0.778\nMSE: 1.192', color='black')

    # 添加饼图

    # [smr, srad, vpd, pi, pd, AI, root, ba, sand, clay, oc, awc, dem, demcv, slope, slopecv]
    print("variable numbers:"+ str(len(bar_values)))
    pie_data = [np.sum(bar_values[:7]), np.sum(bar_values[7]), np.sum(bar_values[8]), np.sum(bar_values[9:13]), np.sum(bar_values[13:])]   # 数值
    pie_labels = ['Climate', 'Vegetation', 'Fire', 'Topography', 'Soil']
    pie_colors = ['#2C7BB6', '#1A9850', '#D73027', '#878787', '#A6611A']

    # 在柱状图内添加小饼图
    ax_pie = fig.add_axes([0.02, 0.15, 0.45, 0.45])  # 调整位置和大小
    pie_wedge_collection = ax_pie.pie(pie_data, labels=pie_labels, colors=pie_colors,
               autopct='%1.1f%%', textprops={'fontsize': 12})
    for j, pie_wedge in enumerate(pie_wedge_collection[0]):
        pie_wedge.set_edgecolor('black')
        pie_wedge.set_linewidth(0.8)
        pie_wedge.set_alpha(1)

    # wedges, texts, autotexts = ax_pie.pie(pie_data, labels=pie_labels, colors=pie_colors,
    #            autopct='%1.1f%%', textprops={'fontsize': 12})
    #
    # for text in texts:
    #     text.set_position((text.get_position()[0], text.get_position()[1] + 0.1))
    # for autotext in autotexts:
    #     autotext.set_position((autotext.get_position()[0], autotext.get_position()[1] + 0.1))
    # ax_pie.legend(wedges, pie_labels, loc='center left', bbox_to_anchor=(1,0,0.5,1))

    # 中间散点图1
    ax2 = fig.add_subplot(gs[0, 1])
    # ax2.scatter(scatter_height, height_shap, color=sorted_colors[-1], alpha=0.5, s=20)
    bin_edges = np.arange(np.nanmin(scatter_height), np.nanmax(scatter_height), 0.02)
    bin_centers = []
    y_means = []
    y_errors = []
    for i in range(len(bin_edges) - 1):
        bin_mask = (scatter_height >= bin_edges[i]) & (scatter_height < bin_edges[i + 1])
        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        bin_centers.append(bin_center)
        y_means.append(np.nanmean(height_shap[bin_mask]))
        y_errors.append(np.nanstd(height_shap[bin_mask]))

    ax2.errorbar(bin_centers, y_means, yerr=y_errors, fmt='o', capsize=0, label='Errorbar', markerfacecolor=sorted_colors[-1], markeredgecolor='black', ecolor='black')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    # 添加平滑曲线
    # z = np.polyfit(scatter_height, height_shap, 2)
    # p = np.poly1d(z)
    # x_smooth = np.linspace(min(scatter_height), max(scatter_height), 100)
    # y_fit = p(x_smooth)
    # y_pred = p (scatter_height)
    # residuals = height_shap- y_pred
    # std_err = np.nanstd(residuals)
    # t_value = 1.984
    # y_err = std_err * np.sqrt(1/ len(scatter_height) + (x_smooth- np.nanmean(scatter_height))**2/np.nansum((scatter_height-np.nanmean(scatter_height))**2))
    # y_upper = y_fit + t_value * y_err
    # y_lower = y_fit - t_value * y_err
    # ax2.plot(x_smooth, p(x_smooth), color='red', alpha=0.8)
    # ax2.fill_between(x_smooth, y_lower, y_upper, color='red', alpha=0.4)
    # sns.regplot(x=scatter_height, y=height_shap, scatter=False,
    #             color='red', ax=ax2)

    ax2.set_xlabel(max_feature)
    ax2.set_ylabel('SHAP value for ' + max_feature)
    ax2.grid(True, alpha=0.2)

    # 右侧散点图1
    ax3 = fig.add_subplot(gs[0, 2])
    # ax3.scatter(scatter_elevation, elevation_shap, color=sorted_colors[-2], alpha=0.3, s=10)
    bin_edges = np.arange(np.nanmin(scatter_elevation), np.nanmax(scatter_elevation), 0.05)
    bin_centers = []
    y_means = []
    y_errors = []
    for i in range(len(bin_edges)-1):
        bin_mask = (scatter_elevation >= bin_edges[i])&(scatter_elevation< bin_edges[i+1])
        bin_center = (bin_edges[i]+bin_edges[i+1])/2
        bin_centers.append(bin_center)
        y_means.append(np.nanmean(elevation_shap[bin_mask]))
        y_errors.append(np.nanstd(elevation_shap[bin_mask]))

    ax3.errorbar(bin_centers, y_means, yerr=y_errors, fmt='o', capsize=0, label='Errorbar', markerfacecolor=sorted_colors[-2], markeredgecolor='black', ecolor='black')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    # 添加平滑曲线和置信区间
    # z = np.polyfit(scatter_elevation, elevation_shap, 2)
    # p = np.poly1d(z)
    # x_smooth = np.linspace(min(scatter_elevation), max(scatter_elevation), 100)
    # y_fit = p(x_smooth)
    # y_pred = p(scatter_elevation)
    # residuals = elevation_shap - y_pred
    # std_err = np.nanstd(residuals)
    # t_value = 1.984
    # y_err = std_err * np.sqrt(1 / len(scatter_elevation) + (x_smooth - np.nanmean(scatter_elevation)) ** 2 / np.nansum(
    #     (scatter_elevation - np.nanmean(scatter_elevation)) ** 2))
    # y_upper = y_fit + t_value * y_err
    # y_lower = y_fit - t_value * y_err
    # ax3.plot(x_smooth, p(x_smooth), color='red', alpha=0.8)
    # ax3.fill_between(x_smooth, y_lower, y_upper, color='red', alpha=0.4)
    # sns.regplot(x=scatter_elevation, y=elevation_shap, scatter=False,
    #             color='red', ax=ax3)

    ax3.set_xlabel(second_feature)
    ax3.set_ylabel('SHAP value for ' + second_feature)
    ax3.grid(True, alpha=0.2)

    # 右侧散点图2
    ax4 = fig.add_subplot(gs[1, 2])
    # ax4.scatter(scatter_4, shap_4, color=sorted_colors[-4], alpha=0.3, s=10)
    bin_edges = np.arange(np.nanmin(scatter_4), np.nanmax(scatter_4), 2)
    bin_centers = []
    y_means = []
    y_errors = []
    for i in range(len(bin_edges) - 1):
        bin_mask = (scatter_4 >= bin_edges[i]) & (scatter_4 < bin_edges[i + 1])
        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        bin_centers.append(bin_center)
        y_means.append(np.nanmean(shap_4[bin_mask]))
        y_errors.append(np.nanstd(shap_4[bin_mask]))

    ax4.errorbar(bin_centers, y_means, yerr=y_errors, fmt='o', capsize=0, label='Errorbar', markerfacecolor=sorted_colors[-4], markeredgecolor='black', ecolor='black')
    ax4.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    # 添加平滑曲线和置信区间
    # z = np.polyfit(scatter_4, shap_4, 2)
    # p = np.poly1d(z)
    # x_smooth = np.linspace(min(scatter_4), max(scatter_4), 100)
    # y_fit = p(x_smooth)
    # y_pred = p(scatter_4)
    # residuals = shap_4 - y_pred
    # std_err = np.nanstd(residuals)
    # t_value = 1.984
    # y_err = std_err * np.sqrt(1 / len(scatter_4) + (x_smooth - np.nanmean(scatter_4)) ** 2 / np.nansum(
    #     (scatter_4 - np.nanmean(scatter_4)) ** 2))
    # y_upper = y_fit + t_value * y_err
    # y_lower = y_fit - t_value * y_err
    # ax4.plot(x_smooth, p(x_smooth), color='red', alpha=0.8)
    # ax4.fill_between(x_smooth, y_lower, y_upper, color='red', alpha=0.4)
    # sns.regplot(x=scatter_4, y=shap_4, scatter=False,
    #             color='red', ax=ax4)

    ax4.set_xlabel(forth_feature)
    ax4.set_ylabel('SHAP value for ' + forth_feature)
    ax4.grid(True, alpha=0.2)

    # 中间散点图2
    ax5 = fig.add_subplot(gs[1, 1])
    # ax5.scatter(scatter_3, shap_3, color=sorted_colors[-3], alpha=0.3, s=10)
    bin_edges = np.arange(np.nanmin(scatter_3), np.nanmax(scatter_3), 5)
    bin_centers = []
    y_means = []
    y_errors = []
    for i in range(len(bin_edges) - 1):
        bin_mask = (scatter_3 >= bin_edges[i]) & (scatter_3 < bin_edges[i + 1])
        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        bin_centers.append(bin_center)
        y_means.append(np.nanmean(shap_3[bin_mask]))
        y_errors.append(np.nanstd(shap_3[bin_mask]))

    ax5.errorbar(bin_centers, y_means, yerr=y_errors, fmt='o', capsize=0, label='Errorbar', markerfacecolor=sorted_colors[-3], markeredgecolor='black', ecolor='black')
    ax5.axhline(0, color='gray', linestyle='--', linewidth=0.8)


    # 添加平滑曲线和置信区间
    # z = np.polyfit(scatter_3, shap_3, 2)
    # p = np.poly1d(z)
    # x_smooth = np.linspace(min(scatter_3), max(scatter_3), 100)
    # y_fit = p(x_smooth)
    # y_pred = p(scatter_3)
    # residuals = shap_3 - y_pred
    # std_err = np.nanstd(residuals)
    # t_value = 1.984
    # y_err = std_err * np.sqrt(1 / len(scatter_3) + (x_smooth - np.nanmean(scatter_3)) ** 2 / np.nansum(
    #     (scatter_3 - np.nanmean(scatter_3)) ** 2))
    # y_upper = y_fit + t_value * y_err
    # y_lower = y_fit - t_value * y_err
    # ax5.plot(x_smooth, p(x_smooth), color='red', alpha=0.8)
    # ax5.fill_between(x_smooth, y_lower, y_upper, color='red', alpha=0.4)

    # sns.regplot(x=scatter_3, y=shap_3, scatter=False,
    #             color='red', ax=ax5)

    ax5.set_xlabel(third_feature)
    ax5.set_ylabel('SHAP value for ' + third_feature)
    ax5.grid(True, alpha=0.2)

    ax1.text(-0.05,1,"a", transform = ax1.transAxes,weight=1000)
    ax2.text(-0.05, 1, "b", transform=ax2.transAxes,weight=1000)
    ax3.text(-0.05, 1, "c", transform=ax3.transAxes,weight=1000)
    ax4.text(-0.05, 1, "e", transform=ax4.transAxes,weight=1000)
    ax5.text(-0.05, 1, "d", transform=ax5.transAxes,weight=1000)



    # 调整布局
    plt.tight_layout()
    return fig


def plot_figure_3(model, X, shap_values, explainer, features, save_path=None):
    # 计算绝对值均值
    shap_values_bar = np.abs(shap_values)

    mean_shap = np.mean(shap_values_bar, axis=0)


    '''weighted mean'''
    # num_bins = 40
    # mean_shap = np.zeros(17)
    # for i in range(17):
    #     # 对每个变量进行分箱
    #     bins = np.linspace(shap_values_bar[:, i].min(), shap_values_bar[:, i].max(), num_bins + 1)
    #     bin_indices = np.digitize(shap_values_bar[:, i], bins) - 1
    #
    #     # 计算每个箱中的数据比例
    #     bin_counts = np.bincount(bin_indices, minlength=num_bins)
    #     bin_weights = bin_counts / bin_counts.sum()
    #
    #     # 计算加权均值
    #     weighted_mean = np.average(shap_values_bar[:, i], weights=bin_weights[bin_indices])
    #     mean_shap[i] = weighted_mean


    bar_values = mean_shap
    bar_names = features

    # 最大和第二大的索引
    indices1 = np.argsort(mean_shap)[-4:][::-1]
    max_index = indices1[0]
    second_index = indices1[1]
    third_index = indices1[2]
    forth_index = indices1[3]

    # 子图b和c: SHAP值依赖图
    feature_names = [features[max_index], features[second_index], features[third_index], features[forth_index]]  # 最重要的两个特征

    scatter_1 = shap_values[:, max_index]
    x_1 = X[feature_names[0]]
    scatter_2 = shap_values[:, second_index]
    x_2 = X[feature_names[1]]
    scatter_3 = shap_values[:, third_index]
    x_3 = X[feature_names[2]]
    scatter_4 = shap_values[:, forth_index]
    x_4 = X[feature_names[3]]


    fig = create_shap_visualization(bar_values, bar_names, x_1, scatter_1, x_2, scatter_2, x_3, scatter_3, x_4, scatter_4
                                    , feature_names[0], feature_names[1], feature_names[2], feature_names[3])

    # 计算每个类别的贡献百分比
    # feature_categories = {
    #     'Climate': [ 'mean SM', 'mean SRAD', 'AI', "pre", "pi", "pd"],
    #     'Vegetation': ['root depth'],
    #     'Fire': ['burn area'],
    #     'Soil': ['awc', 'sand', 'clay', 'oc'],
    #     'Topography': ['dem', 'slope', 'dem cv', 'slope cv']
    # }

    return fig

def convert_to_dataframe(X_array, feature_names):
    """
    将numpy ndarray和特征名称列表转换为pandas DataFrame

    参数:
    X_array: numpy.ndarray, 形状为(n_samples, n_features)
    feature_names: list, 特征名称列表，长度应该等于X_array.shape[1]

    返回:
    pandas.DataFrame: 转换后的特征矩阵
    """
    # 检查输入数据的维度是否匹配
    # if len(feature_names) != X_array.shape[1]:
    #     raise ValueError(f"特征名称数量({len(feature_names)})与特征矩阵列数({X_array.shape[1]})不匹配")

    # 转换为DataFrame
    X_df = pd.DataFrame(X_array, columns=feature_names)

    return X_df

def read_data(path):
    data = np.load(path)
    print(path, 'read data')
    return data

def zscore_scaler(data):
    z = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    print(z.shape)
    return z


def main():


    # 准备数据
    # woody cover
    dataset = gdal.Open(r"F:\paper\01_soilMoistureSensitivity_2023\20241224\RF\model1\RF_model1_result.tif")
    region = dataset.GetRasterBand(1).ReadAsArray()

    dataset = np.load("F:/paper/01_soilMoistureSensitivity_2023/20240108/data/1MethodUse/woodycover_yearly_1993_2012_0d25.npy")
    # remove anomaly
    dataset[dataset > 77.352] = np.nan
    # remove spatial outlier annually
    for i in range(20):
        dataset[i][np.isnan(region)] = np.nan
        # wc_5th = np.nanpercentile(dataset[i], 5)
        # wc_95th = np.nanpercentile(dataset[i], 95)
        # dataset[i] = np.where((dataset[i] > wc_5th) & (dataset[i] < wc_95th), dataset[i], np.nan)

    # dataset = zscore_scaler(dataset)

    first_wc = np.nanmean(dataset[:5, :, :], axis=0)
    last_wc = np.nanmean(dataset[15:, :, :], axis=0)
    data = last_wc - first_wc
    # data[np.isnan(region)] = np.nan


    # soil
    dataset = gdal.Open(r"F:\paper\01_soilMoistureSensitivity_2023\20241022\data\soil\SAND_0d25.tif")
    sand = dataset.GetRasterBand(1).ReadAsArray()
    dataset = gdal.Open(r"F:\paper\01_soilMoistureSensitivity_2023\20241022\data\soil\CLAY_0d25.tif")
    clay = dataset.GetRasterBand(1).ReadAsArray()
    dataset = gdal.Open(r"F:\paper\01_soilMoistureSensitivity_2023\20241022\data\soil\AWC_0d25.tif")
    awc = dataset.GetRasterBand(1).ReadAsArray()
    dataset = gdal.Open(r"F:\paper\01_soilMoistureSensitivity_2023\20241022\data\soil\OC_0d25.tif")
    oc = dataset.GetRasterBand(1).ReadAsArray()

    # tropography
    dataset = gdal.Open(r"F:\paper\01_soilMoistureSensitivity_2023\20241022\data\tropography\dem_0d25_cv.tif")
    demcv = dataset.GetRasterBand(1).ReadAsArray()
    dataset = gdal.Open(r"F:\paper\01_soilMoistureSensitivity_2023\20241022\data\tropography\dem_0d25_mean.tif")
    dem = dataset.GetRasterBand(1).ReadAsArray()
    dataset = gdal.Open(r"F:\paper\01_soilMoistureSensitivity_2023\20241022\data\tropography\slope_0d25_cv.tif")
    slopecv = dataset.GetRasterBand(1).ReadAsArray()
    dataset = gdal.Open(r"F:\paper\01_soilMoistureSensitivity_2023\20241022\data\tropography\slope_0d25_mean.tif")
    slope = dataset.GetRasterBand(1).ReadAsArray()

    # fire
    ba = np.load(r'F:\paper\01_soilMoistureSensitivity_2023\20241022\data\fire\Mean_burned_area_1993_2012_0d25.npy')

    # vegetation
    root = np.zeros(shape=(720, 1440)) * np.nan
    dataset = gdal.Open(r"F:\paper\01_soilMoistureSensitivity_2023\20241022\data\vegetation\rootDepth_0d25.tif")
    rd = dataset.GetRasterBand(1).ReadAsArray()
    root[:560, :] = rd

    # climate
    AI = np.zeros(shape=(720, 1440)) * np.nan
    dataset = gdal.Open(r"F:\paper\01_soilMoistureSensitivity_2023\20241022\data\ai_et0_0d25.tif")
    ai = dataset.GetRasterBand(1).ReadAsArray()
    AI[:600, :] = ai
    AI[AI < 0 ]= np.nan
    AI = AI*0.0001

    # AI[(AI < 0.5)] = np.nan

    tmp = np.load("F:/paper/01_soilMoistureSensitivity_2023/20240904/zhousha/annual/ERA5_tmp_annual.npy")
    vpd = np.load("F:/paper/01_soilMoistureSensitivity_2023/20240904/zhousha/annual/ERA5_vpd_annual.npy")
    smr = np.load(
        "F:/paper/01_soilMoistureSensitivity_2023/20240904/zhousha/annual/GLEAMv4.1_soilmoisture_10_100cm_mm_annual.npy")
    srad = np.load(
        "F:/paper/01_soilMoistureSensitivity_2023/20240904/zhousha/annual/TerraClimate_srad_annual.npy")

    pre = np.load("F:/paper/01_soilMoistureSensitivity_2023/20240108/data/1MethodUse/CHIRPS_pre_yearly_1993_2012_0d25.npy")
    pi = np.load(
        "F:/paper/01_soilMoistureSensitivity_2023/20240108/data/1MethodUse/CHIRPS_pre_intensity_yearly_1993_2012_0d25.npy")
    pd = np.load(
        "F:/paper/01_soilMoistureSensitivity_2023/20240108/data/1MethodUse/CHIRPS_rainydays_yearly_1993_2012_0d25.npy")

    # sand = zscore_scaler(sand)
    # clay = zscore_scaler(clay)
    # oc = zscore_scaler(oc)
    # awc = zscore_scaler(awc)
    # root = zscore_scaler(root)
    # slopecv = zscore_scaler(slopecv)
    # slope = zscore_scaler(slope)
    # dem = zscore_scaler(dem)
    # demcv = zscore_scaler(demcv)
    # AI = zscore_scaler(AI)
    # ba = zscore_scaler(ba)
    # tmp = zscore_scaler(tmp)
    # vpd = zscore_scaler(vpd)
    # smr = zscore_scaler(smr)
    # srad = zscore_scaler(srad)
    # pre = zscore_scaler(pre)
    # pi = zscore_scaler(pi)
    # pd = zscore_scaler(pd)

    first_tmp = np.nanmean(tmp[:5, :, :], axis=0)
    last_tmp = np.nanmean(tmp[15:, :, :], axis=0)
    tmp = last_tmp - first_tmp
    first_vpd = np.nanmean(vpd[:5, :, :], axis=0)
    last_vpd = np.nanmean(vpd[15:, :, :], axis=0)
    vpd = last_vpd - first_vpd
    first_smr = np.nanmean(smr[:5, :, :], axis=0)
    last_smr = np.nanmean(smr[15:, :, :], axis=0)
    smr = last_smr - first_smr
    first_srad = np.nanmean(srad[:5, :, :], axis=0)
    last_srad = np.nanmean(srad[15:, :, :], axis=0)
    srad = last_srad - first_srad

    first_pre = np.nanmean(pre[:5, :, :], axis=0)
    last_pre = np.nanmean(pre[15:, :, :], axis=0)
    pre = last_pre - first_pre
    first_pi = np.nanmean(pi[:5, :, :], axis=0)
    last_pi = np.nanmean(pi[15:, :, :], axis=0)
    pi = last_pi - first_pi
    first_pd = np.nanmean(pd[:5, :, :], axis=0)
    last_pd = np.nanmean(pd[15:, :, :], axis=0)
    pd = last_pd - first_pd






    # 构建输入数据
    data = data.flatten()
    sand = sand.flatten()
    clay = clay.flatten()
    oc = oc.flatten()
    awc = awc.flatten()
    root = root.flatten()
    slopecv = slopecv.flatten()
    slope = slope.flatten()
    dem = dem.flatten()
    demcv = demcv.flatten()
    AI = AI.flatten()
    ba = ba.flatten()
    tmp = tmp.flatten()
    vpd = vpd.flatten()
    smr = smr.flatten()
    srad = srad.flatten()
    pre = pre.flatten()
    pi = pi.flatten()
    pd = pd.flatten()


    mask = ~(np.isnan(data) | np.isnan(smr) | np.isnan(srad) | np.isnan(vpd) | np.isnan(pre)
             | np.isnan(AI) | np.isnan(root) | np.isnan(ba)
             | np.isnan(sand) | np.isnan(clay) | np.isnan(oc) | np.isnan(awc)
             | np.isnan(dem) | np.isnan(demcv) | np.isnan(slope) | np.isnan(slopecv)
             | np.isnan(pi) | np.isnan(pd) | np.isinf(pi))

    arrays = [data, smr, srad, vpd, pre, pi, pd, AI, root, ba, sand, clay, oc, awc, dem, demcv, slope, slopecv]

    print(np.unique(mask))

    dataset = np.column_stack(arrays)
    dataset = dataset[mask]

    # X = X[mask]
    # y = data[mask]
    
    # 同步修改112与134行的饼状图颜色映射与数值
    feature_names = ["ΔSM", "ΔSRAD", "ΔVPD", "ΔPRE", "ΔPI", "ΔRD",
                     "AI", "Root depth", "Burn area",
                     "Sand", "Clay", "OC", "AWC",
                     "Dem", "Dem cv", "Slope", "Slope cv"
                     ]

    # X = convert_to_dataframe(X, feature_names)
    # y = convert_to_dataframe(y, ["sensitivity"])

    feature_names_withy = ["y","ΔSM", "ΔSRAD", "ΔVPD", "ΔPRE", "ΔPI", "ΔRD",
                     "AI", "Root depth", "Burn area",
                     "Sand", "Clay", "OC", "AWC",
                     "Dem", "Dem cv", "Slope", "Slope cv"
                     ]
    dataset = convert_to_dataframe(dataset, feature_names_withy)

    # print(dataset[np.isinf(dataset).any(axis=1)]['pi'])
    # print(dataset.columns[np.isinf(dataset).any(axis=0)])

    def calculate_vif(x):
        x = sm.add_constant(x)
        vif = pandas.DataFrame()
        vif["VIF"] = [1/(1- r**2) for r in [sm.OLS(x[y], x.drop(y, axis=1)).fit().rsquared for y in x.columns]]
        vif["features"] = x.columns
        return vif

    print(calculate_vif(dataset.drop(columns='y')))

    df_sample = dataset.sample(frac=0.9, random_state=42)
    print(df_sample)

    # 训练模型
    model, X_train, X_test, y_train, y_test = train_xgboost_model(df_sample.drop('y', axis=1), df_sample['y'])

    # 计算SHAP值
    shap_values, explainer = calculate_shap_values(model, X_train)


    # 绘制Figure 3
    fig = plot_figure_3(model, X_train, shap_values, explainer, features=feature_names)
    # plt.show()
    # plt.savefig(r"H:/paper/01_soilMoistureSensitivity_2023/figures/Fig1.jpg" ,dpi=300, bbox_inches='tight')
    plt.savefig(r"E:\research\01SM_woodycover\00progress\figures\new\Fig1.jpg", dpi=300, bbox_inches='tight')

    # 输出模型性能指标
    # train_score = model.score(X_train, y_train)
    # test_score = model.score(X_test, y_test)
    # print(f"Training R2: {train_score:.3f}")
    # print(f"Testing R2: {test_score:.3f}")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Abosolute Error: {mae:.3f}")
    print(f"Mean Squared Error: {mse:.3f}")
    print(f"R2: {r2:.3f}")



if __name__ == "__main__":
    main()