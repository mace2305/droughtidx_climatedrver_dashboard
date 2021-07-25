#import main.processing as processing
#from . import processing
from main import processing
#import processing
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import cartopy.crs as ccrs
import pandas as pd
import xarray as xr
from scipy import stats
from dask.array import stats as da_stats
import streamlit as st
import seaborn as sns
from matplotlib.patches import Rectangle
import calendar
plt.subplots_adjust(left=0.4, right=0.8, bottom=0.4, top=0.8)

### For mean/std/kurtosis of whole

def tick_maker(coordinate_scale, num_ticks=5):
    return np.array(
        [np.percentile(coordinate_scale, i) for i in np.linspace(0,100,num_ticks)]
        )

def plot_xr_dataarray_parameter(xr_darray, cmap, title, vmax, vmin=None, robust=None):
    fig,p = plt.subplots(figsize=(12,8))
    p = plt.axes(projection=ccrs.PlateCarree())
    xr_darray.plot(vmax=vmax, vmin=vmin, ax=p, cmap=cmap, robust=robust)
    plt.title(title)
    plt.ylabel('latitude')
    plt.xlabel('longitude')
    p.set_yticks(tick_maker(xr_darray.lat))
    p.set_xticks(tick_maker(xr_darray.lon))
    p.axes.coastlines()
    return fig

def generate_mean_std_kurtosis(xarray_ds, EN34df, ENSO_phase, timescale=3):
    # monthstart_idx = ENSO_phase.groupby(pd.Grouper(freq='MS')).all().index
    EN34df_idx = EN34df[EN34df.index.isin(ENSO_phase.index)].index
    # ENSO_phase_xr_ds = EN34df[xarray_ds.time.isin(monthstart_idx)]
    EN34df_idx = EN34df_idx + pd.offsets.MonthBegin(-1)
    xarray = xarray_ds.sel(time=EN34df_idx)

    mean = xarray.sat_gauge_precip.mean(axis=0)
    p1 = plot_xr_dataarray_parameter(mean, 'plasma', 'Mean', 25, )
    
    std = xarray.sat_gauge_precip.std(axis=0).values
    std = xr.DataArray(std, coords=[xarray_ds.lat.values, xarray_ds.lon.values], dims=['lat', 'lon'])
    p2 = plot_xr_dataarray_parameter(std, 'Set1', 'Std', 20, )
    
    v = xarray.sat_gauge_precip.values
    kurtosis = da_stats.kurtosis(v).compute()
    kurtosis = xr.DataArray(kurtosis, coords=[xarray_ds.lat.values, xarray_ds.lon.values], dims=['lat', 'lon'])
    p3 = plot_xr_dataarray_parameter(kurtosis, 'RdBu', 'Kurtosis', 15, -15, True)
    
    skewness = da_stats.skew(v).compute()
    skewness = xr.DataArray(skewness, coords=[xarray_ds.lat.values, xarray_ds.lon.values], dims=['lat', 'lon'])
    p4 = plot_xr_dataarray_parameter(skewness, 'seismic_r', 'Skewness', 5, -5, True)
    return p1, p2, p3, p4

### For scatterplot correlation

def EN34_colors_size(EN34):
    strongEN34 = np.array([(en,ln) for en,ln in zip(EN34.strong_elnino, EN34.strong_lanina)])
    weakmodEN34 = np.array([(en,ln) for en,ln in zip(EN34.elnino, EN34.lanina)])
    strongEN34_en = np.argwhere(strongEN34 == [1,np.nan])[:,0]
    strongEN34_la = np.argwhere(strongEN34 == [np.nan,1])[:,0]
    weakmodEN34_en = ~np.isnan(weakmodEN34[:,0])
    weakmodEN34_la = ~np.isnan(weakmodEN34[:,1])
    EN34_colors = np.array(['k' for i in range(len(EN34))])
    EN34_colors[weakmodEN34_en] = 'r'
    EN34_colors[strongEN34_en] = 'r'
    EN34_colors[weakmodEN34_la] = 'b'
    EN34_colors[strongEN34_la] = 'b'
    EN34_size = np.array([10 for i in range(len(EN34))])
    EN34_size[strongEN34_en] = 50
    EN34_size[strongEN34_la] = 50
    return EN34_colors, EN34_size

def scatterplot_EN34_SPI(x, y, c, s, alpha=0.5):
    fig, p = plt.subplots(figsize=(12.0, 8.0), dpi=400)
    plt.scatter(x,y, c=c, s=s, alpha=alpha);
    plt.xlabel('EN34')
    plt.ylabel('SPI')
    plt.title('Scatter between EN34 and SPI. Larger points indicate strong ENSO event.' \
              '\nRed: El Nino. Blue: La Nina.', fontsize=15)
    plt.axhline(1, c='green', linestyle='dashed', alpha=.5)
    plt.axhline(1.5, c='green', linestyle='dashed')
    plt.axhline(2, c='green', linestyle='dashed', linewidth=2)
    plt.axhline(-1, c='brown', linestyle='dashed', alpha=.5)
    plt.axhline(-1.5, c='brown', linestyle='dashed')
    plt.axhline(-2, c='brown', linestyle='dashed', linewidth=2)
    plt.ylim([-3, 3])
    # print(__name__)
    # if __name__!='main.visuals':plt.show();
    plt.close('all')
    return fig

def generate_EN34_SPI_scatterplots(EN34df):
    # EN34 = processing.dataframe_ENSO_spi(spi_array, xarray_ds, nino34_3mthly, elnino, lanina)
    EN34_colors, EN34_size = EN34_colors_size(EN34df)
    scatter_pwhole = scatterplot_EN34_SPI(EN34df.EN34, EN34df.spi3, c=EN34_colors, s=EN34_size, alpha=0.5)
    elnino_colors, elnino_size = EN34_colors_size(EN34df[(~np.isnan(EN34df.elnino)) & (~np.isnan(EN34df.spi3))])
    scatter_p_en = scatterplot_EN34_SPI(EN34df[(~np.isnan(EN34df.elnino)) & (~np.isnan(EN34df.spi3))].EN34, 
                EN34df[(~np.isnan(EN34df.elnino)) & (~np.isnan(EN34df.spi3))].spi3, 
                c=elnino_colors, s=elnino_size, alpha=0.5)
    lanina_colors, lanina_size = EN34_colors_size(EN34df[(~np.isnan(EN34df.lanina)) & (~np.isnan(EN34df.spi3))])
    scatter_p_ln = scatterplot_EN34_SPI(EN34df[(~np.isnan(EN34df.lanina)) & (~np.isnan(EN34df.spi3))].EN34, 
                EN34df[(~np.isnan(EN34df.lanina)) & (~np.isnan(EN34df.spi3))].spi3, 
                c=lanina_colors, s=lanina_size, alpha=0.5)
    return scatter_pwhole, scatter_p_en, scatter_p_ln

### For time series
# to-do: insert figures (not generate plots) of 6-year stretches, just do this in load.py: then put on experimental page, or preview page
# so even if user does not want to download data, can still view some images

def plot_SPI_ENSO(spi_array, xarray_ds, elnino, lanina):
    time_idx = xarray_ds.time.values
    spi3 = pd.DataFrame(spi_array, time_idx, columns=['SPI-3'])
    vals = spi3.values[:,0]
    fig, p = plt.subplots(figsize=(17,7.5), dpi=400)
    # p = plt.axes()
    spi3.plot(color='black', ax=p)
    plt.vlines(x = elnino.index, ymin=-3, ymax=3, 
               colors='pink', lw=1, label='El_Nino', alpha=1)
    plt.vlines(x = processing.strong_elnino(elnino).index, ymin=-3, ymax=3, linestyles='dashed',
               colors='red', lw=1, label='StrongEl_Nino', alpha=.9)
    plt.vlines(x = lanina.index, ymin=-3, ymax=3, 
               colors='cornflowerblue', lw=1, label='La_Nina', alpha=.6)
    plt.vlines(x = processing.strong_lanina(lanina).index, ymin=-3, ymax=3, linestyles='dashed',
               colors='blue', lw=1, label='StrongLa_Nina', alpha=.9)
    plt.hlines(y = 1, xmin=0, xmax=2019, lw=0.5, linestyles='dashed')
    plt.hlines(y = 2, xmin=0, xmax=2019, lw=0.5, linestyles='dashed', colors='blue')
    plt.hlines(y = -1, xmin=0, xmax=2019, lw=0.5, linestyles='dashed')
    plt.hlines(y = -2, xmin=0, xmax=2019, lw=0.5, linestyles='dashed', colors='red')
    plt.fill_between(spi3.index, vals, 1, where=vals>1, 
                     color='lime', alpha=.5)
    plt.fill_between(spi3.index, vals, -1, where=vals<-1, 
                     color='saddlebrown', alpha=.8)
    plt.ylabel('SPI')
    plt.xlabel('Year')
    plt.title('Non-IPO corrected, 5 consecutive month of 3-mth rolling mean of +/- 0.65 OISST anomaly.',
              fontsize=20)
    plt.legend(loc='lower right')
    plt.grid(True, which='both', axis='both')
    # plt.show()
    return fig

### For time series
def plot_params(ax, elnino, lanina):
    plt.vlines(x = elnino.index, ymin=-3.5, ymax=3.5, 
               colors='pink', lw=1, label='El_Nino', alpha=1)
    plt.vlines(x = processing.strong_elnino(elnino).index, ymin=-3.5, ymax=3.5, linestyles='dashed',
               colors='red', lw=1, label='StrongEl_Nino', alpha=.9)
    plt.vlines(x = lanina.index, ymin=-3.5, ymax=3.5, 
               colors='cornflowerblue', lw=1, label='La_Nina', alpha=.6)
    plt.vlines(x = processing.strong_lanina(lanina).index, ymin=-3.5, ymax=3.5, linestyles='dashed',
               colors='blue', lw=1, label='StrongLa_Nina', alpha=.9)
    plt.hlines(y = 1, xmin=0, xmax=2019, lw=0.5, linestyles='dashed')
    plt.hlines(y = 2, xmin=0, xmax=2019, lw=0.5, linestyles='dashed', colors='blue')
    plt.hlines(y = -1, xmin=0, xmax=2019, lw=0.5, linestyles='dashed')
    plt.hlines(y = -2, xmin=0, xmax=2019, lw=0.5, linestyles='dashed', colors='red')
    plt.ylabel('SPI')
    plt.xlabel('Year')
    plt.title('Non-IPO corrected, 5 consecutive month of 3-mth rolling mean of +/- 0.65 OISST anomaly.',
              fontsize=15)
    ax.legend(bbox_to_anchor=(1.02, 1.05))
    plt.grid(True, which='both', axis='x')
    # plt.show()

def plot_SPI_ENSO_multiple_latintervals(spi_multilats, rf, lat_slices, elnino, lanina, yrs_to_plot=72):
    time_idx = rf.time.values
    excess = len(time_idx) - len(spi_multilats[0])
    time_idx = time_idx[excess : ]

    sns.set_style("white")
    sns.set_context('talk')
    fig_ls = list() # hold all six-year stretches

    for t in range(int(np.ceil(len(time_idx)/yrs_to_plot))): # per six-year stretch
        f, ax = plt.subplots(figsize=(15,7))
        for i, (spi, lat_range) in enumerate([*zip(spi_multilats, lat_slices)]):
        # print(spi.shape)
        # print(t*yrs_to_plot)
        # print(f'--> {spi[t*yrs_to_plot : (t+1)*yrs_to_plot].shape}')
        # print((t+1)*yrs_to_plot)
        # print(f'-----> {time_idx[t*yrs_to_plot : (t+1)*yrs_to_plot].shape}')
        # print(f'--------> {lat_slices[0]}')
            spi3 = pd.DataFrame(
                spi[t*yrs_to_plot : (t+1)*yrs_to_plot], 
                time_idx[t*yrs_to_plot : (t+1)*yrs_to_plot], 
                columns=[lat_slices[i]])
        # vals = spi3.values[:,0]
            spi3.plot(ax=ax, alpha=1)
        plot_params(ax, elnino, lanina)
        fig_ls.append(f)
        plt.close('all')
    return fig_ls

### Monthly boxplots

def generate_monthlySPI3_boxplot(EN34df, title, timescale=3):
    # df = pd.DataFrame(EN34df.spi3)
    df = pd.DataFrame(EN34df[f'spi{timescale}'])
    df['month'] = df.index.month
    fig, ax = plt.subplots(figsize=(10,7))
    sns.boxplot(x='month', y=f'spi{timescale}', data=df, ax=ax)
    ax.set_ylim([3,-3])
    ax.hlines(0, -.5, 11.5, 'r', linestyle='dashed');
    plt.title(f'Latitude: {title}', fontsize=15)
    return fig

def boxplot2(spi, rf, title, nino34_mthly, elnino, lanina, timescale=3):
    df = processing.dataframe_ENSO_spi(spi, rf, nino34_mthly, elnino, lanina, )
    # print(df)
    # df = pd.DataFrame(EN34df[f'spi{timescale}'])
    df['month'] = df.index.month
    fig, ax = plt.subplots(figsize=(10,4))
    sns.boxplot(x='month', y=f'spi{timescale}', data=df, ax=ax)
    ax.set_ylim([3,-3])
    ax.hlines(0, -.5, 11.5, 'r', linestyle='dashed');
    plt.title(f'Latitude: {title}', fontsize=15)
    return fig


### Correlation heatmaps

def correlation_SPI_ENSO(EN34df, lat_range):
    # EN34 = processing.dataframe_ENSO_spi(spi_array, xarray_ds, nino34_3mthly, elnino, lanina)
    df = EN34df.get(['EN34', 'spi3'])
    df['month'] = df.index.month
    idx = pd.MultiIndex.from_tuples(
        list(zip(*[[str(lat_range[0]) for _ in range(3)], ['month', 'r', 'p']]))
    )
    df = pd.DataFrame([(gn, *(stats.pearsonr(gd['spi3'], gd['EN34']))) 
          for gn,gd in df.groupby('month')], columns=idx)
    return df

def multiple_latitudeintervals_correlation_heatmap(spi_array, xarray_ds, nino34_mthly, elnino, lanina, interval=5):
    df = pd.DataFrame()
    for lat_range, spi in zip(processing.lat_slicing(xarray_ds, interval=interval), spi_array):
        df_new = correlation_SPI_ENSO(spi, xarray_ds, nino34_mthly, elnino, lanina, lat_range)
        df = pd.concat([df, df_new], axis=1)
    print(df)
    df.index = [calendar.month_abbr[i+1] for i in range(0,12)]
    f, ax = plt.subplots(figsize=(17,8))
    mask = np.array(df.iloc[:, df.columns.get_level_values(1)=='p'] > 0.05)
    sns.heatmap(df.iloc[:, df.columns.get_level_values(1)=='r'].T, 
                annot=True, fmt='.2', ax=ax,
                yticklabels=df.loc[:, pd.IndexSlice[:, 'r']].columns.get_level_values(0),
                cmap='coolwarm_r', linewidths=1
               )
    # x= np.arange(len(df.index)+1)
    # y= np.arange(len(df.columns)/3+1)
    for pt in np.argwhere(mask):
         ax.add_patch(Rectangle((pt), 1,1, ec='white', fc='white', 
                                hatch='/', 
                                alpha=.5
                     ))
    plt.ylabel('Latitude')
    plt.yticks(rotation=0)
    plt.title('Correlation between EN34 and SPI3 across the months and latitudes.\nFaded/hatched numbers indicate p-values >0.05.',
              fontsize=15
             );
    return f

def correlation_heatmap2(spi, rf, nino34_mthly, elnino, lanina, lat_slices, current_spi_timescale):
    df = pd.DataFrame()
    for lat_range, s in zip(lat_slices, spi):
        EN34df_s = processing.dataframe_ENSO_spi(s, rf, nino34_mthly, elnino, lanina, )
        df_new = correlation_SPI_ENSO(EN34df_s, lat_range)
        df = pd.concat([df, df_new], axis=1)
    df.index = [calendar.month_abbr[i+1] for i in range(0,12)]
    f, ax = plt.subplots(figsize=(17,8))
    mask = np.array(df.iloc[:, df.columns.get_level_values(1)=='p'] > 0.05)
    sns.heatmap(df.iloc[:, df.columns.get_level_values(1)=='r'].T, 
                annot=True, fmt='.2', ax=ax,
                yticklabels=df.loc[:, pd.IndexSlice[:, 'r']].columns.get_level_values(0),
                cmap='coolwarm_r', linewidths=1
               )
    # x= np.arange(len(df.index)+1)
    # y= np.arange(len(df.columns)/3+1)
    for pt in np.argwhere(mask):
         ax.add_patch(Rectangle((pt), 1,1, ec='white', fc='white', 
                                hatch='/', 
                                alpha=.5
                     ))
    plt.ylabel('Latitude')
    plt.yticks(rotation=0)
    plt.title('Correlation between EN34 and SPI3 across the months and latitudes.\nFaded/hatched numbers indicate p-values >0.05.',
              fontsize=15
             );
    return f


# rf = load.load_rainfall('gpcp')
# fig = plt.figure()
# rf.sel(time='2020-01-01').sat_gauge_precip.plot(
#     subplot_kws=dict(projection=ccrs.PlateCarree()), cmap='terrain_r')
# p = plt.axes(projection=ccrs.PlateCarree())
# p.axes.coastlines()
# p.axes.gridlines(draw_labels=True)
# fig
