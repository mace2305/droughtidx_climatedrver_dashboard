import pandas as pd
import numpy as np
from climate_indices import indices, compute
import streamlit as st
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt

### explore.py

def is_consecutive_5mth(df):
    datelist = df.index
    ls = list()
    ls2 = list()
    for idx, my_date in enumerate(datelist[:]):
        if idx in [0,1]: 
            ls.append([my_date, None]); continue
        if (
            (my_date.month >= 5) and
            (my_date.month - datelist[idx - 4].month == 4)
        ) or (
            (my_date.year - datelist[idx - 4].year == 1) and 
            (datelist[idx - 4].month == 12 - (4-my_date.month))
        ):
            ls.append([my_date, True])
            ls2.append([my_date, df[my_date]])
        else:
            ls.append([my_date, False])
    return ls, pd.DataFrame(ls2, columns=['time','3mth_rmean']).set_index('time')

def ENSO_TTR2018(nino34):
    """
    Get ENSO phases from Nino 3.4. indexed by PSL
    """
    nino34_mthly = nino34.NINO34.to_pandas().resample('M').mean()
    nino34_3mthly = nino34_mthly.rolling(3).mean()
    nino34_3mthly_abv65 = nino34_3mthly[nino34_3mthly > .65]
    nino34_3mthly_bel65 = nino34_3mthly[nino34_3mthly < -.65]
    return nino34_mthly, *is_consecutive_5mth(nino34_3mthly_abv65), *is_consecutive_5mth(nino34_3mthly_bel65)

def lat_slicing(xarray_ds, interval):
    lhs, rhs = xarray_ds.lat[0], xarray_ds.lat[-1]
    if interval==0:
        return [(lhs.item(), rhs.item())]
    if lhs > rhs: # True if descending, vice versa
        first = np.ceil(lhs.item()); last = np.floor(rhs.item()); direction = -interval
    else: 
        first = np.floor(lhs.item()); last = np.ceil(rhs.item()); direction = interval
    return [(i, i+direction) for i in np.arange(first, last, direction)] # e.g. [(35, 30), (30, 25), ..]

def get_precipitation_means(xarray_ds, lat_slices):
    if lat_slices == None:
        return xarray_ds.sat_gauge_precip.mean(axis=(1,2)).values 
    else: return np.array(
        [xarray_ds.sel(lat=slice(l,r)).sat_gauge_precip.mean(axis=(1,2)).values 
         for l,r in lat_slices])

def generate_spi(precip_arr, timescale=3):
    return indices.spi(
        values = precip_arr, scale = timescale, distribution = indices.Distribution.gamma,
        data_start_year = 1900, calibration_year_initial = 1900, calibration_year_final = 2021,
        periodicity = compute.Periodicity.monthly
    )

def get_SPI_sliced(xarray_ds, interval):
    if interval == 'Whole-length': 
        lat_slices = None
        precip_means = get_precipitation_means(xarray_ds, lat_slices=None)
        spi_intervals = generate_spi(precip_means)
    else: 
        lat_slices = lat_slicing(xarray_ds, interval=interval)
        precip_means = get_precipitation_means(xarray_ds, lat_slices)
        spi_intervals = np.apply_along_axis(generate_spi, 1, precip_means)
    return spi_intervals, lat_slices

def weakmod_elnino(df): return df[((df>=.65) & (df<1.5)).values]
def weakmod_lanina(df): return df[((df<=-.65) & (df>-1.5)).values]
def strong_elnino(df): return df[(df>=1.5).values]
def strong_lanina(df): return df[(df<=-1.5).values]

def dataframe_ENSO_spi(spi_array, xarray_ds, nino34_mthly, elnino, lanina, timescale=3):
    # nino34_mthly, elnino_full, elnino, lanina_full, lanina = ENSO_definition()
    time_idx = xarray_ds.time.values
    # print(f'xarray_ds is {xarray_ds}')
    # print(f'time_idx.shape is {time_idx.shape}')
    # print(f'spi_array is {spi_array}')
    # print(f'spi_array.shape is {spi_array.shape}')
    excess = len(time_idx) - len(spi_array)
    time_idx = time_idx[excess : ]
    spi3_me = pd.DataFrame(spi_array, time_idx, columns=[f'SPI-{timescale}']).groupby(
        pd.Grouper(freq='M')).sum() #me:month-end
    # print(f'1: {spi3_me}')
    spi3_me.index.rename('time', inplace=True)
    spi3_me = spi3_me.shift(-1)# note: GPCP was operational up to 2019
    # print(f'2: {spi3_me}')
    spi3_me = spi3_me.replace(0, np.nan) #note: to incorporate jan & feb of 1983 where rolling-mean(3) is calculated & therefore voided 
    # print(f'3: {spi3_me}')
    EN34 = pd.DataFrame(nino34_mthly) \
        .merge(elnino, on=['time'], how='outer') \
        .merge(~np.isnan(strong_elnino(elnino)), on=['time'], how='outer') \
        .merge(lanina, on=['time'], how='outer') \
        .merge(~np.isnan(strong_lanina(lanina)), on=['time'], how='outer') \
        .merge(spi3_me, on=['time'], how='outer')
    EN34.columns=['EN34','elnino','strong_elnino','lanina','strong_lanina',
                  'spi3'
                 ]
    EN34 = EN34[~np.isnan(EN34.spi3)]
    return EN34




### regression.py

def train_test_split(endog, exog, train_split, val_split, option_valid_split, option_exog):
    # print(f'state.train_split is {state.train_split}')
    # print(f'state.val_split is {state.val_split}')
    # print(f'state.test_split is {state.test_split}')
    # st.write(endog[0 : state.train_split].shape)
    # st.write(len(endog))
    train_s = int(np.floor(train_split/100*len(endog)))
    test_s = int(np.floor((train_split+val_split)/100*len(endog)))
    
    if option_valid_split:
        endog_train, endog_val, endog_test = endog[0 : train_s], \
        endog[train_s : test_s], \
        endog[test_s : ]
        if option_exog: 
            exog_train, exog_val, exog_test = exog[0 : train_s], \
            exog[train_s : test_s], \
            exog[test_s : ]
    else: 
        endog_val, exog_val = None, None
        endog_train, endog_test = endog[ : train_s], endog[train_s : ]
        if option_exog: 
            exog_train, exog_test = exog[ : train_s], exog[train_s : ]
            
    return endog_train, endog_val, endog_test, exog_train, exog_val, exog_test, train_s, test_s

def determine_split_ratio(option_valid_split, option_train_val_test_split_ratio, option_train_test_split_ratio):
    if option_valid_split:
        train_split = option_train_val_test_split_ratio[0]
        val_split = option_train_val_test_split_ratio[1]-option_train_val_test_split_ratio[0]
        test_split = 100-option_train_val_test_split_ratio[1]
        string = f"**Train-val-test split is now {train_split}-{val_split}-{test_split}**"
    else:
        train_split = option_train_test_split_ratio
        val_split = 0
        test_split = 100-option_train_test_split_ratio
        string =  f"**Train-test split is now {train_split}-{test_split}**"
    return train_split, val_split, test_split, string

def generate_order_triplets(option_ARIMA_SARIMA, option_SARIMAX_config, items_SARIMAX):
    if option_ARIMA_SARIMA == 'ARIMA':
        p,d,q,P,D,Q,s = '','','','','','','',
        try: p,d,q = [range(0,i+1) for i in items_SARIMAX]
        except AttributeError:st.write('Please submit configurations first.')

        if option_SARIMAX_config=='AUTO':
            pdq = list(itertools.product(p,d,q))
            permutations_generated = pdq
            permutations_generated_seasonal = (0,0,0,0)
            return permutations_generated, permutations_generated_seasonal, pd.DataFrame(pdq, columns=['p','d','q'], index=pd.RangeIndex(1,len(pdq)+1))
        elif option_SARIMAX_config=='MANUAL':
            pdq=(items_SARIMAX)
            permutations_generated = pdq
            permutations_generated_seasonal = (0,0,0,0)
            return permutations_generated, permutations_generated_seasonal, pd.DataFrame(pdq, index=['p','d','q']).T

    elif option_ARIMA_SARIMA == 'SARIMA':
        p,d,q,P,D,Q,s = [range(0,i+1) if idx != 6 else i for idx,i in enumerate(items_SARIMAX)]

        if option_SARIMAX_config=='AUTO':
            pdq = list(itertools.product(p,d,q))
            PDQs = [i+(s,) for i in list(itertools.product(P,D,Q))]
            permutations_generated = pdq
            permutations_generated_seasonal = PDQs
            pdqPDQs = [i+(s,) for i in list(itertools.product(p,d,q,P,D,Q))]
            return permutations_generated, permutations_generated_seasonal, pd.DataFrame(pdqPDQs, columns=['p','d','q','P','D','Q','s'], index=pd.RangeIndex(1,len(pdqPDQs)+1))
        elif option_SARIMAX_config=='MANUAL': 
            pdqPDQs=items_SARIMAX
            permutations_generated = pdqPDQs[:3]
            permutations_generated_seasonal = pdqPDQs[3:]
            return permutations_generated, permutations_generated_seasonal, pd.DataFrame(pdqPDQs, index=['p','d','q','P','D','Q','s']).T

def run_SARIMAX(endog, exog, param, param_seasonal, purpose, option_ARIMA_SARIMA=None):
    mod = sm.tsa.statespace.SARIMAX(endog, exog, 
                                    order=param,
                                    seasonal_order=param_seasonal,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit(disp=0)
    if purpose=='autodetermine':
        if option_ARIMA_SARIMA == 'ARIMA':string='ARIMA{} - AIC:{}'.format(param, results.aic)
        elif option_ARIMA_SARIMA == 'SARIMA': string='SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic)
        return results.aic, string
    elif purpose=='best_model':
        results = mod.fit(disp=0)
        summary_table = results.summary().tables[1]
        fig = plt.figure(figsize=(15, 12))
        results.plot_diagnostics(fig=fig)
        return fig, summary_table.as_html(), results


def one_step_validation_forecast(results, train_s, test_s, exog_val, endog_full, ):
    pred = results.get_prediction(
        start=train_s, 
        end=test_s-1,
        exog=exog_val, dynamic=False
        )
    pred_ci = pred.conf_int();

    fig = plt.figure(figsize=(12, 6))
    ax = endog_full[:].plot(label='observed', fig=fig);
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7);
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('SPI')
    plt.legend()

    # Compute the mean square error
    y_forecasted = pred.predicted_mean
    print(f'>>> pred mean shape {pred.predicted_mean.shape}')
    y_truth = endog_full[train_s : ]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    val_start_date = endog_full.index[train_s : test_s][0].strftime('%Y-%m-%d')
    val_end_date = endog_full.index[train_s : test_s][-1].strftime('%Y-%m-%d')
    return fig, 'The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)), val_start_date, val_end_date

if __name__=='__main__':
    pass
























