from main import processing, utils
#import processing
import sys, os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib')))
import xarray as xr
from pathlib import Path
from timeit import default_timer as timer
import streamlit as st

default_pickle_dir = Path(__file__).resolve().parents[1] / './data_sets/prepared'
print(f'Pickles stored at: {default_pickle_dir}')

def printvars():
   tmp = globals().copy()
   [print(k,'  :  ',v,' type:' , type(v)) for k,v in tmp.items() if not k.startswith('_') and k!='tmp' and k!='In' and k!='Out' and not hasattr(v, '__call__')]
   print('----------------')
   tmp = locals().copy()
   [print(k,'  :  ',v,' type:' , type(v)) for k,v in tmp.items() if not k.startswith('_') and k!='tmp' and k!='In' and k!='Out' and not hasattr(v, '__call__')]
   print('================\n\n')

def retrieve(item, folder, process):
    """
    find {item} in {folder}
    if found: return it
    if not found: run process to generate item, pickle item, return it
    """
    os.makedirs(folder, exist_ok=True)
    if utils.find(f'*{item}.pkl', default_pickle_dir):
        print('loaded pickle')
        item_obj = utils.open_pickle(utils.find(f'*{item}.pkl', folder)[0])
    else:
        print('dataset generated manually')
        item_obj = process()
        utils.to_pickle(f'{item}', item_obj, folder)
    return item_obj

def load_datasets(option_sidebar_ENSO_def, option_sidebar_rainfall_ds):
    # printvars()
    nino34_mthly, elnino_full, elnino, lanina_full, lanina = load_nino34(option_sidebar_ENSO_def)
    rf = load_rainfall(option_sidebar_rainfall_ds)
    return nino34_mthly, elnino_full, elnino, lanina_full, lanina, rf

# Nino34

def get_nino34oisst():
    return xr.open_dataset(
    r'./data_sets/ENSO/nino34_weekly_SoTO_1981_to_present_OISST.nc').rename(
        {'WEDCEN2':'time'})

def load_nino34(source = 'Timbal, Turkington, Rahmat, 2019 (OISST)'):
    print(f'{utils.time_now()}: {source}')
    nino34oisst = None
    if source == 'Timbal, Turkington, Rahmat, 2019 (OISST)':
        # https://stateoftheocean.osmc.noaa.gov/sur/pac/nino34.php
        nino34oisst = retrieve('nino34oisst', default_pickle_dir, get_nino34oisst)
        # if utils.find(f'*nino34oisst.pkl', default_pickle_dir):
        #     nino34oisst = utils.open_pickle(utils.find(f'*nino34oisst.pkl', default_pickle_dir)[0])
        # else:
        #     nino34oisst = xr.open_dataset(
        #     r'./data_sets/ENSO/nino34_weekly_SoTO_1981_to_present_OISST.nc').rename(
        #         {'WEDCEN2':'time'})
        #     utils.to_pickle('nino34oisst', nino34oisst, default_pickle_dir)
    nino34_mthly, elnino_full, elnino, lanina_full, lanina = processing.ENSO_TTR2018(nino34oisst)
    return nino34_mthly, elnino_full, elnino, lanina_full, lanina


# Rainfall

def get_GPCP_rfds():
    gpcp_dir = r'./data_sets/RF_SPI/GPCP_monthly_1983_to_2019'
    gpcp_files = Path(gpcp_dir).glob('*.nc4')
    return xr.open_mfdataset(gpcp_files)

def load_rainfall(source='GPCP'):
    print(f'{utils.time_now()}: {source}')
    if source == 'GPCP':
        rf = retrieve('GPCP', default_pickle_dir, get_GPCP_rfds)
        # gpcp_dir = r'./data_sets/RF_SPI/GPCP_monthly_1983_to_2019'
        # gpcp_files = Path(gpcp_dir).glob('*.nc4')
        # rf = xr.open_mfdataset(gpcp_files)
    return rf



### helper functions

def precheck_loaded_datasets(warning_prompt=False):
    try: 
        type(st.session_state.elnino)
        print('Datasets loaded.')
    except AttributeError:
        print('Datasets not loaded')
        st.write('Dataset reloaded according to current settings...')
        try: 
            st.session_state.nino34_mthly, st.session_state.elnino_full, st.session_state.elnino, st.session_state.lanina_full, st.session_state.lanina, st.session_state.rf = load_datasets(
                # option_sidebar_ENSO_def, option_sidebar_rainfall_ds
                st.session_state.current_sidebar_ENSO_def, st.session_state.current_sidebar_rainfall_ds
                )
        except UnboundLocalError: 
            st.write('Please select an appropriate rainfall dataset.')
        except AttributeError: 
            st.write('Please select an appropriate ENSO calculation method.')
    return st.session_state.nino34_mthly, st.session_state.elnino_full, st.session_state.elnino, st.session_state.lanina_full, st.session_state.lanina, st.session_state.rf 

