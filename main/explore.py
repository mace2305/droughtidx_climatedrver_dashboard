# sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib')))
import sys, os
try: 
    from main import load, visuals, processing
except:
    sys.path.append('..')
    from main import load, visuals, processing
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import streamlit as st
import numpy as np
state = st.session_state

def app():
    sns.set_context('poster')
    sns.set_style('white')
    mpl.rcParams['figure.figsize'] = [12.0, 8.0]
    mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['savefig.dpi'] = 200
    pos_a, pos_b = st.beta_columns(2)
    with pos_a: st.markdown("#### Other drought indices to include: \nSPEI, PDSI, STI")
    with pos_b: st.markdown("#### Other climate indices: \nIOD, MJO, PDO")
    st.write('\n')

    pos1, pos2, pos3 = st.beta_columns(3)
    pos4, pos5, pos6, _ = st.beta_columns((4, 5, 6, 2))
    pos7, pos8, pos9, pos9b = st.beta_columns(4)
    pos10, _ = st.beta_columns((18,1))
    pos11, _ = st.beta_columns((18,1))
    pos12, _ = st.beta_columns((18,1))


    ### ENSO calculation method sidebar selection
    option_sidebar_ENSO_def = st.sidebar.selectbox('ENSO calculation method',
        ('Timbal, Turkington, Rahmat, 2019 (OISST)', 'N.A.'),
        key='current_sidebar_ENSO_def')


    ### Rainfall dataset sidebar selection
    option_sidebar_rainfall_ds = st.sidebar.selectbox('Rainfall dataset',
        ('GPCP', 'N.A.'),
        key='current_sidebar_rainfall_ds')


    # Initialize sidebar states
    if 'sidebar_ENSO_def' not in state:
        state.sidebar_ENSO_def = option_sidebar_ENSO_def
        state.sidebar_rainfall_ds = option_sidebar_rainfall_ds


    # Detect sidebar state changes
    if state.sidebar_ENSO_def != option_sidebar_ENSO_def or state.sidebar_rainfall_ds != option_sidebar_rainfall_ds:
        with pos1:
            st.write('Loaded before were:')
            st.write(state.sidebar_ENSO_def)
            st.write(state.sidebar_rainfall_ds)
        with pos1: st.subheader('You have changed dataset parameters, please reload and generate new images where necessary. Below plots are from the older configuration!')


    # Dataset loading
    with pos4: 
        if st.button('Load datasets'):
            state.nino34_mthly, state.elnino_full, state.elnino, state.lanina_full, state.lanina, state.rf = load.precheck_loaded_datasets()
            st.write('Datasets loaded.')


    ### ENSO phase selector dropbox
    def enso_phase_selected():
        if option_ENSO_phase == 'El Nino': ENSO_phase = state.elnino
        elif option_ENSO_phase == 'La Nina': ENSO_phase = state.lanina
        if option_ENSO_phase == 'Weak-mod El Nino': ENSO_phase = processing.weakmod_elnino(state.elnino)
        elif option_ENSO_phase == 'Weak-mod La Nina': ENSO_phase = processing.weakmod_lanina(state.lanina)
        if option_ENSO_phase == 'Strong El Nino': ENSO_phase = processing.strong_elnino(state.elnino)
        elif option_ENSO_phase == 'Strong La Nina': ENSO_phase = processing.strong_lanina(state.lanina)
        return ENSO_phase


    def clean_generated_SPI_plots():
        try: 
            del state.scatter_pwhole
            del state.scatter_p_en
            del state.scatter_p_ln
            del state.time_series_whole
            del state.monthly_spi_boxplots
            del state.SPI_ens_corr_heatmap
        except: print('not able to delete all')


    with pos6:
        ### SPI timescale
        option_spi_timescale = st.select_slider('SPI timescale',
            options=[1,'3 (default)',6,12,48], value='3 (default)', )

        ### Lat/lon interval selection
        option_lat_interval = st.select_slider(f'{option_sidebar_rainfall_ds} lat/lon interval selection',
            options=[25,15,10,5,2,'Whole-length'], value='Whole-length', 
            key='current_lat_interval')


    ### Generating mean/std/kurtosis
    if 'mean' not in state:
        state.mean, state.std, state.kurtosis = None, None, None


    with pos5: 
        option_ENSO_phase = st.radio('ENSO phase', ('El Nino', 'La Nina'))
        if st.button('Generate mean, std, kurtosis & skewness of SPI'): 
            state.nino34_mthly, state.elnino_full, state.elnino, state.lanina_full, state.lanina, state.rf = load.precheck_loaded_datasets(warning_prompt=True)
            state.spi, state.lat_slices = processing.get_SPI_sliced(state.rf, option_lat_interval)
            state.EN34df = processing.dataframe_ENSO_spi(state.spi, state.rf, state.nino34_mthly, state.elnino, state.lanina)
            with st.spinner("Processing data..."):
                state.mean, state.std, state.kurtosis, state.skewness = visuals.generate_mean_std_kurtosis(state.rf, state.EN34df, enso_phase_selected(), option_spi_timescale)
                state.gen_enso_spi_enso_phase_selected = option_ENSO_phase

    with pos6: 
        ### Generating SPI plots
        if st.button('Generate SPI products'):
            if state.generate_spi_before: clean_generated_SPI_plots()

            state.nino34_mthly, state.elnino_full, state.elnino, state.lanina_full, state.lanina, state.rf = load.precheck_loaded_datasets(warning_prompt=True)
            state.spi, state.lat_slices = processing.get_SPI_sliced(state.rf, option_lat_interval)
            state.EN34df = processing.dataframe_ENSO_spi(state.spi, state.rf, state.nino34_mthly, state.elnino, state.lanina)
            state.scatter_pwhole, state.scatter_p_en, state.scatter_p_ln = visuals.generate_EN34_SPI_scatterplots(state.EN34df)

            state.option_spi_timescale = option_spi_timescale
            state.option_lat_interval = option_lat_interval
            state.SPI_scatterplots_generated = True

            state.time_series_whole = visuals.plot_SPI_ENSO(state.spi, state.rf, state.elnino, state.lanina)
            state.SPI_timeseries_generated = True

            if option_lat_interval == 'Whole-length':
                title = 'Whole length of latitude provided'
                state.monthly_spi_boxplots = visuals.generate_monthlySPI3_boxplot(
                    state.EN34df, title)
            if option_lat_interval != 'Whole-length':
                state.SPI_ens_corr_heatmap = visuals.multiple_latitudeintervals_correlation_heatmap(
                    state.EN34df, interval=option_lat_interval)

            state.SPI_boxplots_generated = True
            state.SPI_ens_corr_heatmap_generated = True
            state.generate_spi_before = True

        st.markdown('**Includes SPI-ENSO scatterplots, timeseries, boxplots and correlation heatmap.**')


    if 'option_spi_timescale' not in state:
        state.option_spi_timescale = option_spi_timescale
        state.option_lat_interval = option_lat_interval
        state.SPI_scatterplots_generated = False
        state.SPI_timeseries_generated = False
        state.SPI_boxplots_generated = False
        state.SPI_ens_corr_heatmap_generated = False
        state.SPI_ens_corr_heatmap_generated = False
        state.generate_spi_before = False


    if (state.option_spi_timescale != option_spi_timescale or state.option_lat_interval != option_lat_interval) and (state.SPI_ens_corr_heatmap_generated == True):
        with pos3: st.subheader(f'Scatterplots/timeseries/boxplots are for SPI timescale: {state.option_spi_timescale} & Lat/lon interval: {state.option_lat_interval}')
    state.option_spi_timescale = option_spi_timescale


    # Plot retrieval zone
    try: 
        if state.mean != None:
            with pos5:
                st.subheader(state.gen_enso_spi_enso_phase_selected)
                st.write('Enlarge image by clicking expand image icon.')
            with pos7:st.pyplot(state.mean)
            with pos8:st.pyplot(state.std)
            with pos9:st.pyplot(state.kurtosis)
            with pos9b:st.pyplot(state.skewness)
    except: pass

    try: 
        if state.scatter_pwhole != None:
            with pos7:st.pyplot(state.scatter_pwhole)
            with pos8:st.pyplot(state.scatter_p_en)
            with pos9:st.pyplot(state.scatter_p_ln)
            with pos10:st.pyplot(state.time_series_whole)
            with pos11:st.pyplot(state.monthly_spi_boxplots)
            with pos12:
                if option_lat_interval == 'Whole-length':
                    st.subheader('To generate correlation heatmap, choose a latitude interval.')
                st.pyplot(state.SPI_ens_corr_heatmap) 
    except: pass


if __name__=='__main__':
    sns.set_context('poster')
    sns.set_style('white')
    # mpl.rcParams['figure.figsize'] = [12.0, 8.0]
    # mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['savefig.dpi'] = 400
    plt.subplots_adjust(left=0.4, right=0.8, bottom=0.4, top=0.8)
    # plt.tight_layout()

    option_sidebar_ENSO_def = ('Timbal, Turkington, Rahmat, 2019 (OISST)', 'N.A.')
    option_sidebar_ENSO_def_short = ('TTR2019', )
    option_sidebar_rainfall_ds = ('GPCP', 'N.A.')
    option_sidebar_rainfall_ds_short = ('GPCP', )
    current_sidebar_ENSO_def = option_sidebar_ENSO_def[0]
    current_sidebar_rainfall_ds = option_sidebar_rainfall_ds[0]
    current_sidebar_ENSO_def_short = option_sidebar_ENSO_def_short[0]
    current_sidebar_rainfall_ds_short = option_sidebar_rainfall_ds_short[0]

    def gen_SPI_product(spi, nino34_mthly, elnino, lanina, rf, current_lat_interval, current_spi_timescale, ENSO_phase, lat_slices=None):

        if current_lat_interval == 'Whole-length':
            EN34df = processing.dataframe_ENSO_spi(spi, rf, nino34_mthly, elnino, lanina)

            # gen mean, std, kurtosis
            means, stds, kurtosiss, skewnesss = list(), list(), list(), list()
            for i, e_ in enumerate(ENSO_phase):
                mean, std, kurtosis, skewness = visuals.generate_mean_std_kurtosis(rf, EN34df, e_)
                means.append(mean); stds.append(std); kurtosiss.append(kurtosis), skewnesss.append(skewness)

            scatter_pwhole, scatter_p_en, scatter_p_ln = visuals.generate_EN34_SPI_scatterplots(EN34df)
            time_series_whole = visuals.plot_SPI_ENSO(spi, rf, elnino, lanina)
            title = 'Whole length of latitude provided'
            monthly_spi_boxplots = visuals.generate_monthlySPI3_boxplot(EN34df, title, current_spi_timescale)
            SPI_ens_corr_heatmap = None
            
        else: # spi is a LIST of spi arrays
            scatter_pwhole, scatter_p_en, scatter_p_ln = None, None, None
            means, stds, kurtosiss, skewnesss = None, None, None, None
            time_series_whole = visuals.plot_SPI_ENSO_multiple_latintervals(spi, rf, lat_slices, elnino, lanina)
            monthly_spi_boxplots = list()
            for lat_range, s in zip(lat_slices, spi):
                title = lat_range
                EN34df_s = processing.dataframe_ENSO_spi(s, rf, nino34_mthly, elnino, lanina, )
                monthly_spi_boxplots.append(visuals.generate_monthlySPI3_boxplot(EN34df_s, title, current_spi_timescale))
            SPI_ens_corr_heatmap = visuals.correlation_heatmap2(spi, rf, nino34_mthly, elnino, lanina, lat_slices, current_spi_timescale)

        return scatter_pwhole, scatter_p_en, scatter_p_ln, time_series_whole, \
            monthly_spi_boxplots, SPI_ens_corr_heatmap, means, stds, kurtosiss, skewnesss

    def get_stat_products(current_spi_timescale):
        dir = f'../cached_images/spi{current_spi_timescale}/'
        os.makedirs(dir, exist_ok=True)
        fn1 = f'{current_sidebar_rainfall_ds_short}'
        fn2 = f'{current_sidebar_ENSO_def_short}'
        fn1 = dir + fn1 + '_'
        print(fn1)

        # load datasets
        nino34_mthly, elnino_full, elnino, lanina_full, lanina, rf = load.load_datasets(
            current_sidebar_ENSO_def, current_sidebar_rainfall_ds)
        ENSO_phase = (elnino, processing.weakmod_elnino(elnino), processing.strong_elnino(elnino), 
        lanina, processing.weakmod_lanina(lanina), processing.strong_lanina(lanina)) 
        ENSO_phase_n = ['elnino', 'weakmod_elnino', 'strong_elnino', 
        'lanina', 'weakmod_lanina', 'strong_lanina']

        option_lat_interval = [25,15,10,5,'Whole-length'] # Lat/lon interval selection
        for current_lat_interval in option_lat_interval:
            current_lat_interval = 'Whole-length'
            spi, lat_slices = processing.get_SPI_sliced(rf, current_lat_interval)

            if current_lat_interval == 'Whole-length': 

                fscwhole, fsc_en, fsc_ln, f_ts, fmbp, _, means, stds, kurtosiss, skewnesss = gen_SPI_product(spi, nino34_mthly, elnino, lanina, rf, current_lat_interval, current_spi_timescale, ENSO_phase)

                for i, (mean, std, kurtosis, skewness) in enumerate(zip(means, stds, kurtosiss, skewnesss)):
                    mean.savefig(f'{fn1}_{ENSO_phase_n[i]}_mean', bbox_inches='tight', pad_inches=1)
                    std.savefig(f'{fn1}_{ENSO_phase_n[i]}_std', bbox_inches='tight', pad_inches=1)
                    kurtosis.savefig(f'{fn1}_{ENSO_phase_n[i]}_kurtosis', bbox_inches='tight', pad_inches=1)
                    skewness.savefig(f'{fn1}_{ENSO_phase_n[i]}_skewness', bbox_inches='tight', pad_inches=1)
                sys.exit()

                fscwhole.savefig(f'{fn1}{current_lat_interval}_{fn2}_sc', bbox_inches = "tight", pad_inches=.5)
                fsc_en.savefig(f'{fn1}{current_lat_interval}_{fn2}_scen', bbox_inches = "tight", pad_inches=.5)
                fsc_ln.savefig(f'{fn1}{current_lat_interval}_{fn2}_scln', bbox_inches = "tight", pad_inches=.5)
                f_ts.savefig(f'{fn1}{current_lat_interval}_{fn2}_ts', bbox_inches = "tight", pad_inches=.5)
                fmbp.savefig(f'{fn1}{current_lat_interval}_{fn2}_mbp', bbox_inches = "tight", pad_inches=.5)

            elif current_lat_interval != 'Whole-length': # per latitude cut

                _, _, _, f_ts, fmbp, fchm, _, _, _, = gen_SPI_product(spi, nino34_mthly, elnino, lanina, rf, current_lat_interval, current_spi_timescale, ENSO_phase, lat_slices)

                for bp, lat_sl in zip(fmbp, lat_slices):
                    lat_sl_string = '_'.join([str(int(i)) for i in lat_sl])
                    bp.savefig(f'{fn1}{current_lat_interval}_{fn2}_mbp_{lat_sl_string}', bbox_inches='tight', pad_inches=1)

                for ts, lat_sl in zip(f_ts, lat_slices):
                    lat_sl_string = '_'.join([str(int(i)) for i in lat_sl])
                    ts.savefig(f'{fn1}{current_lat_interval}_{fn2}_ts_{lat_sl_string}', bbox_inches='tight', pad_inches=1)

                fchm.savefig(f'{fn1}{current_lat_interval}_{fn2}_chm', bbox_inches='tight', pad_inches=1)
            # break
        
    option_spi_timescale = [3,1,6,12,48] # SPI timescale
    for current_spi_timescale in option_spi_timescale: get_stat_products(current_spi_timescale)
        
        




# To-do:
# DONE* NO latitude interval support yet for the plotting of
# \\ scatter, time-series
# 0!. experimental page: put the multi-year (6-year stretches) time series of 5-interval latitudes vs EN34
# 0. add blurb to open sidebar to manage data sources
# 0b. add text of date-range incorporated
# 1. if st.button('Load assumptions blurb, to-dos and limitations of above techniques.'):
# 2. auto-detect latitude limits and give following slider
# appointment = st.slider(
#     "Set latitude range (limits determined from datasets chosen)",
#     0.0, 100.0, (25.0, 75.0))
# st.write("You're scheduled for:", appointment)
# 3. ask user to send in a improvements list --> add in one for opinion, 
# second input bar for email
# third for name
# 4. can try to pre-gen some plots, esp the basic stat ones. also pre-gen calculations like SPI of whole GPCP dataset, of sliced GPCP dataset by XX latitudes


# state.old_state = [(state[f'{key}']) for key in state.keys() if key in 
#     ['my_enso_definition', 'my_rf_dataset']
# ]

# if st.button('asd'):
#     asd = [(key, state[f'{key}']) for key in state.keys()]
#     st.write(asd)
#     print(asd)

# m = st.markdown("""
# <style>
# div.stButton > button:first-child {
#     background-color: #44c767;
# }
# </style>""", unsafe_allow_html=True)

