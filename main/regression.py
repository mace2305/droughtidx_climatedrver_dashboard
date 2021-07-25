from main import SessionState, processing, load, utils
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import matplotlib as mpl
import seaborn as sns; sns.set()
import statsmodels.api as sm
from stqdm import stqdm

def app():
    sns.set_context('poster')
    sns.set_style('white')
    mpl.rcParams['figure.figsize'] = [12.0, 8.0]
    mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['savefig.dpi'] = 200
    pos1, pos2 = st.beta_columns(2)
    pos3, pos3b, pos3c = st.beta_columns(3)
    _, pos4, _ = st.beta_columns((2, 5, 2))
    pos4a, _ = st.beta_columns((20,1))
    pos4b, pos5b = st.beta_columns(2)
    pos6, _ = st.beta_columns((20,1))
    pos7, _ = st.beta_columns((20,1))
    pos8, _ = st.beta_columns((20,1))
    pos9, _ = st.beta_columns((20,1))
    session = SessionState.get(run_id=0)
    state = st.session_state

    def reset_button():
        if st.button("Reset"):
            session.run_id += 1 

    # dataframe = dataframe_ENSO_spi(spi3_whole, gpcp)
    def check_dataframe_existence():
        if 'EN34df' not in state: 
            state.nino34_mthly, state.elnino_full, state.elnino, state.lanina_full, state.lanina, state.rf = load.precheck_loaded_datasets(warning_prompt=True)
            state.spi, state.lat_slices = processing.get_SPI_sliced(state.rf, state.current_lat_interval)
            state.EN34df = processing.dataframe_ENSO_spi(state.spi, state.rf, state.nino34_mthly, state.elnino, state.lanina)
        

    # prepare data
 
    def execute_user_config():
        # if state.option_valid_split:
        #     state.train_split = state.option_train_val_test_split_ratio[0]
        #     state.val_split = state.option_train_val_test_split_ratio[1]-state.option_train_val_test_split_ratio[0]
        #     state.test_split = 100-state.option_train_val_test_split_ratio[1]
        #     with pos2:st.markdown(f"**Train-val-test split is now {state.train_split}-{state.val_split}-{state.test_split}**")
        # else:
        #     state.train_split = state.option_train_test_split_ratio
        #     state.val_split = 0
        #     state.test_split = 100-state.option_train_test_split_ratio
        #     with pos2:st.markdown(f"**Train-test split is now {state.train_split}-{state.test_split}**")
        state.train_split, state.val_split, state.test_split, string = processing.determine_split_ratio(
            state.option_valid_split, state.option_train_val_test_split_ratio, state.option_train_test_split_ratio)
        with pos2:st.markdown(string)
        check_dataframe_existence()
        exog_data = state.EN34df.EN34
        state.exog_full = sm.add_constant(exog_data)
        state.endog_full = state.EN34df.spi3
        state.endog_train, state.endog_val, state.endog_test, state.exog_train, state.exog_val, state.exog_test, state.train_s, state.test_s = processing.train_test_split(
            state.endog_full, state.exog_full, state.train_split, state.val_split, state.option_valid_split, state.option_exog)


    with pos1:
        state.option_valid_split = st.checkbox("Include validation set")
        st.button("Generate dataset split", on_click=execute_user_config, key='dataset_split_already')
        if not state.dataset_split_already:st.markdown('_Dataset has not been split._')
        else:st.markdown('_Dataset has been split as per specs._')
        st.write('\n')


    with pos2: 
        if not state.option_valid_split: 
            state.option_train_test_split_ratio=st.slider("Train-test set percentage (%)", 0,100,80, step=5, )
            state.option_train_val_test_split_ratio=None
        else:
            state.option_train_val_test_split_ratio=st.slider("Train-val-test set split (%)", 0, 100, (60,80), step=5, )
            state.option_train_test_split_ratio=None


    with pos3:
        st.radio('ARIMA or SARIMA?',('ARIMA', 'SARIMA'), key='option_ARIMA_SARIMA')
        if state.option_ARIMA_SARIMA == 'ARIMA':
            st.markdown('###### If ARIMA(X), P, D, Q and s parameters will not be available.')
            st.write('\n')

    with pos3b:
        st.radio('Include exogenous variable?', ('Yes', 'No'), key='option_exog')
        if state.option_exog:pass
    with pos3c:st.radio('Auto grid-search for best config or choose your own?',('AUTO', 'MANUAL'), key='option_SARIMAX_config')


    def gen_dataframe_permutations():
        state.permutations_generated, state.permutations_generated_seasonal, df = processing.generate_order_triplets(
            state.option_ARIMA_SARIMA, state.option_SARIMAX_config, state.items_SARIMAX)
        with pos4:st.dataframe(df)


    if 'submitted_permutations' not in state: state.submitted_permutations=False
    if 'submitted_bestcombination' not in state: state.submitted_bestcombination=False
    with pos4: 
        if state.option_ARIMA_SARIMA == 'SARIMA' and state.option_SARIMAX_config == 'AUTO':
            with st.beta_expander("AUTO: designate SARIMA permutations", expanded=True):
                st.write('Set parameters if intending for auto-SARIMAX function to search for best configuration.')
                with st.form("SARIMAX permutations"):
                    st.number_input("p:", 0, 20, 1, key='SARIMAX_p')
                    st.number_input("d:", 0, 20, 0, key='SARIMAX_d')
                    st.number_input("q:", 0, 20, 0, key='SARIMAX_q')
                    st.number_input("P:", 0, 20, 1, key='SARIMAX_P')
                    st.number_input("D:", 0, 20, 0, key='SARIMAX_D')
                    st.number_input("Q:", 0, 20, 0, key='SARIMAX_Q')
                    st.number_input("s:", 0, 20, 12, key='SARIMAX_s')
                    state.submitted_permutations = st.form_submit_button("Submit configurations.")
            if state.submitted_permutations:
                state.keys_SARIMAX = ['SARIMAX_p', 'SARIMAX_d', 'SARIMAX_q',
                        'SARIMAX_P', 'SARIMAX_D', 'SARIMAX_Q', 'SARIMAX_s']
                state.items_SARIMAX = [state[key] for key in state if key in state.keys_SARIMAX]
                items = state.items_SARIMAX
                items_not0 = [i+1 for i in items]
                string = ' x '.join([str(i) for i in items_not0[:-1]]) + f', with s of {items_not0[-1]-1}'
                st.markdown(f"Permutations to be test: **{np.prod(items_not0[:-1])}** =  --> {string}")
                gen_dataframe_permutations()

        elif state.option_ARIMA_SARIMA == 'SARIMA' and state.option_SARIMAX_config == 'MANUAL':
            with st.beta_expander("AUTO: designate SARIMA permutations", expanded=True):
                st.write('Set parameters if intending for auto-SARIMAX function to search for best configuration.')
                with st.form("SARIMAX permutations"):
                    st.number_input("p:", 0, 20, 1, key='SARIMAX_p')
                    st.number_input("d:", 0, 20, 0, key='SARIMAX_d')
                    st.number_input("q:", 0, 20, 0, key='SARIMAX_q')
                    st.number_input("P:", 0, 20, 1, key='SARIMAX_P')
                    st.number_input("D:", 0, 20, 0, key='SARIMAX_D')
                    st.number_input("Q:", 0, 20, 0, key='SARIMAX_Q')
                    st.number_input("s:", 0, 20, 12, key='SARIMAX_s')
                    state.submitted_bestcombination = st.form_submit_button("Submit configurations.")
            if state.submitted_bestcombination:
                state.keys_SARIMAX = ['SARIMAX_p', 'SARIMAX_d', 'SARIMAX_q',
                        'SARIMAX_P', 'SARIMAX_D', 'SARIMAX_Q', 'SARIMAX_s']
                state.items_SARIMAX = [state[key] for key in state if key in state.keys_SARIMAX]
                items = state.items_SARIMAX
                st.markdown(f"Testing order: {items[0],items[1],items[2]}x{items[3],items[4],items[5]}{items[6]}")
                gen_dataframe_permutations()

        elif state.option_ARIMA_SARIMA == 'ARIMA' and state.option_SARIMAX_config == 'AUTO':
            with st.beta_expander("AUTO: designate ARIMA permutations", expanded=True):
                st.write('Set parameters if intending for auto-ARIMA function to search for best configuration.')
                with st.form("ARIMA permutations"):
                    st.number_input("p:", 0, 20, 1, key='ARIMA_p')
                    st.number_input("d:", 0, 20, 0, key='ARIMA_d')
                    st.number_input("q:", 0, 20, 0, key='ARIMA_q')
                    state.submitted_permutations = st.form_submit_button("Submit configurations.")
            if state.submitted_permutations:
                state.keys_SARIMAX = ['ARIMA_p', 'ARIMA_d', 'ARIMA_q',]
                state.items_SARIMAX = [state[key] for key in state if key in state.keys_SARIMAX]
                items = state.items_SARIMAX
                items_not0 = [i+1 for i in items]
                string = ' x '.join([str(i) for i in items_not0])
                st.markdown(f"Permutations to be test: **{np.prod(items_not0)}** =  --> {string}")
                gen_dataframe_permutations()

        elif state.option_ARIMA_SARIMA == 'ARIMA' and state.option_SARIMAX_config == 'MANUAL':
            with st.beta_expander("MANUAL: indicate ARIMA parameters", expanded=True):
                st.write('Set parameters if only intending to test 1 specific permutation.')
                with st.form("ARIMA best parameters"):
                    st.number_input("p:", 0, 20, 1, key='ARIMA_p')
                    st.number_input("d:", 0, 20, 0, key='ARIMA_d')
                    st.number_input("q:", 0, 20, 0, key='ARIMA_q')
                    state.submitted_bestcombination = st.form_submit_button("Submit configurations.")
            if state.submitted_bestcombination:
                state.keys_SARIMAX = ['ARIMA_p', 'ARIMA_d', 'ARIMA_q',]
                state.items_SARIMAX = [state[key] for key in state if key in state.keys_SARIMAX]
                items = state.items_SARIMAX
                st.markdown(f"Testing order: {items[0],items[1],items[2]}")
                gen_dataframe_permutations()


    # state.option_ARIMA_SARIMA \\ arima vs. sarima
    # state.option_SARIMAX_config \\ auto vs. manual
    # state.option_exog \\ X or no-X            
 
    # Step 5 — Fitting an ARIMA Time Series Model
    def autodetermine_optimal_SARIMAX_config(endog, exog,):
        best_param, best_param_seasonal, best_score = '', '', -1

        pdq = state.permutations_generated
        if state.option_ARIMA_SARIMA == 'ARIMA':seasonal_pdq = [state.permutations_generated_seasonal] # (0,0,0,0)

        elif state.option_ARIMA_SARIMA == 'SARIMA':seasonal_pdq = state.permutations_generated_seasonal

        for param in stqdm(pdq):
            for param_seasonal in stqdm(seasonal_pdq):
                try: 
                    res_aic, string=processing.run_SARIMAX(
                        endog, exog, param, param_seasonal, purpose='autodetermine', option_ARIMA_SARIMA=state.option_ARIMA_SARIMA, )
                    st.write(string)
                    if (res_aic < best_score) or (best_score == -1):
                        best_param, best_param_seasonal, best_score = param, param_seasonal, res_aic
                except:
                    raise
        return best_param, best_param_seasonal

    with pos4a:
        if st.button('Start analysis.'): 
            with st.spinner('Analyzing...'):
                try:
                    execute_user_config()
                    print(f'{utils.time_now()} - GOGOGO')
                    with st.spinner("Processing data..."):
                        if state.option_SARIMAX_config  == 'AUTO': 
                            state.best_param, state.best_param_seasonal = autodetermine_optimal_SARIMAX_config(state.endog_train, state.exog_train)
                        elif state.option_SARIMAX_config  == 'MANUAL':
                            state.best_param, state.best_param_seasonal = state.permutations_generated, state.permutations_generated_seasonal
                        state.plot_diagnostics, summary_table, state.results=processing.run_SARIMAX(
                            state.endog_train, state.exog_train, state.best_param, state.best_param_seasonal, purpose='best_model')    
                        state.summary_table = pd.read_html(summary_table, header=0, index_col=0)[0]
                    st.success(f'New products generated @: {utils.time_now()}')
                except: raise


    ## Step 6 — Validating Forecasts
    try:
        if state.option_valid_split:
            state.validation_onestep_fc, state.validation_onestep_fc_blurb, state.val_start_date, state.val_end_date = processing.one_step_validation_forecast(
                state.results, state.train_s, state.test_s, state.exog_val, state.endog_full, 
            )
    except ValueError: print('Need to regen dataset split.')
    except AttributeError: print('its hapen')


    # Plot retrieval zone
    try: 
        if state.plot_diagnostics != None:
            with st.spinner('Retrieving plots...'):
                with pos6:
                    st.markdown(f'### Best param: {state.best_param}, best seasonal param order: {state.best_param_seasonal}')
                    st.write('\n')
                with pos6:st.pyplot(state.plot_diagnostics)
                with pos7:st.table(state.summary_table)
                if state.option_valid_split:
                    with pos8:
                        st.markdown(f'## Validating over {state.val_start_date} to {state.val_end_date}')
                        st.pyplot(state.validation_onestep_fc)
                        st.markdown(f'### {state.validation_onestep_fc_blurb}')
    except: print('its happening4')
    