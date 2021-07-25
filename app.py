import streamlit as st
from streamlit.hashing import _CodeHasher

try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server

# Custom imports 
from main.multipage import MultiPage
from main import explore, feature, regression, references

def main():
    # Create an instance of the app 
    app = MultiPage()

    # Title of the main page
    st.set_page_config(layout='wide')
    st.title("Drought Index to Climate Index Explorer")
    st.sidebar.title(":floppy_disk: Dashboard")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Add all your applications (pages) here
    app.add_page("Data exploration", explore.app)
    app.add_page("Regression", regression.app)
    app.add_page("Feature Page", feature.app)
    app.add_page("References", references.app)

    # The main app
    app.run()

if __name__ == "__main__":
    main()
