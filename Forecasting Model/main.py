import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client, Client
from statsforecast import StatsForecast
from statsforecast.models import CrostonOptimized

@st.cache_resource
def init_connection():
    url = st.secrets['supabase_url']
    api_key = st.secrets['supabase_key']
    
    client:Client = create_client(url,api_key)
    
    return client

supabase = init_connection()

@st.cache_resource(ttl=600)
def run_query():
    return supabase.table('car_table').select('*').execute()

@st.cache_resource(ttl=600)
def create_dataframe():
    rows = run_query()
    df = pd.json_normalize(rows.data)
    df['volume'] = df['volume'].astype(int)

    return df

@st.cache_resource
def plot_volume(ids):
    fig, ax = plt.subplots()

    df['volume'] = df['volume'].astype(int)

    x = df[df["parts_id"] == 2674]['date']

    for id in ids:
        ax.plot(x,
                df[df['parts_id'] == id]['volume'], label=id)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.legend(loc='best')
    fig.autofmt_xdate()

    st.pyplot(fig)


@st.cache_resource
def format_dataset(ids):
    model_df = df[df['parts_id'].isin(ids)]
    model_df = model_df.drop(['id'], axis=1)
    model_df.rename({"parts_id": "unique_id", "date": "ds",
                    "volume": "y"}, axis=1, inplace=True)

    return model_df


@st.cache_resource
def create_sf_object(model_df):
    models = [CrostonOptimized()]
    
    sf = StatsForecast(
        df=model_df,
        models=models,
        freq="MS",
        n_jobs=-1
    )
    
    return sf

@st.cache_resource(show_spinner='Making Predictions')
def make_predictions(ids, horizon):
    model_df = format_dataset(ids)
    
    sf = create_sf_object(model_df)
    
    forecast_df = sf.forecast(h=horizon)
    
    return forecast_df.to_csv(header=True)

if __name__ == "__main__":
    st.title("Forecast product demand")

    df = create_dataframe()

    st.subheader("Select a product")
    product_ids = st.multiselect(
        "Select product ID", options=df['parts_id'].unique())

    plot_volume(product_ids)

    with st.expander("Forecast"):
        if len(product_ids) == 0:
            st.warning("Select at least one product ID to forecast")
        else:
            horizon = st.slider("Horizon", 1, 12, step=1)

            forecast_btn = st.button("Forecast", type="primary")
            
            if forecast_btn:
                csv_file = make_predictions(product_ids, horizon)
                
                st.download_button(
                    label='Download Prediction',
                    data=csv_file,
                    file_name='predictions.csv',
                    mime='text/csv'
                )
            
    

