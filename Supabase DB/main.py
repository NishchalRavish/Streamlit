import pandas as pd
import streamlit as st
from supabase import create_client, Client

@st.cache_resource
def initialize_connection():
    url = st.secrets['supabase_url']
    api_key = st.secrets['supabase_key']
    
    client: Client = create_client(url, api_key)
    
    return client

supabase = initialize_connection()

st.title('Querying the DB')

@st.cache_resource(ttl=600)
def run_query():
    return supabase.table('car_table').select('*').execute()

rows = run_query()

df = pd.json_normalize(rows.data)
st.write(df)