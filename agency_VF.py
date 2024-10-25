import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import warnings
from io import StringIO
import random
from pandas.tseries.offsets import QuarterEnd
from plotly.subplots import make_subplots
from pandas.tseries.offsets import QuarterEnd
from plotly.subplots import make_subplots
from PIL import Image  # Import PIL for handling images
from sklearn.linear_model import LinearRegression



# Streamlit configuration
st.set_page_config(
    layout="wide",
    page_title="Century 21 Market Dashboard",
    page_icon="century21_logo.ico",  # Use an ICO format for the favicon
    initial_sidebar_state="expanded"
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


st.markdown(
    """
    <style>
    /* Global font style */
    @import url('https://fonts.googleapis.com/css2?family=Century+Gothic&display=swap');
    body {
        font-family: 'Century Gothic', Arial, sans-serif;
        background-color: #f9f9f9; /* Light background */
    }

    /* Title style */
    h1 {
        font-size: 36px;
        font-weight: bold;
        color: #343534; /* Dark color for text */
        text-align: center;
        margin-top: 20px;
    }

    h2, h3 {
        color: #343534; /* Dark secondary text */
        font-weight: normal;
    }

    /* Sidebar style */
    .sidebar .sidebar-content {
        background-color: #b2a284; /* Light primary color */
        padding: 15px;
    }

    /* Sidebar header */
    .sidebar .sidebar-content h2 {
        color: #343534; /* Dark text for sidebar headers */
    }

    /* Sidebar logo styling */
    .sidebar img {
        width: 180px;
        margin-bottom: 10px;
    }

    /* Header style with logo */
    .header {
        position: sticky;
        top: 0;
        background-color: #ffffff;
        z-index: 1000;
        padding: 10px 0;
        border-bottom: 2px solid #b2a284; /* Light primary border underline */
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 30px;
    }
    .header img {
        width: 250px;
    }

    /* Button styles */
    .stButton>button {
        background-color: #b2a284; /* Light primary color */
        color: #ffffff;
        border: none;
        padding: 8px 20px;
        border-radius: 5px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #343534; /* Darker hover state */
        color: #ffffff;
    }

    /* Multiselect selected tag styles */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #b2a284 !important; /* Update the tag background color */
        color: white !important; /* Ensure the text is white for contrast */
    }

    .stMultiSelect .st-bs {
        background-color: #b2a284 !important; /* Background for the selection box */
        color: white !important;
    }

    /* Slider styles */
    .stSlider>div>div {
        background-color: transparent !important; /* Transparent slider background */
    }
    
    .stSlider .css-1aumxhk {
        background-color: #b2a284 !important; /* Light color for the slider handle */
    }

    /* Multiselect, dropdown, and date picker styles */
    .stMultiSelect .st-multiselect .st-bn, .stSelectbox .st-bn, .stDateInput>div>input {
        background-color: #b2a284 !important;
        color: #ffffff !important;
    }

    /* Dataframe styling */
    .dataframe tr, .dataframe td, .dataframe th {
        border: none;
        color: #343534;
        font-size: 14px;
    }
    
    /* Styled box for dynamic display */
    .dynamic-box {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }

    /* Text inside dynamic box */
    .dynamic-box p {
        color: #343534;
        font-size: 14px;
        line-height: 1.5;
    }

    /* Highlighted metrics in box */
    .dynamic-box .highlight {
        color: #b2a284; /* Light primary color for emphasis */
        font-weight: bold;
    }

    /* Table styles */
    .stTable tr {
        background-color: transparent;  /* Remove table row background */
    }

    .stTable th {
        background-color: transparent; /* Transparent table headers */
        color: #343534; /* Dark text for table headers */
        font-size: 16px;
    }

    .stTable td {
        font-size: 14px;
        color: #343534;
    }

    /* Chart titles and labels */
    .stPlotlyChart .main-svg .plot-container .xtitle, .ytitle {
        color: #343534 !important; /* Dark text on charts */
    }

    /* Dropdown selections styling */
    .stSelectbox select {
        background-color: #b2a284 !important;
        color: #ffffff !important;
        border: none;
    }

    /* Checkbox styling */
    .stCheckbox input {
        accent-color: #b2a284; /* Light color for checkboxes */
    }

    /* Hover styles for clickable elements */
    .stButton>button, .stCheckbox label, .stMultiSelect .st-bn {
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    .stButton>button:hover, .stMultiSelect .st-bn:hover {
        background-color: #343534;
        color: #ffffff;
    }

    /* Sidebar elements styling */
    .sidebar .stSlider>div, .sidebar .stMultiSelect .st-bn, .sidebar .stCheckbox input {
        background-color: #b2a284;
        color: #ffffff;
    }

    /* Customize scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-thumb {
        background-color: #b2a284;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background-color: #343534;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)



# Try using Streamlit's st.image() for logo display
try:
    logo = Image.open("Ever-One_RGB_C_GB.png")  # Ensure this is the correct path
    st.image(logo, caption="", width=500)
except FileNotFoundError:
    st.error("L'image 'Ever-One_RGB_C_GB.png' est introuvable. Veuillez vérifier le chemin du fichier.")



# Title of the application with Century 21 logo at the top
st.markdown("""
<div class="header">
</div>
<h1>🏠 Tableau de Bord d'Analyse pour les agences Clermont A, Masure</h1>
""", unsafe_allow_html=True)

# --------------------------------------
# Step 1: Load the agency data from Streamlit secrets
# --------------------------------------
# Fetching the agency data from Streamlit secrets
data = st.secrets["data"]["my_agency_data"]
df_agency = pd.read_csv(StringIO(data))

# Display the agency data for verification
st.write(df_agency)

# --------------------------------------
# Display Logos and Images in Sidebar
# --------------------------------------
st.sidebar.header(" ")

# Load and Display Logos in Sidebar with Consistent Sizing
logo_century = Image.open('century_logo_clean.png')

# Display the logos in the sidebar
st.sidebar.image(logo_century, caption='Century 21', width=90)

# --------------------------------------
# Step 2: Data Loading with Delimiter Detection (Market Data)
# --------------------------------------
@st.cache_data
def load_market_data():
    """Load the market data from 'all_cities_belgium.csv'."""
    try:
        # Check delimiter and load market data
        with open('all_cities_belgium.csv', 'r', encoding='utf-8-sig') as f:
            first_line = f.readline()
            if ',' in first_line and '\t' not in first_line:
                sep = ','
            elif '\t' in first_line:
                sep = '\t'
            else:
                sep = ','
        df_market = pd.read_csv('all_cities_belgium.csv', sep=sep, encoding='utf-8-sig')
        return df_market
    except FileNotFoundError:
        st.error("Fichier 'all_cities_belgium.csv' non trouvé dans le dépôt.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier 'all_cities_belgium.csv' : {e}")
        st.stop()

# Load the market data
df_market = load_market_data()

# Display market data for verification
st.write(df_market.head())  # This is for testing. You can remove it later.

# --------------------------------------
# Step 3: Data Cleaning and Preprocessing
# --------------------------------------
def preprocess_data(df_market, df_agency):
    # Standardize column names for consistency: lowercase
    df_market.columns = df_market.columns.str.strip().str.lower()
    df_agency.columns = df_agency.columns.str.strip().str.lower()


    # Market Data Preprocessing
    if 'date' in df_market.columns:
        df_market['date'] = pd.to_datetime(df_market['date'], errors='coerce', dayfirst=True)
    else:
        st.error("Colonne 'date' non trouvée dans 'all_cities_belgium.csv'. Veuillez vérifier que la colonne 'date' existe et est correctement orthographiée.")
        st.stop()

    df_market['year'] = df_market['date'].dt.year
    df_market['quarter'] = df_market['date'].dt.quarter
    df_market['month'] = df_market['date'].dt.month

    # Split 'map' column into 'latitude' and 'longitude'
    if 'map' in df_market.columns:
        try:
            df_market[['latitude', 'longitude']] = df_market['map'].str.strip().str.split(',', expand=True).astype(float)
        except Exception as e:
            st.error(f"Erreur lors du traitement de la colonne 'map' dans 'all_cities_belgium.csv' : {e}")
            st.stop()
    else:
        st.error("Colonne 'map' non trouvée dans 'all_cities_belgium.csv'.")
        st.stop()

    # Clean 'prix médian' column
    if 'prix médian' in df_market.columns:
        df_market['prix médian'] = df_market['prix médian'].replace('[\€,]', '', regex=True).astype(float)
        df_market['prix médian'].fillna(df_market['prix médian'].median(), inplace=True)
    else:
        st.error("Colonne 'prix médian' non trouvée dans 'all_cities_belgium.csv'.")
        st.stop()

    # Clean 'nombre de ventes' column
    if 'nombre de ventes' in df_market.columns:
        df_market['nombre de ventes'] = pd.to_numeric(df_market['nombre de ventes'], errors='coerce').fillna(0).astype(int)
    else:
        st.error("Colonne 'nombre de ventes' non trouvée dans 'all_cities_belgium.csv'.")
        st.stop()

    # Drop rows where 'commune' is missing
    if 'commune' in df_market.columns:
        df_market.dropna(subset=['commune'], inplace=True)
        df_market.reset_index(drop=True, inplace=True)
    else:
        st.error("Colonne 'commune' non trouvée dans 'all_cities_belgium.csv'.")
        st.stop()

    # Agency Data Preprocessing
    if 'date de transaction' in df_agency.columns:
        df_agency['date'] = pd.to_datetime(df_agency['date de transaction'], errors='coerce', dayfirst=True)
    else:
        st.error("Colonne 'date de transaction' non trouvée dans 'my_agency_data.csv'.")
        st.stop()

    df_agency['year'] = df_agency['date'].dt.year
    df_agency['quarter'] = df_agency['date'].dt.quarter
    df_agency['month'] = df_agency['date'].dt.month

    # Split 'map' column into 'latitude' and 'longitude'
    if 'map' in df_agency.columns:
        try:
            df_agency[['latitude', 'longitude']] = df_agency['map'].str.strip().str.split(',', expand=True).astype(float)
        except Exception as e:
            st.error(f"Erreur lors du traitement de la colonne 'map' dans 'my_agency_data.csv' : {e}")
            st.stop()
    else:
        st.error("Colonne 'map' non trouvée dans 'my_agency_data.csv'.")
        st.stop()

    # Assuming each row is a transaction; set 'nombre de ventes' to 1
    df_agency['nombre de ventes'] = 1

    # Rename 'communes' to 'commune' for consistency
    if 'communes' in df_agency.columns:
        df_agency.rename(columns={'communes': 'commune'}, inplace=True)
    else:
        st.error("Colonne 'communes' non trouvée dans 'my_agency_data.csv'.")
        st.stop()

    # Drop rows where 'commune' is missing
    if 'commune' in df_agency.columns:
        df_agency.dropna(subset=['commune'], inplace=True)
        df_agency.reset_index(drop=True, inplace=True)
    else:
        st.error("Colonne 'commune' non trouvée dans 'my_agency_data.csv'.")
        st.stop()

    # Merge geographic data if missing in agency data
    if df_agency['latitude'].isnull().any() or df_agency['longitude'].isnull().any():
        df_agency = df_agency.merge(
            df_market[['commune', 'latitude', 'longitude']].drop_duplicates(),
            on='commune',
            how='left',
            suffixes=('', '_market')
        )
        df_agency['latitude'] = df_agency['latitude'].fillna(df_agency['latitude_market'])
        df_agency['longitude'] = df_agency['longitude'].fillna(df_agency['longitude_market'])
        df_agency.drop(columns=['latitude_market', 'longitude_market'], inplace=True)

    # Ensure no missing values in critical columns
    df_agency.dropna(subset=['latitude', 'longitude', 'date'], inplace=True)

    return df_market, df_agency

df_market, df_agency = preprocess_data(df_market, df_agency)

# --------------------------------------
# Step 4: Sidebar Filters for Interactivity
# --------------------------------------
# Sidebar Header and Filters
st.sidebar.header("📊 Filtres")

# Function to generate unique keys
def generate_key(*args):
    """Generates a unique key based on the provided arguments."""
    return "_".join(map(str, args))

st.markdown(
    """
    <style>
    /* Custom styles for sidebar section */
    .sidebar .sidebar-content {
        background-color: #343534 !important; /* Darker color for the entire sidebar */
        color: #ffffff !important; /* White text for better contrast */
    }

    /* Remove background from sliders */
    .stSlider > div > div {
        background-color: transparent !important; /* Transparent background for the slider */
    }

    /* Style for the slider handle and track */
    .stSlider .css-1aumxhk {
        background-color: #b2a284 !important; /* Light color for the slider handle */
    }
    
    /* Multiselect and dropdown style */
    .stMultiSelect .st-multiselect .st-bn, .stSelectbox .st-bn {
        background-color: #b2a284 !important; /* Light color for dropdowns */
        color: #ffffff !important; /* White text */
    }

    /* Checkbox styling */
    .stCheckbox input {
        accent-color: #b2a284; /* Light color for checkboxes */
    }

    /* Expander headers */
    .streamlit-expanderHeader {
        background-color: #b2a284 !important; /* Light background for expanders */
        color: #343534 !important; /* Dark text */
    }

    /* General sidebar font and text styling */
    .sidebar-content {
        color: #ffffff !important; /* White text for sidebar content */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Date Range Filter without background color on the slider
min_year = int(min(df_market['year'].min(), df_agency['year'].min()))
max_year = int(max(df_market['year'].max(), df_agency['year'].max()))
selected_years = st.sidebar.slider(
    "Sélectionner l'intervalle de dates", 
    min_year, max_year, (min_year, max_year)
)

# Toggle for commune filter inside expander
with st.sidebar.expander("Sélectionner les Communes", expanded=False):  # Default to collapsed
    # Get list of all communes and communes where the agency has sold
    all_communes = sorted(df_market['commune'].unique())
    agency_communes = sorted(df_agency['commune'].unique())
    
    # Checkboxes to control commune selection
    select_all_communes = st.checkbox("Toutes les communes", value=True, key=generate_key('communes', 'all'))
    select_agency_communes = st.checkbox("Mes communes", value=False, key=generate_key('communes', 'agency'))
    
    # Logic to display relevant communes
    if select_agency_communes and not select_all_communes:
        selected_communes = st.multiselect("Communes Disponibles", agency_communes, default=agency_communes)
    elif select_all_communes:
        selected_communes = st.multiselect("Communes Disponibles", all_communes, default=all_communes)
    else:
        selected_communes = st.multiselect("Communes Disponibles", all_communes, default=[])

# Toggle for building type filter inside expander
with st.sidebar.expander("Sélectionner les types de bâtiments", expanded=False):  # Default to collapsed
    if 'type de bâtiment' in df_market.columns:
        building_types = sorted(df_market['type de bâtiment'].dropna().unique())
        selected_building_types = st.multiselect("Types de bâtiments Disponibles", building_types, default=building_types)
    else:
        st.error("Colonne 'type de bâtiment' non trouvée dans 'all_cities_belgium.csv'.")
        st.stop()

# Separate expanders for column names of "Données du Marché" and "Données de l'Agence"
with st.sidebar.expander("📋 Colonnes des Données", expanded=False):
    # Données du Marché
    st.write("### Données du Marché :")
    st.write([
        "province",
        "arrondissement",
        "commune",
        "type de bâtiment",
        "map",
        "date",
        "nombre de ventes",
        "prix médian"
    ])
    
    # Données de l'Agence
    st.write("### Données de l'Agence :")
    st.write([
        "agence",
        "date de transaction",
        "communes",
        "map"
    ])

# Apply Filters to both market and agency data based on selections
df_market_filtered = df_market[
    (df_market['year'] >= selected_years[0]) &
    (df_market['year'] <= selected_years[1]) &
    (df_market['commune'].isin(selected_communes)) &
    (df_market['type de bâtiment'].isin(selected_building_types))
]

df_agency_filtered = df_agency[
    (df_agency['year'] >= selected_years[0]) &
    (df_agency['year'] <= selected_years[1]) &
    (df_agency['commune'].isin(selected_communes))
]


# --------------------------------------
# Step 0: Market Share Calculations
# --------------------------------------

# Total Sales in Market
total_sales_market = df_market_filtered.groupby(['year', 'quarter', 'commune']).agg({'nombre de ventes': 'sum'}).reset_index()
total_sales_market.rename(columns={'nombre de ventes': 'nombre de ventes_market'}, inplace=True)

# Total Sales by Agency
total_sales_agency = df_agency_filtered.groupby(['year', 'quarter', 'commune']).agg({'nombre de ventes': 'sum'}).reset_index()
total_sales_agency.rename(columns={'nombre de ventes': 'nombre de ventes_agency'}, inplace=True)

# Merge to calculate Market Share
market_share = pd.merge(
    total_sales_market,
    total_sales_agency,
    on=['year', 'quarter', 'commune'],
    how='left'
).fillna({'nombre de ventes_agency': 0})

market_share['market_share (%)'] = (market_share['nombre de ventes_agency'] / market_share['nombre de ventes_market']) * 100
market_share['market_share (%)'] = market_share['market_share (%)'].replace([np.inf, -np.inf], 0).fillna(0)

market_share['date'] = pd.to_datetime(market_share['year'].astype(str) + '-Q' + market_share['quarter'].astype(str))
market_share.sort_values('date', inplace=True)





# --------------------------------------
# Step 1: Ventes de l'Agence par Commune et Trimestre
# --------------------------------------
st.header("1. Part de chaque Commune dans les ventes de l'agence par Trimestre")

# Filtrer les communes où l'agence a vendu des biens
sold_communes = df_agency_filtered['commune'].unique()
df_agency_sold_communes = df_agency_filtered[df_agency_filtered['commune'].isin(sold_communes)]

# Récupérer la première date de vente
earliest_agency_sale_date = df_agency_filtered['date'].min()
earliest_year = earliest_agency_sale_date.year
earliest_quarter = earliest_agency_sale_date.quarter

# Filtrer les données à partir du premier trimestre de vente
df_agency_filtered = df_agency_filtered[(df_agency_filtered['year'] > earliest_year) | 
                                        ((df_agency_filtered['year'] == earliest_year) & 
                                         (df_agency_filtered['quarter'] >= earliest_quarter))]

# Agréger les ventes totales de l'agence par commune et trimestre
total_agency_sales = df_agency_filtered.groupby(['commune', 'year', 'quarter']).agg(
    total_agency_sales=('nombre de ventes', 'sum')
).reset_index()

# Calculer les ventes totales pour chaque trimestre
total_sales_per_quarter = total_agency_sales.groupby(['year', 'quarter']).agg(
    total_sales_quarter=('total_agency_sales', 'sum')
).reset_index()

# Fusionner les ventes trimestrielles totales avec les ventes par commune
total_agency_sales = pd.merge(
    total_agency_sales,
    total_sales_per_quarter,
    on=['year', 'quarter'],
    how='left'
)

# Calculer le pourcentage de ventes par commune
total_agency_sales['sales_percentage'] = (total_agency_sales['total_agency_sales'] / total_agency_sales['total_sales_quarter']) * 100

# Créer des étiquettes de trimestre
total_agency_sales['quarter_label'] = total_agency_sales['year'].astype(str) + " T" + total_agency_sales['quarter'].astype(str)

# Table pivot pour afficher les pourcentages par commune et trimestre
sales_percentage_table = total_agency_sales.pivot_table(
    index='commune', 
    columns='quarter_label', 
    values='sales_percentage',
    fill_value=0
)

# Trier par commune pour une meilleure lisibilité
sales_percentage_table = sales_percentage_table.sort_index()

# Function to apply a green gradient
def apply_green_gradient(data):
    styles = pd.DataFrame('', index=data.index, columns=data.columns)

    # Normalize the data for coloring
    max_value = data.max().max()  # Find the max value to normalize
    for column in data.columns:
        for index in data.index:
            value = data.loc[index, column]
            if max_value != 0:
                green_shade = int((value / max_value) * 255)  # Color scale
                styles.at[index, column] = f'background-color: rgb({255 - green_shade}, 255, {255 - green_shade})'
            else:
                styles.at[index, column] = 'background-color: rgb(255, 255, 255)'

    return styles

# Apply the gradient on the sales percentage table
styled_sales_percentage_table = sales_percentage_table.style.apply(apply_green_gradient, axis=None).format("{:.2f}%")

# Display the table in Streamlit
st.dataframe(styled_sales_percentage_table, use_container_width=True)

# --------------------------------------
# Unified Analysis Section for Selected Communes
# --------------------------------------

# Define historically important communes (based on the average sales percentage)
historically_important_communes = sales_percentage_table.mean(axis=1).sort_values(ascending=False)
historically_important_communes = historically_important_communes[historically_important_communes > 10].index.tolist()

# Generate analysis for historically important communes
analysis_text = "### Aperçu Global des Communes Significatives :\n\n"

for commune in historically_important_communes:
    # Metrics for each commune
    last_quarter = sales_percentage_table.loc[commune].iloc[-1]
    second_last_quarter = sales_percentage_table.loc[commune].iloc[-2]
    average_sales = sales_percentage_table.loc[commune].mean()
    trend = last_quarter - second_last_quarter
    volatility = sales_percentage_table.loc[commune].std()

    # Compact and Fluid Insights Generation
    analysis_text += f"**{commune}** : "

    # Historique et Importance
    if average_sales > 15:
        analysis_text += f"Historiquement très importante ({average_sales:.2f}%). "
    elif average_sales > 10:
        analysis_text += f"Importance notable ({average_sales:.2f}%). "

    # Tendance des deux derniers trimestres
    if trend > 2:
        analysis_text += f"Augmentation notable de {trend:.2f}%, opportunité à exploiter. "
    elif 0 < trend <= 2:
        analysis_text += f"Légère augmentation ({trend:.2f}%), progression à surveiller. "
    elif -2 < trend < 0:
        analysis_text += f"Légère baisse ({abs(trend):.2f}%), variation normale. "
    elif trend <= -2:
        analysis_text += f"Baisse marquée de {abs(trend):.2f}%, réévaluation nécessaire. "

    # Volatilité
    if volatility > 10:
        analysis_text += "Volatilité élevée, conditions instables à surveiller. "
    elif volatility > 5:
        analysis_text += "Volatilité modérée. "
    
    # Stratégie et alignement
    if trend < 0 and average_sales > 10:
        analysis_text += "  "

    # Add line break for readability
    analysis_text += "\n"

# Ajouter les changements significatifs dans les autres communes (uniquement les grandes variations)
significant_changes = []

# Calcul des changements significatifs dans d'autres communes
for commune in sales_percentage_table.index:
    if commune not in historically_important_communes:
        last_quarter = sales_percentage_table.loc[commune].iloc[-1]
        second_last_quarter = sales_percentage_table.loc[commune].iloc[-2]
        change = last_quarter - second_last_quarter
        if abs(change) > 5:  # Seuil pour les changements importants
            significant_changes.append((commune, change))

if significant_changes:
    analysis_text += "### Changements Significatifs dans d'Autres Communes :\n\n"
    for commune, change in significant_changes:
        if change > 0:
            analysis_text += f"- **{commune}** : Augmentation de {change:.2f}%, potentiel à exploiter. "
        else:
            analysis_text += f"- **{commune}** : Diminution de {abs(change):.2f}%, réévaluation nécessaire. "

# Affichage dans Streamlit
st.markdown(analysis_text, unsafe_allow_html=True)


# --------------------------------------
# Step 2: Répartition en Pourcentage des Ventes par Commune au Dernier Trimestre
# --------------------------------------
st.header("2. Répartition en Pourcentage des Ventes par Commune au Dernier Trimestre")

# Convert 'date' column to datetime if necessary
df_agency_filtered['date'] = pd.to_datetime(df_agency_filtered['date'])

# Extract the latest year and quarter
latest_year = df_agency_filtered['year'].max()
latest_quarter = df_agency_filtered[df_agency_filtered['year'] == latest_year]['quarter'].max()

# Filter data for the last quarter
df_last_quarter = df_agency_filtered[
    (df_agency_filtered['year'] == latest_year) &
    (df_agency_filtered['quarter'] == latest_quarter)
]

# Check if data is available for the last quarter
if df_last_quarter.empty:
    st.warning("Aucune donnée disponible pour le dernier trimestre.")
else:
    # Aggregate total sales by commune for the last quarter
    sales_by_commune = df_last_quarter.groupby('commune').agg(
        total_sales=('nombre de ventes', 'sum')
    ).reset_index()

    # Calculate total sales
    total_sales = sales_by_commune['total_sales'].sum()

    # Calculate sales percentage by commune
    sales_by_commune['sales_percentage'] = (sales_by_commune['total_sales'] / total_sales) * 100

    # Create pie chart
    fig_pie = px.pie(
        sales_by_commune,
        values='sales_percentage',
        names='commune',
        title=f"Répartition en Pourcentage des Ventes par Commune - {latest_year} T{latest_quarter}",
        hover_data=['total_sales'],
        labels={'total_sales': 'Nombre de Ventes', 'sales_percentage': 'Pourcentage des Ventes'},
        color_discrete_sequence=px.colors.sequential.Greens
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')

    # Display the chart in Streamlit
    st.plotly_chart(fig_pie, use_container_width=True)

    # Identify communes with the highest contributions
    top_communes = sales_by_commune.sort_values('sales_percentage', ascending=False).head(3)
    top_communes_list = top_communes['commune'].tolist()




# --------------------------------------
# Step 3: Évolution de la Part de Marché par Commune
# --------------------------------------

st.header("3. Évolution de la Part de Marché par Commune")

# Question stratégique
st.markdown("""
**Comment nos ventes évoluent-elles dans chaque commune par rapport au marché au fil du temps ?**
""")

# Étape 1: Filtrer les données à partir du premier trimestre de vente de l'agence
first_sale_date = market_share.loc[market_share['nombre de ventes_agency'] > 0, 'date'].min()

if pd.isnull(first_sale_date):
    st.warning("Aucune vente effectuée par l'agence dans les données disponibles.")
    st.stop()
else:
    first_sale_quarter = f"{first_sale_date.year} T{first_sale_date.quarter}"
    st.write(f"**Premier trimestre avec une vente de l'agence :** {first_sale_quarter}")

# Filtrer les données
market_share_filtered = market_share[market_share['date'] >= first_sale_date].copy()

if market_share_filtered.empty:
    st.warning("Aucune donnée disponible après le premier trimestre de vente de l'agence.")
    st.stop()

# Étape 2: Calculer la part de marché totale pour chaque commune
market_share_filtered['Part_de_Marche_Totale (%)'] = (market_share_filtered['nombre de ventes_agency'] / market_share_filtered['nombre de ventes_market']) * 100

# Remplacer les NaN et les valeurs infinies
market_share_filtered['Part_de_Marche_Totale (%)'] = market_share_filtered['Part_de_Marche_Totale (%)'].replace([np.inf, -np.inf], 0).fillna(0)

# Obtenir les 5 communes avec la part de marché totale la plus élevée
top_communes_by_market_share = market_share_filtered.groupby('commune')['Part_de_Marche_Totale (%)'].mean().sort_values(ascending=False).head(5).index.tolist()

# Obtenir la liste des communes où l'agence a déjà vendu des biens
communes_with_sales = market_share_filtered.loc[market_share_filtered['nombre de ventes_agency'] > 0, 'commune'].unique().tolist()
communes_with_sales.sort()

# Créer une boîte de sélection pour que l'utilisateur puisse choisir les communes
selected_communes = st.multiselect(
    "Sélectionnez les communes à afficher",
    options=communes_with_sales,
    default=top_communes_by_market_share,
    key=generate_key('step14', 'communes_select')
)

if not selected_communes:
    st.warning("Veuillez sélectionner au moins une commune pour afficher les données.")
    st.stop()
else:
    st.write(f" {' '}")

# Étape 3: Préparer les données pour les graphiques
filtered_sales = market_share_filtered[market_share_filtered['commune'].isin(selected_communes)].copy()
filtered_sales['date_label'] = filtered_sales.apply(lambda row: f"{row['year']} T{row['quarter']}", axis=1)

# Vérifier les colonnes nécessaires
required_columns = ['nombre de ventes_agency', 'nombre de ventes_market']
if not all(col in filtered_sales.columns for col in required_columns):
    st.error(f"Colonnes manquantes dans filtered_sales : {required_columns}")
    st.stop()

# Étape 4: Calcul des Variations en Pourcentage
filtered_sales = filtered_sales.sort_values(['commune', 'date'])

filtered_sales['variation_agence'] = filtered_sales.groupby('commune')['nombre de ventes_agency'].pct_change() * 100
filtered_sales['variation_marché'] = filtered_sales.groupby('commune')['nombre de ventes_market'].pct_change() * 100

# Remplacer les valeurs négatives et NaN pour le scatter plot
filtered_sales['variation_agence_clean'] = filtered_sales['variation_agence'].apply(lambda x: x if x > 0 else 0).fillna(0)

# Classification des communes
trend_threshold = 5  # 5% de changement
outperforming_communes = []
underperforming_communes = []
matching_communes = []

for commune in selected_communes:
    commune_data = filtered_sales[filtered_sales['commune'] == commune].sort_values('date')
    if len(commune_data) >= 2:
        last_quarter_agence = commune_data.iloc[-1]['variation_agence']
        last_quarter_marche = commune_data.iloc[-1]['variation_marché']

        if not np.isnan(last_quarter_agence) and not np.isnan(last_quarter_marche):
            if last_quarter_agence > last_quarter_marche + trend_threshold:
                outperforming_communes.append(commune)
            elif last_quarter_agence < last_quarter_marche - trend_threshold:
                underperforming_communes.append(commune)
            else:
                matching_communes.append(commune)
        else:
            matching_communes.append(commune)
    else:
        matching_communes.append(commune)

# Étape 5: Visualisation - Graphique en Ligne avec Petits Multiples
st.markdown("#### Évolution des Ventes Agence vs Marché par Commune")

sales_melted = filtered_sales.melt(
    id_vars=['commune', 'date_label'],
    value_vars=['nombre de ventes_agency', 'nombre de ventes_market'],
    var_name='Type',
    value_name='Ventes'
)
sales_melted['Type'] = sales_melted['Type'].replace({
    'nombre de ventes_agency': 'Ventes Agence',
    'nombre de ventes_market': 'Ventes Marché'
})

fig_line = px.line(
    sales_melted,
    x='date_label',
    y='Ventes',
    color='Type',
    facet_col='commune',
    facet_col_wrap=2,
    markers=True,
    title='Évolution des Ventes Agence vs Marché par Commune',
    labels={'date_label': 'Trimestre', 'Ventes': 'Nombre de Ventes'},
    height=250 * (len(selected_communes) // 2 + len(selected_communes) % 2),  # Ajuster la hauteur en fonction du nombre de facettes
    color_discrete_map={
        'Ventes Agence': '#D4A437',  # Century 21 Gold
        'Ventes Marché': '#3B3B3B'   # Century 21 Dark Gray
    }
)

fig_line.update_layout(
    hovermode='x unified',
    legend_title='Type de Ventes',
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(
        family='Century Gothic',
        size=12,
        color='#3B3B3B'
    )
)

st.plotly_chart(fig_line, use_container_width=True, key=generate_key('step14', 'line_chart'))

# Étape 6: Analyse et Recommandations
st.markdown("""
### Analyse et Implications Stratégiques :
""")

if outperforming_communes:
    communes_list = ', '.join([f"**{c}**" for c in outperforming_communes])
    st.markdown(f"- **Performance Supérieure au Marché** : Les communes {communes_list} affichent une croissance des ventes supérieure à celle du marché, indiquant une efficacité accrue de nos stratégies dans ces zones.\n")

if underperforming_communes:
    communes_list = ', '.join([f"**{c}**" for c in underperforming_communes])
    st.markdown(f"- **Performance Inférieure au Marché** : Les communes {communes_list} montrent une croissance des ventes inférieure à celle du marché, suggérant des opportunités d'amélioration ou des défis spécifiques à ces zones.\n")

if matching_communes:
    communes_list = ', '.join([f"**{c}**" for c in matching_communes])
    st.markdown(f"- **Performance Alignée avec le Marché** : Les communes {communes_list} maintiennent une croissance des ventes similaire à celle du marché, ce qui indique une stabilité dans nos opérations.\n")



def add_recommendations(communes, recommendations_list):
    for commune in communes:
        recommendation = random.choice(recommendations_list).format(commune=commune)
        st.markdown(f"- {recommendation}")


# --------------------------------------
# Step 4.: Part de Marché dans le Temps par Commune (Graphique en Lignes)
# --------------------------------------

st.header("4. Part de Marché dans le Temps par Commune")

# Question stratégique pour guider cette section
st.markdown("Comment la performance de notre agence évolue-t-elle dans les principales communes au fil du temps ?")

# Déterminer la date où l'agence a réalisé sa première vente
first_sale_date = df_agency['date'].min()

# Filtrer les données à partir de la date de la première vente de l'agence
market_share_filtered = market_share[market_share['date'] >= first_sale_date].copy()

# Étape pour calculer la part de marché totale par commune (pour définir les communes par défaut)
total_market_share_per_commune = market_share_filtered.groupby('commune').agg(
    Ventes_Agence=('nombre de ventes_agency', 'sum'),
    Ventes_Marché=('nombre de ventes_market', 'sum')
).reset_index()

total_market_share_per_commune['Part_de_Marche_Totale (%)'] = (total_market_share_per_commune['Ventes_Agence'] / total_market_share_per_commune['Ventes_Marché']) * 100

# Trier les communes par la part de marché totale décroissante
top_communes_by_market_share = total_market_share_per_commune.sort_values('Part_de_Marche_Totale (%)', ascending=False).head(5)['commune'].tolist()

# Obtenir la liste des communes où l'agence a déjà vendu des biens
communes_with_sales = market_share_filtered.loc[market_share_filtered['nombre de ventes_agency'] > 0, 'commune'].unique().tolist()
communes_with_sales.sort()

# Créer une boîte de sélection pour que l'utilisateur puisse choisir les communes
selected_communes = st.multiselect(
    "Sélectionnez les communes à afficher",
    options=communes_with_sales,
    default=top_communes_by_market_share,  # Par défaut, les 5 communes avec la plus haute part de marché
    key=generate_key('step2bis', 'communes_select')
)

if not selected_communes:
    st.warning("Veuillez sélectionner au moins une commune pour afficher les données.")
    st.stop()

# Filtrer les données pour les communes sélectionnées
market_share_top_n = market_share_filtered[market_share_filtered['commune'].isin(selected_communes)].copy()

if market_share_top_n.empty:
    st.warning("Aucune donnée disponible pour les communes sélectionnées.")
    st.stop()

# Création du graphique en lignes pour la part de marché dans le temps
fig_market_share_time = px.line(
    market_share_top_n,
    x='date',
    y='market_share (%)',
    color='commune',
    markers=True,
    title='Évolution de la Part de Marché dans les Communes Sélectionnées (Depuis la Première Vente de l’Agence)',
    labels={'date': 'Date', 'market_share (%)': 'Part de Marché (%)', 'commune': 'Commune'},
    color_discrete_sequence=px.colors.qualitative.Dark24
)

# Ajouter des annotations pour les pics significatifs
for commune in selected_communes:
    commune_data = market_share_top_n[market_share_top_n['commune'] == commune]
    if not commune_data.empty:
        max_share = commune_data['market_share (%)'].max()
        max_date = commune_data[commune_data['market_share (%)'] == max_share]['date'].iloc[0]
        fig_market_share_time.add_annotation(
            x=max_date,
            y=max_share,
            text=f"Pic à {commune}",
            showarrow=True,
            arrowhead=1
        )

# Affichage du graphique dans Streamlit
fig_market_share_time.update_layout(
    hovermode='x unified',
    font=dict(
        family='Century Gothic',
        size=12,
        color='#3B3B3B'
    )
)

st.plotly_chart(fig_market_share_time, use_container_width=True, key=generate_key('step2bis', 'market_share_over_time_line'))

# --------------------------------------
# Analyse dynamique des tendances par commune
# --------------------------------------

# Initialiser les listes pour stocker les communes par tendance
increasing_communes = []
decreasing_communes = []
stable_communes = []

# Définir un seuil pour considérer une variation significative
trend_threshold = 0.05  # 5% de changement en part de marché

# Analyser chaque commune
for commune in selected_communes:
    commune_data = market_share_top_n[market_share_top_n['commune'] == commune].sort_values('date')
    if len(commune_data) >= 2:
        first_share = commune_data.iloc[0]['market_share (%)']
        last_share = commune_data.iloc[-1]['market_share (%)']
        change = last_share - first_share
        percent_change = (change / first_share) if first_share != 0 else 0

        if percent_change > trend_threshold:
            increasing_communes.append(commune)
        elif percent_change < -trend_threshold:
            decreasing_communes.append(commune)
        else:
            stable_communes.append(commune)
    else:
        stable_communes.append(commune)

# Générer le texte d'analyse dynamique
analysis_text = "### Analyse et Implications Stratégiques :\n"

# Communes avec croissance de part de marché
if increasing_communes:
    communes_list = ', '.join([f"**{c}**" for c in increasing_communes])
    analysis_text += f"- **Croissance de Part de Marché** : Les communes {communes_list} montrent une augmentation constante de la part de marché, indiquant un engagement local fort ou un positionnement compétitif de l'agence.\n"

# Communes avec baisse de part de marché
if decreasing_communes:
    communes_list = ', '.join([f"**{c}**" for c in decreasing_communes])
    analysis_text += f"- **Baisse de Part de Marché** : Les communes {communes_list} présentent une diminution de la part de marché, suggérant des défis potentiels tels qu'une concurrence accrue ou des besoins en ajustement stratégique.\n"

# Communes avec part de marché stable
if stable_communes:
    communes_list = ', '.join([f"**{c}**" for c in stable_communes])
    analysis_text += f"- **Stabilité de Part de Marché** : Les communes {communes_list} maintiennent une part de marché stable, offrant une base solide pour développer de nouvelles initiatives.\n"

# Afficher l'analyse
st.markdown(analysis_text)






# --------------------------------------
# Step 0: Market Share Calculations
# --------------------------------------

# Total Sales in Market
total_sales_market = df_market_filtered.groupby(['year', 'quarter', 'commune']).agg({'nombre de ventes': 'sum'}).reset_index()
total_sales_market.rename(columns={'nombre de ventes': 'nombre de ventes_market'}, inplace=True)

# Total Sales by Agency
total_sales_agency = df_agency_filtered.groupby(['year', 'quarter', 'commune']).agg({'nombre de ventes': 'sum'}).reset_index()
total_sales_agency.rename(columns={'nombre de ventes': 'nombre de ventes_agency'}, inplace=True)

# Merge to calculate Market Share
market_share = pd.merge(
    total_sales_market,
    total_sales_agency,
    on=['year', 'quarter', 'commune'],
    how='left'
).fillna({'nombre de ventes_agency': 0})

market_share['market_share (%)'] = (market_share['nombre de ventes_agency'] / market_share['nombre de ventes_market']) * 100
market_share['market_share (%)'] = market_share['market_share (%)'].replace([np.inf, -np.inf], 0).fillna(0)

market_share['date'] = pd.to_datetime(market_share['year'].astype(str) + '-Q' + market_share['quarter'].astype(str))
market_share.sort_values('date', inplace=True)


# --------------------------------------
# Step 5: Volume des Ventes vs Part de Marché par Commune
# --------------------------------------

st.header("5. Volume des Ventes vs Part de Marché par Commune")

# Question commerciale guidant la section
st.markdown("""
**Comment pouvons-nous aligner notre stratégie pour capturer les plus gros marchés où nous sommes encore sous-représentés ?**
""")

# Étape 1: Filtrer les données depuis la première vente jusqu'à la dernière période
# Identifier la première date de vente de l'agence
first_sale_date = market_share.loc[market_share['nombre de ventes_agency'] > 0, 'date'].min()

if pd.isnull(first_sale_date):
    st.warning("Aucune vente effectuée par l'agence dans les données disponibles.")
    st.stop()
else:
    first_sale_quarter = f"{first_sale_date.year} T{first_sale_date.quarter}"
    st.write(f"**Premier trimestre avec une vente de l'agence :** {first_sale_quarter}")

# Identifier la dernière date de vente (agence ou marché)
last_sale_date = market_share.loc[
    (market_share['nombre de ventes_agency'] > 0) |
    (market_share['nombre de ventes_market'] > 0),
    'date'
].max()

if pd.isnull(last_sale_date):
    st.warning("Aucune vente trouvée dans les données disponibles.")
    st.stop()
else:
    last_sale_quarter = f"{last_sale_date.year} T{last_sale_date.quarter}"
    st.write(f"**Dernier trimestre avec une vente :** {last_sale_quarter}")

# Filtrer les données entre la première et la dernière date de vente
market_share_filtered = market_share[
    (market_share['date'] >= first_sale_date) &
    (market_share['date'] <= last_sale_date)
].copy()

if market_share_filtered.empty:
    st.warning("Aucune donnée disponible entre la première et la dernière période de vente.")
    st.stop()

# Étape 2: Afficher toutes les communes où l'agence a réalisé des ventes
communes_with_sales = market_share_filtered.loc[market_share_filtered['nombre de ventes_agency'] > 0, 'commune'].unique()

filtered_sales = market_share_filtered[market_share_filtered['commune'].isin(communes_with_sales)].copy()

if filtered_sales.empty:
    st.warning("Aucune donnée disponible pour les communes sélectionnées.")
    st.stop()

# Étape 3: Calculer la part de marché pour chaque commune
sales_vs_market_share = filtered_sales.groupby('commune').agg({
    'nombre de ventes_market': 'sum',
    'nombre de ventes_agency': 'sum'
}).reset_index()

sales_vs_market_share['market_share (%)'] = (sales_vs_market_share['nombre de ventes_agency'] / sales_vs_market_share['nombre de ventes_market']) * 100

# Filtrer les communes avec des ventes positives pour éviter les divisions par zéro
sales_vs_market_share = sales_vs_market_share[
    (sales_vs_market_share['nombre de ventes_market'] > 0) &
    (sales_vs_market_share['nombre de ventes_agency'] > 0)
].copy()

# Trier les communes par ordre décroissant de part de marché
sales_vs_market_share = sales_vs_market_share.sort_values('market_share (%)', ascending=False).reset_index(drop=True)

# Calculer la moyenne de la part de marché
mean_market_share = sales_vs_market_share['market_share (%)'].mean()

# Créer une colonne pour catégoriser les communes en fonction de leur performance
def categorize_commune(row):
    if row['market_share (%)'] >= mean_market_share:
        return 'Haute Part de Marché'
    else:
        return 'Faible Part de Marché'

sales_vs_market_share['Performance'] = sales_vs_market_share.apply(categorize_commune, axis=1)

# Nettoyer les valeurs de 'market_share (%)' pour éviter les NaN et les valeurs négatives
sales_vs_market_share['market_share_clean'] = sales_vs_market_share['market_share (%)'].clip(lower=0).fillna(0)

# Graphique à bulles affichant toutes les communes, sans filtrer sur 2% pour le graphique
fig_bubble = px.scatter(
    sales_vs_market_share,
    x='nombre de ventes_market',
    y='nombre de ventes_agency',
    size='market_share_clean',
    color='Performance',
    hover_name='commune',
    title='Volume des Ventes vs Part de Marché par Commune',
    labels={
        'nombre de ventes_market': 'Ventes Totales du Marché',
        'nombre de ventes_agency': 'Ventes de l\'Agence',
        'market_share_clean': 'Part de Marché (%)'
    },
    size_max=60,
    template='simple_white',
    color_discrete_map={
        'Haute Part de Marché': '#D4A437',  # Century 21 Gold
        'Faible Part de Marché': '#C0392B'   # Red for underperformance
    }
)

fig_bubble.update_layout(
    xaxis_title='Ventes Totales du Marché',
    yaxis_title='Ventes de l\'Agence',
    legend_title='Performance',
    hovermode='closest',
    height=500,  # Réduire la hauteur pour une meilleure compacité
    margin=dict(l=50, r=50, t=50, b=50),
    font=dict(
        family='Century Gothic',
        size=12,
        color='#3B3B3B'
    )
)

st.plotly_chart(fig_bubble, use_container_width=True)

# Analyse des Communes Clés
st.subheader("Analyse des Communes Clés")

# Ne conserver que les communes avec une part de marché supérieure ou égale à 2%
top_communities_over_2_percent = sales_vs_market_share[sales_vs_market_share['market_share (%)'] >= 2].copy()

# Présenter les analyses sous forme de tableau sans recommandations
analysis_data = []

for index, row in top_communities_over_2_percent.iterrows():
    commune = row['commune']
    ventes_marche = int(row['nombre de ventes_market'])
    ventes_agence = int(row['nombre de ventes_agency'])
    part_de_marche = row['market_share (%)']

    analysis_data.append({
        'Commune': commune,
        'Ventes Marché': ventes_marche,
        'Ventes Agence': ventes_agence,
        'Part de Marché (%)': f"{part_de_marche:.2f}"
    })

# Convertir les données d'analyse en DataFrame
analysis_df = pd.DataFrame(analysis_data)

# Afficher la table d'analyse
st.table(analysis_df)

# Générer les recommandations basées sur l'analyse des données pour les communes avec un marché >= 2%
recommendations_high = top_communities_over_2_percent[top_communities_over_2_percent['Performance'] == 'Haute Part de Marché']
recommendations_low = top_communities_over_2_percent[top_communities_over_2_percent['Performance'] == 'Faible Part de Marché']

# Générer le texte des recommandations
recommendations_text = "En nous basant sur l'analyse ci-dessus, voici les recommandations pour les communes clés :\n\n"


# Ajouter une note sur la période couverte
recommendations_text += "\n_**Note** : Cette analyse couvre l'ensemble des périodes durant lesquelles nous avons été actifs. Elle offre une vue globale de nos performances par commune depuis le début de nos activités jusqu'à la dernière période de vente._"

# Afficher les recommandations


# --------------------------------------
# Step 6: Part de Marché par Commune au Fil du Temps
# --------------------------------------

st.header("6. Part de Marché par Commune au Fil du Temps")

# Question commerciale guidant la section
st.markdown("""
Comment la part de marché de chaque commune a-t-elle évolué au fil du temps, et quelles stratégies pouvons-nous adopter pour optimiser notre position ?
""")

# Étape 1: Définir la date de début de l'analyse à Q1 2022
analysis_start_date = pd.to_datetime('2022-01-01')

# Afficher la période d'analyse
st.write(f"**Période d'analyse :** À partir du {analysis_start_date.strftime('%Y-%m-%d')}")

# Filtrer les données de market_share à partir de la date de début de l'analyse
market_share_filtered = market_share[market_share['date'] >= analysis_start_date].copy()

if market_share_filtered.empty:
    st.warning("Aucune donnée disponible à partir de la date de début de l'analyse.")
    st.stop()

# Étape 2: Sélectionner les communes avec des ventes réalisées par l'agence dans la période d'analyse
communes_with_sales = market_share_filtered.loc[market_share_filtered['nombre de ventes_agency'] > 0, 'commune'].unique().tolist()
communes_with_sales.sort()

# Calculer les top 5 communes en termes de ventes de l'agence sur la période d'analyse
top_communes = market_share_filtered.groupby('commune')['nombre de ventes_agency'].sum().sort_values(ascending=False).head(5).index.tolist()

# Créer une boîte de sélection multiple avec les top 5 communes sélectionnées par défaut
selected_communes = st.multiselect(
    "Sélectionnez les communes à afficher",
    options=communes_with_sales,
    default=top_communes,
    key=generate_key('step6', 'communes_select')
)

if not selected_communes:
    st.warning("Veuillez sélectionner au moins une commune pour afficher les données.")
    st.stop()


# Filtrer les données pour les communes sélectionnées
market_share_top_n = market_share_filtered[market_share_filtered['commune'].isin(selected_communes)].copy()

if market_share_top_n.empty:
    st.warning("Aucune donnée disponible pour les communes sélectionnées.")
    st.stop()

# Étape 3: Calculer la part de marché moyenne par date pour les communes sélectionnées
avg_market_share_by_date = market_share_top_n.groupby('date')['market_share (%)'].mean().reset_index()

# Étape 4: Créer le graphique combiné : Barres pour les communes et ligne pour la moyenne des parts de marché
fig_combined = go.Figure()

# Ajouter les barres pour chaque commune
for commune in selected_communes:
    commune_data = market_share_top_n[market_share_top_n['commune'] == commune]
    fig_combined.add_trace(
        go.Bar(
            x=commune_data['date'],
            y=commune_data['market_share (%)'],
            name=commune,
            hoverinfo='x+y+name'
        )
    )

# Ajouter la ligne pour la moyenne des parts de marché pour les communes sélectionnées
fig_combined.add_trace(
    go.Scatter(
        x=avg_market_share_by_date['date'],
        y=avg_market_share_by_date['market_share (%)'],
        mode='lines+markers',
        name='Moyenne des Parts de Marché (%)',
        line=dict(color='#D4A437', width=3),
        marker=dict(size=6)
    )
)

# Mettre à jour la mise en page du graphique
fig_combined.update_layout(
    barmode='group',
    title='Part de Marché par Commune au Fil du Temps (À partir de Q1 2022)',
    xaxis=dict(title='Date', tickformat='%Y-Q%q'),
    yaxis=dict(title='Part de Marché (%)', tickformat=',.2f', range=[0, 100]),
    hovermode='x unified',
    height=600,
    legend_title_text='Communes',
    margin=dict(l=40, r=40, t=60, b=40),
    font=dict(
        family='Century Gothic',
        size=12,
        color='#3B3B3B'
    )
)

# Afficher le graphique
st.plotly_chart(fig_combined, use_container_width=True, key=generate_key('step6', 'market_share_grouped'))

# Étape 5: Création de la table d'analyse
analysis_df = market_share_top_n.groupby('commune').agg(
    Ventes_Agence=('nombre de ventes_agency', 'sum'),
    Ventes_Marché=('nombre de ventes_market', 'sum'),
).reset_index()

# Calculer la part de marché totale en pourcentage
analysis_df['Part_de_Marche_Totale (%)'] = (analysis_df['Ventes_Agence'] / analysis_df['Ventes_Marché']) * 100

# Remplacer les valeurs infinies et NaN
analysis_df['Part_de_Marche_Totale (%)'] = analysis_df['Part_de_Marche_Totale (%)'].replace([np.inf, -np.inf], 0).fillna(0)

# Arrondir les valeurs
analysis_df['Part_de_Marche_Totale (%)'] = analysis_df['Part_de_Marche_Totale (%)'].round(2)

# Renommer 'commune' en 'Commune' pour l'affichage
analysis_df.rename(columns={'commune': 'Commune'}, inplace=True)

# Sélectionner les colonnes à afficher
analysis_display = analysis_df[['Commune', 'Ventes_Agence', 'Ventes_Marché', 'Part_de_Marche_Totale (%)']]

# Trier les communes par Part_de_Marche_Totale (%) décroissante
analysis_display = analysis_display.sort_values('Part_de_Marche_Totale (%)', ascending=False).reset_index(drop=True)

# Étape 6: Afficher la table d'analyse
st.subheader("Analyse des Communes Clés")
st.table(analysis_display)


# Calculer la moyenne globale de la part de marché totale
global_avg_market_share = analysis_display['Part_de_Marche_Totale (%)'].mean()

# Classifier les communes en fonction de leur part de marché totale par rapport à la moyenne
recommendations_high = analysis_display[analysis_display['Part_de_Marche_Totale (%)'] >= global_avg_market_share]['Commune'].tolist()
recommendations_low = analysis_display[analysis_display['Part_de_Marche_Totale (%)'] < global_avg_market_share]['Commune'].tolist()



# Ajouter une note sur la période couverte
st.markdown(f"\n_**Note** : Cette analyse couvre tous les trimestres à partir de Q1 2022 ({analysis_start_date.strftime('%Y-%m-%d')}) jusqu'à la date la plus récente, offrant une vue complète de nos performances par commune sur cette période._")






# --------------------------------------
# Step 7: Carte de Chaleur du Volume de Ventes par Commune avec Détails au Survol (sans cercles visibles)
# --------------------------------------
st.header("7. Carte de Chaleur du Volume de Ventes par Commune avec Détails au Survol")

# Vérification des colonnes nécessaires
required_columns = {'latitude', 'longitude', 'nombre de ventes', 'commune'}
if not df_agency_filtered.empty and required_columns.issubset(df_agency_filtered.columns):
    # Préparation des données pour la heatmap
    # Agréger les ventes par commune pour obtenir le total des ventes par commune
    sales_by_commune = df_agency_filtered.groupby('commune').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'nombre de ventes': 'sum'
    }).reset_index()
    
    # Vérifier si les données agrégées ne sont pas vides
    if not sales_by_commune.empty:
        # Créer une carte Folium centrée autour de la latitude et longitude moyennes
        map_center = [sales_by_commune['latitude'].mean(), sales_by_commune['longitude'].mean()]
        sales_heatmap = folium.Map(location=map_center, zoom_start=10, tiles='CartoDB positron', control_scale=True)
        
        # Préparer les données pour la HeatMap
        heat_data = sales_by_commune[['latitude', 'longitude', 'nombre de ventes']].values.tolist()
        
        # Ajouter la couche heatmap pour les données de vente
        HeatMap(
            data=heat_data,
            radius=15,  # Rayon ajusté pour réduire la taille
            blur=10,    # Flou ajusté pour une répartition plus compacte
            max_zoom=12
        ).add_to(sales_heatmap)

        # Ajouter des tooltips au survol des communes sans cercles
        for _, row in sales_by_commune.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                icon=folium.DivIcon(html=""),  # Pas d'icône visible
                tooltip=f"<b>Commune:</b> {row['commune']}<br><b>Ventes:</b> {row['nombre de ventes']}"
            ).add_to(sales_heatmap)
        
        # Rendre la carte Folium dans Streamlit de manière compacte
        folium_static(sales_heatmap, width=700, height=500)
        
        
    else:
        st.warning("Aucune donnée de vente agrégée disponible pour tracer la carte de chaleur.")
else:
    st.error("Données insuffisantes pour générer la carte de chaleur du volume de ventes. Veuillez vérifier que les colonnes 'latitude', 'longitude', 'nombre de ventes' et 'commune' sont présentes et contiennent des données valides.")

# --------------------------------------
# Step 8: Communes les Plus Performantes par Part de Marché (Bar Chart)
# --------------------------------------
st.header("8. Communes les Plus Performantes par Part de Marché")

# Vérifier si 'market_share_filtered' est défini et contient les colonnes nécessaires
if 'market_share_filtered' not in locals():
    st.error("Le DataFrame 'market_share_filtered' n'est pas défini. Veuillez vous assurer que les étapes précédentes sont exécutées correctement.")
    st.stop()

required_columns = {'commune', 'market_share (%)'}
if not required_columns.issubset(market_share_filtered.columns):
    st.error(f"Le DataFrame 'market_share_filtered' doit contenir les colonnes suivantes : {required_columns}")
    st.stop()

# Filtrer les communes où l'agence a vendu au moins un bien
communes_with_sales = market_share_filtered.groupby('commune').agg({
    'market_share (%)': 'sum'
}).reset_index()
communes_with_sales = communes_with_sales[communes_with_sales['market_share (%)'] > 0]

# Agréger les données pour calculer la part de marché moyenne par commune
avg_market_share_commune_step8 = market_share_filtered[market_share_filtered['commune'].isin(communes_with_sales['commune'])].groupby('commune').agg({
    'market_share (%)': 'mean'
}).reset_index()

# Limiter aux 15 communes les plus performantes par part de marché moyenne
top_communes_step8 = avg_market_share_commune_step8.sort_values('market_share (%)', ascending=False).head(15)

# Créer le graphique à barres pour les communes les plus performantes
fig_top_communes_step8 = px.bar(
    top_communes_step8,
    x='commune',
    y='market_share (%)',
    title=f'Top 15 Communes les Plus Performantes par Part de Marché pour la période T1 2022 - Présent', ### PAs oublier de changer la maniere d'afficher la date
    labels={'market_share (%)': 'Part de Marché (%)', 'commune': 'Commune'},
    text='market_share (%)',
    color='market_share (%)',
    color_continuous_scale='Greens'
)

# Mettre à jour la mise en page du graphique
fig_top_communes_step8.update_layout(
    xaxis_title='Commune',
    yaxis_title='Part de Marché (%)',
    hovermode='x unified',
    coloraxis_colorbar=dict(title="Part de Marché (%)"),
    height=600,
    margin=dict(l=50, r=50, t=50, b=50),
    font=dict(
        family='Century Gothic',
        size=12,
        color='#3B3B3B'
    )
)

# Afficher les valeurs de part de marché sur les barres
fig_top_communes_step8.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

# Afficher le graphique dans Streamlit
st.plotly_chart(fig_top_communes_step8, use_container_width=True)


# --------------------------------------
# Step 9: Distribution des Ventes par Type de Propriété
# --------------------------------------
st.header("9. Distribution des Ventes par Type de Propriété pour le marché total")

# Vérifiez si la colonne 'type de bâtiment' est présente dans les données du marché
if 'type de bâtiment' in df_market.columns:
    # Agréger les ventes par type de bâtiment dans les données du marché
    property_type_market = df_market.groupby('type de bâtiment').agg({'nombre de ventes': 'sum'}).reset_index()

    # Créer un graphique en camembert pour la répartition des ventes par type de propriété
    fig_pie_property_market = px.pie(
        property_type_market,
        names='type de bâtiment',
        values='nombre de ventes',
        title="Répartition des Ventes sur le Marché par Type de Propriété",
        hole=0.3  # Pour créer un effet de "donut"
    )
    st.plotly_chart(fig_pie_property_market, use_container_width=True)

    # Analyse et recommandations
    st.markdown("""
    Répartition des ventes sur le marché en fonction des différents types de propriétés.
    """)
else:
    # Affichez un message si la colonne 'type de bâtiment' n'est pas disponible
    st.warning("La colonne 'type de bâtiment' n'est pas disponible dans les données du marché.")




# --------------------------------------
# Step 10: Benchmark des Concurrents par Commune
# --------------------------------------
st.header("10. Benchmark des Concurrents par Commune")

# Convertir les colonnes 'date' en datetime si nécessaire
df_market_filtered['date'] = pd.to_datetime(df_market_filtered['date'])
df_agency_filtered['date'] = pd.to_datetime(df_agency_filtered['date'])

# Obtenir la première date de vente de l'agence
earliest_agency_sale_date = df_agency_filtered['date'].min()

# Filtrer les données du marché à partir de la première date de vente de l'agence
df_market_filtered = df_market_filtered[df_market_filtered['date'] >= earliest_agency_sale_date]

# Calculer les ventes totales du marché par commune
total_sales_market = df_market_filtered.groupby('commune').agg({'nombre de ventes': 'sum'}).reset_index()
total_sales_market.rename(columns={'nombre de ventes': 'ventes_marché'}, inplace=True)

# Calculer les ventes totales de l'agence par commune
total_sales_agency = df_agency_filtered.groupby('commune').agg({'nombre de ventes': 'sum'}).reset_index()
total_sales_agency.rename(columns={'nombre de ventes': 'ventes_agence'}, inplace=True)

# Fusionner les données du marché et de l'agence
competitor_sales = pd.merge(
    total_sales_market,
    total_sales_agency,
    on='commune',
    how='left'
).fillna({'ventes_agence': 0})

# Calculer les ventes des concurrents
competitor_sales['ventes_concurrents'] = competitor_sales['ventes_marché'] - competitor_sales['ventes_agence']
competitor_sales['ventes_concurrents'] = competitor_sales['ventes_concurrents'].apply(lambda x: x if x >= 0 else 0)

# Checkbox pour filtrer "Mes Communes"
mes_communes = st.checkbox("Afficher uniquement les communes où l'agence a vendu", value=True)

# Filtrer pour les communes où l'agence a vendu des biens ou les top 20 communes du marché
if mes_communes:
    communes_with_sales = total_sales_agency[total_sales_agency['ventes_agence'] > 0]['commune'].unique()
    competitor_sales_filtered = competitor_sales[competitor_sales['commune'].isin(communes_with_sales)]
else:
    competitor_sales_filtered = competitor_sales.sort_values('ventes_marché', ascending=False).head(20)

# Préparer les données pour le graphique
competitor_sales_melted = competitor_sales_filtered.melt(
    id_vars='commune',
    value_vars=['ventes_agence', 'ventes_concurrents'],
    var_name='Type de Ventes',
    value_name='Nombre de Ventes'
)

# Renommer pour plus de clarté
competitor_sales_melted['Type de Ventes'] = competitor_sales_melted['Type de Ventes'].map({
    'ventes_agence': 'Ventes de l\'Agence',
    'ventes_concurrents': 'Ventes des Concurrents'
})

if not competitor_sales_melted.empty:
    # Créer le graphique à barres empilées
    fig_competitor = px.bar(
        competitor_sales_melted,
        x='commune',
        y='Nombre de Ventes',
        color='Type de Ventes',
        title='Ventes de l\'Agence vs Ventes des Concurrents par Commune',
        labels={'Nombre de Ventes': 'Nombre de Ventes', 'commune': 'Commune'},
        barmode='stack',
        color_discrete_map={
            'Ventes de l\'Agence': '#D4A437',
            'Ventes des Concurrents': '#3B3B3B'
        }
    )
    fig_competitor.update_layout(
        xaxis={'categoryorder': 'total descending'},
        hovermode='x unified',
        height=600,
        font=dict(
            family='Century Gothic',
            size=12,
            color='#3B3B3B'
        )
    )

    # Afficher le graphique
    st.plotly_chart(fig_competitor, use_container_width=True, key=generate_key('step12', 'competitor_benchmark'))

    # Observations et recommandations
    st.markdown("### Observations et Recommandations")

    # Analyse interne pour déterminer les communes clés
    top_competitor_communes = competitor_sales_filtered.sort_values('ventes_concurrents', ascending=False)['commune'].tolist()
    underperforming_communes = competitor_sales_filtered[competitor_sales_filtered['ventes_agence'] < competitor_sales_filtered['ventes_concurrents']]['commune'].tolist()
    top_agency_communes = competitor_sales_filtered.sort_values('ventes_agence', ascending=False)['commune'].tolist()

    # Observations
    if top_competitor_communes:
        st.markdown(f"**Communes avec forte concurrence :** {', '.join(top_competitor_communes[:3])}")
    if underperforming_communes:
        st.markdown(f"**Communes où l'agence sous-performe :** {', '.join(underperforming_communes[:3])}")
    if top_agency_communes:
        st.markdown(f"**Communes où l'agence performe bien :** {', '.join(top_agency_communes[:3])}")



# Continuing from where we left off

# --------------------------------------
# Step 11: Tendances des Ventes au Fil du Temps (Trimestriel) - Analyse Avancée
# --------------------------------------
st.header("11. Tendances des Ventes au Fil du Temps (Trimestriel)")

# Calculs nécessaires

# Obtenir la liste des communes où l'agence a déjà vendu des biens
communes_with_sales = df_agency_filtered['commune'].unique().tolist()

# Filtrer les données de marché pour ne garder que les communes où l'agence a déjà vendu
df_market_filtered_communes = df_market_filtered[df_market_filtered['commune'].isin(communes_with_sales)].copy()

# Total des ventes de l'agence par trimestre
total_sales_quarter_agency = df_agency_filtered.groupby(['year', 'quarter']).agg(
    total_sales_agency=('nombre de ventes', 'sum')
).reset_index()

# Total des ventes du marché par trimestre (en filtrant les communes où l'agence a vendu)
total_sales_quarter_market = df_market_filtered_communes.groupby(['year', 'quarter']).agg(
    total_sales_market=('nombre de ventes', 'sum')
).reset_index()

# Fusion des données d'agence et de marché
sales_quarter = pd.merge(
    total_sales_quarter_agency,
    total_sales_quarter_market,
    on=['year', 'quarter'],
    how='outer'
).fillna(0)

# Ajout d'une colonne pour la date au format datetime (pour faciliter le tri et l'affichage)
sales_quarter['date'] = pd.to_datetime(sales_quarter['year'].astype(str) + '-Q' + sales_quarter['quarter'].astype(str))
sales_quarter['date'] = sales_quarter['date'] + QuarterEnd(0)

# Calcul de la part de marché de l'agence par trimestre uniquement dans les communes où l'agence a vendu
sales_quarter['market_share (%)'] = (sales_quarter['total_sales_agency'] / sales_quarter['total_sales_market']) * 100
sales_quarter['market_share (%)'] = sales_quarter['market_share (%)'].replace([float('inf'), -float('inf')], 0).fillna(0)

# Tri des données à partir du trimestre le plus ancien
sales_quarter = sales_quarter.sort_values('date')

# Calcul des différences trimestrielles pour les ventes et la part de marché
sales_quarter['sales_change (%)'] = sales_quarter['total_sales_agency'].pct_change() * 100
sales_quarter['market_share_change (%)'] = sales_quarter['market_share (%)'].pct_change() * 100

# Calcul des ventes globales dans les communes où l'agence n'a pas encore vendu
df_market_no_sales = df_market_filtered[~df_market_filtered['commune'].isin(communes_with_sales)]
total_sales_quarter_market_no_sales = df_market_no_sales.groupby(['year', 'quarter']).agg(
    total_sales_market_no_sales=('nombre de ventes', 'sum')
).reset_index()

# Fusionner avec les données principales
sales_quarter = pd.merge(
    sales_quarter,
    total_sales_quarter_market_no_sales,
    on=['year', 'quarter'],
    how='left'
).fillna(0)

# Analyse des ventes globales où l'agence n'est pas encore présente
sales_quarter['market_opportunity_growth'] = sales_quarter['total_sales_market_no_sales'].pct_change() * 100

# Création du graphique en utilisant Plotly
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Ligne pour les ventes totales de l'agence
fig.add_trace(go.Scatter(
    x=sales_quarter['date'],
    y=sales_quarter['total_sales_agency'],
    mode='lines+markers',
    name='Ventes Totales Agence',
    line=dict(color='#D4A437'),  # Century 21 Gold
    marker=dict(size=8),
    text=[f"{int(sales)} ventes" for sales in sales_quarter['total_sales_agency']],
    textposition='top center',
    hovertemplate='Trimestre: %{x|%Y-Q%q}<br>Ventes Totales: %{y}'
), secondary_y=False)

# Ligne pour la part de marché de l'agence (uniquement dans les communes où l'agence a vendu)
fig.add_trace(go.Scatter(
    x=sales_quarter['date'],
    y=sales_quarter['market_share (%)'],
    mode='lines+markers',
    name='Part de Marché (%)',
    line=dict(color='#3B3B3B'),  # Century 21 Dark Gray
    marker=dict(size=8),
    text=[f"{share:.2f}%" for share in sales_quarter['market_share (%)']],
    textposition='top center',
    hovertemplate='Trimestre: %{x|%Y-Q%q}<br>Part de Marché: %{y:.2f}%'
), secondary_y=True)

# Mise à jour du layout du graphique
fig.update_layout(
    title='Performance des Ventes et Part de Marché au Fil du Temps (Trimestriel)',
    xaxis=dict(title='Date', tickformat='%Y-Q%q'),
    yaxis=dict(
        title='Ventes Totales Agence',
        showgrid=False,
    ),
    yaxis2=dict(
        title='Part de Marché (%)',
        overlaying='y',
        side='right',
        showgrid=False,
    ),
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5
    ),
    height=600,
    font=dict(
        family='Century Gothic',
        size=12,
        color='#3B3B3B'
    )
)

# Annotations pour les meilleurs et pires trimestres en termes de part de marché
if not sales_quarter.empty:
    best_quarter_market_share = sales_quarter.loc[sales_quarter['market_share (%)'].idxmax()]
    worst_quarter_market_share = sales_quarter.loc[sales_quarter['market_share (%)'].idxmin()]
    best_quarter_sales = sales_quarter.loc[sales_quarter['total_sales_agency'].idxmax()]
    worst_quarter_sales = sales_quarter.loc[sales_quarter['total_sales_agency'].idxmin()]

    fig.add_annotation(
        x=best_quarter_market_share['date'],
        y=best_quarter_market_share['market_share (%)'],
        text=f"🔝 Meilleur Trimestre Part de Marché: {best_quarter_market_share['market_share (%)']:.2f}%",
        showarrow=True,
        arrowhead=2,
        yshift=10,
        arrowcolor="#3B3B3B",
        font=dict(size=10, color="#3B3B3B")
    )

    fig.add_annotation(
        x=worst_quarter_market_share['date'],
        y=worst_quarter_market_share['market_share (%)'],
        text=f"⚠️ Pire Trimestre Part de Marché: {worst_quarter_market_share['market_share (%)']:.2f}%",
        showarrow=True,
        arrowhead=2,
        yshift=-10,
        arrowcolor="#C0392B",
        font=dict(size=10, color="#C0392B")
    )

    fig.add_annotation(
        x=best_quarter_sales['date'],
        y=best_quarter_sales['total_sales_agency'],
        text=f"🔝 Meilleur Trimestre Ventes: {int(best_quarter_sales['total_sales_agency'])} ventes",
        showarrow=True,
        arrowhead=2,
        yshift=10,
        arrowcolor="#D4A437",
        font=dict(size=10, color="#D4A437")
    )

    fig.add_annotation(
        x=worst_quarter_sales['date'],
        y=worst_quarter_sales['total_sales_agency'],
        text=f"⚠️ Pire Trimestre Ventes: {int(worst_quarter_sales['total_sales_agency'])} ventes",
        showarrow=True,
        arrowhead=2,
        yshift=-10,
        arrowcolor="#C0392B",
        font=dict(size=10, color="#C0392B")
    )

# Affichage du graphique dans Streamlit
st.plotly_chart(fig, use_container_width=True)

# Analyse approfondie et recommandations pour le gérant
if not sales_quarter.empty:
    last_sales_change = sales_quarter['sales_change (%)'].iloc[-1]
    last_market_share_change = sales_quarter['market_share_change (%)'].iloc[-1]
    market_opportunity_growth = sales_quarter['market_opportunity_growth'].iloc[-1]

    # Analyse des performances avec insight détaillé
    st.markdown(f"""
    ### Analyse Approfondie :
    - **Tendances des Ventes** : Le dernier trimestre montre une variation de **{last_sales_change:.2f}%** pour les ventes totales de l'agence. Si cette tendance est positive, elle montre un positionnement solide sur le marché. Si elle est négative, des ajustements stratégiques peuvent être nécessaires.
    - **Part de Marché** : La part de marché de l'agence a varié de **{last_market_share_change:.2f}%**. Une baisse pourrait indiquer une montée de la concurrence ou un recentrage stratégique, tandis qu'une hausse signale une efficacité accrue des actions locales.
    - **Opportunités Inexploitées** : Les ventes dans les communes où l'agence n'est pas encore présente ont crû de **{market_opportunity_growth:.2f}%**. Cela montre un potentiel à exploiter dans ces zones, où une présence pourrait générer de nouvelles opportunités de marché.
    """)


# --------------------------------------
# Step 12: Prévisions des Ventes Futures avec Machine Learning
# --------------------------------------
st.header("12. Prévisions des Ventes Futures avec Machine Learning")

# Question stratégique guidant la section
st.markdown("""
**Comment pouvons-nous utiliser les données historiques pour prévoir les ventes futures et mieux planifier nos stratégies commerciales ?**
""")

# Vérifier si suffisamment de données sont disponibles
if len(sales_quarter) < 4:
    st.warning("Pas assez de données historiques pour construire un modèle de prédiction fiable.")
else:
    # Préparation des données pour le modèle
    sales_quarter_ml = sales_quarter.copy()
    sales_quarter_ml['quarter_number'] = sales_quarter_ml['year'] * 4 + sales_quarter_ml['quarter']  # Convertir les années et trimestres en un numéro de trimestre unique
    sales_quarter_ml = sales_quarter_ml.sort_values('quarter_number')

    # Caractéristiques (features) et cible (target)
    X = sales_quarter_ml[['quarter_number', 'total_sales_market']]
    y = sales_quarter_ml['total_sales_agency']

    # Diviser les données en ensembles d'entraînement et de test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Créer et entraîner le modèle de régression linéaire
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédire sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Afficher les performances du modèle
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.markdown(f"**Performance du Modèle :**")
    st.write(f"- Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"- Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"- R² Score: {r2:.2f}")

    # Prévision pour les prochains trimestres
    last_quarter_number = X['quarter_number'].max()
    future_quarters = pd.DataFrame({
        'quarter_number': [last_quarter_number + i for i in range(1, 5)],
        'total_sales_market': [X['total_sales_market'].mean()] * 4  # Supposons que le marché reste stable
    })

    future_sales_pred = model.predict(future_quarters)

    # Ajouter les prévisions au dataframe pour affichage
    future_quarters['total_sales_agency_pred'] = future_sales_pred
    future_quarters['year'] = (future_quarters['quarter_number'] - 1) // 4
    future_quarters['quarter'] = ((future_quarters['quarter_number'] - 1) % 4) + 1
    future_quarters['date'] = pd.to_datetime(future_quarters['year'].astype(str) + '-Q' + future_quarters['quarter'].astype(str)) + QuarterEnd(0)

    # Afficher les prévisions dans un graphique
    fig_pred = go.Figure()

    # Données historiques
    fig_pred.add_trace(go.Scatter(
        x=sales_quarter['date'],
        y=sales_quarter['total_sales_agency'],
        mode='lines+markers',
        name='Ventes Historiques Agence',
        line=dict(color='#D4A437'),
        marker=dict(size=8),
        hovertemplate='Date: %{x|%Y-Q%q}<br>Ventes: %{y}'
    ))

    # Prévisions futures
    fig_pred.add_trace(go.Scatter(
        x=future_quarters['date'],
        y=future_quarters['total_sales_agency_pred'],
        mode='lines+markers',
        name='Prévisions Ventes Futures',
        line=dict(color='#3B3B3B', dash='dash'),
        marker=dict(size=8),
        hovertemplate='Date: %{x|%Y-Q%q}<br>Prévision Ventes: %{y:.2f}'
    ))

    fig_pred.update_layout(
        title='Prévisions des Ventes Futures de l\'Agence',
        xaxis_title='Date',
        yaxis_title='Nombre de Ventes',
        hovermode='x unified',
        height=600,
        font=dict(
            family='Century Gothic',
            size=12,
            color='#3B3B3B'
        )
    )

    st.plotly_chart(fig_pred, use_container_width=True)

    # Recommandations
    st.markdown("""
    **Analyse :** Les prévisions indiquent une tendance future basée sur les données historiques. Le modèle de régression linéaire utilise les ventes passées et le volume du marché pour estimer les ventes futures.

    **Recommandations :**
    - **Planification Anticipée :** Utilisez ces prévisions pour planifier les ressources nécessaires, y compris le personnel, le stock et les budgets marketing.
    - **Surveillance Continue :** Mettez régulièrement à jour le modèle avec de nouvelles données pour affiner les prévisions.
    - **Stratégies Proactives :** Si les prévisions indiquent une baisse, envisagez des actions pour stimuler les ventes, comme des promotions ou des campagnes publicitaires.
    """)

    # Transparence et limitations du modèle
    st.markdown("""
    **Note :** Les prévisions sont basées sur des tendances historiques et des hypothèses sur la stabilité du marché. Les résultats réels peuvent varier en fonction de nombreux facteurs, y compris les conditions économiques, la concurrence et les événements imprévus.

    **Action Demandée :** Considérez ces prévisions comme un outil pour soutenir la prise de décision stratégique, et non comme une certitude.
    """)

# --------------------------------------
# Footer or Additional Content
# --------------------------------------
st.markdown(
    """
    <hr style="border: 1px solid #D4A437;" />
    <p style="text-align: center; color: #777777;">
        © 2024 Century 21 Dashboard | Créé avec Streamlit
    </p>
    """,
    unsafe_allow_html=True
)


