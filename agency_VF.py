# agency_1.py
import streamlit as st

# Streamlit configuration
st.set_page_config(
    layout="wide",
    page_title="🏠 Tableau de Bord d'Analyse de la Part de Marché Immobilière",
    page_icon=":house:"
)


# --------------------------------------
# Step 1: Setup and Imports
# --------------------------------------
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import warnings
from io import StringIO



# Title of the application
st.title("🏠 Tableau de Bord d'Analyse de la Part de Marché Immobilière")


# Read data from Streamlit secrets
data = st.secrets["data"]["my_agency_data"]
df_agency = pd.read_csv(StringIO(data))

# Now you can use df_agency as a regular pandas DataFrame
st.write(df_agency)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Access market data from Streamlit secrets
market_data = st.secrets["data"]["all_cities_belgium"]
df_market = pd.read_csv(StringIO(market_data))

# Now you can use df_market as a regular DataFrame
st.write(df_market)

# --------------------------------------
# Step 2: Data Loading with Delimiter Detection
# --------------------------------------
@st.cache_data
def load_data():
    try:
        # Load market data
        with open('all_cities_belgium.csv', 'r', encoding='utf-8-sig') as f:
            first_line = f.readline()
            if ',' in first_line and '\t' not in first_line:
                sep = ','
            elif '\t' in first_line:
                sep = '\t'
            else:
                sep = ','
        df_market = pd.read_csv('all_cities_belgium.csv', sep=sep, encoding='utf-8-sig')
    except FileNotFoundError:
        st.error("Fichier 'all_cities_belgium.csv' non trouvé.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier 'all_cities_belgium.csv' : {e}")
        st.stop()

    try:
        # Load agency data
        with open('my_agency_data.csv', 'r', encoding='utf-8-sig') as f:
            first_line = f.readline()
            if ',' in first_line and '\t' not in first_line:
                sep_agency = ','
            elif '\t' in first_line:
                sep_agency = '\t'
            else:
                sep_agency = ','
        df_agency = pd.read_csv('my_agency_data.csv', sep=sep_agency, encoding='utf-8-sig')
    except FileNotFoundError:
        st.error("Fichier 'my_agency_data.csv' non trouvé.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier 'my_agency_data.csv' : {e}")
        st.stop()

    return df_market, df_agency

df_market, df_agency = load_data()

# --------------------------------------
# Step 3: Data Cleaning and Preprocessing
# --------------------------------------
def preprocess_data(df_market, df_agency):
    # Standardize column names for consistency: lowercase
    df_market.columns = df_market.columns.str.strip().str.lower()
    df_agency.columns = df_agency.columns.str.strip().str.lower()

    # Sidebar to display column names for verification
    st.sidebar.header("📋 Colonnes des Données")
    st.sidebar.subheader("Données du Marché :")
    st.sidebar.write(df_market.columns.tolist())
    st.sidebar.subheader("Données de l'Agence :")
    st.sidebar.write(df_agency.columns.tolist())

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
st.sidebar.header("📊 Filtres")

# Date Range Filter
min_year = int(min(df_market['year'].min(), df_agency['year'].min()))
max_year = int(max(df_market['year'].max(), df_agency['year'].max()))
selected_years = st.sidebar.slider("Sélectionner l'intervalle de dates", min_year, max_year, (min_year, max_year))

# Commune Selection
communes = sorted(df_market['commune'].unique())
selected_communes = st.sidebar.multiselect("Sélectionner les Communes", communes, default=communes)

# Building Type Filter
if 'type de bâtiment' in df_market.columns:
    building_types = sorted(df_market['type de bâtiment'].dropna().unique())
    selected_building_types = st.sidebar.multiselect("Sélectionner les types de bâtiments", building_types, default=building_types)
else:
    st.error("Colonne 'type de bâtiment' non trouvée dans 'all_cities_belgium.csv'.")
    st.stop()

# Apply Filters
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
# Step 1: Agency Sales by Commune and Quarter (With White to Green Gradient)
# --------------------------------------
st.header("1. Ventes de l'Agence par Commune et Trimestre")

# Filter communes to only those where properties were sold by the agency
sold_communes = df_agency_filtered['commune'].unique()

# Filter agency data for sales in sold communes
df_agency_sold_communes = df_agency_filtered[df_agency_filtered['commune'].isin(sold_communes)]

# Get the earliest quarter where sales occurred in the agency data
earliest_agency_sale_date = df_agency_filtered['date'].min()
earliest_year = earliest_agency_sale_date.year
earliest_quarter = earliest_agency_sale_date.quarter

# Filter agency data to start from the earliest quarter of agency sales
df_agency_filtered = df_agency_filtered[(df_agency_filtered['year'] > earliest_year) | 
                                        ((df_agency_filtered['year'] == earliest_year) & 
                                         (df_agency_filtered['quarter'] >= earliest_quarter))]

# Aggregate total agency sales by commune and quarter
# Agency sales by commune and quarter
total_agency_sales = df_agency_filtered.groupby(['commune', 'year', 'quarter']).agg(
    total_agency_sales=('nombre de ventes', 'sum')
).reset_index()

# Calculate the total agency sales for each quarter
total_sales_per_quarter = total_agency_sales.groupby(['year', 'quarter']).agg(
    total_sales_quarter=('total_agency_sales', 'sum')
).reset_index()

# Merge the quarter totals back to agency sales
total_agency_sales = pd.merge(
    total_agency_sales,
    total_sales_per_quarter,
    on=['year', 'quarter'],
    how='left'
)

# Calculate percentage of agency sales for each commune
total_agency_sales['sales_percentage'] = (total_agency_sales['total_agency_sales'] / total_agency_sales['total_sales_quarter']) * 100

# Create unique quarter labels for readability in the table
total_agency_sales['quarter_label'] = total_agency_sales['year'].astype(str) + " T" + total_agency_sales['quarter'].astype(str)

# Pivot the table to create a format where rows are communes and columns are quarters, showing sales percentage
sales_percentage_table = total_agency_sales.pivot_table(
    index='commune', 
    columns='quarter_label', 
    values='sales_percentage',
    fill_value=0
)

# Sort index for better readability
sales_percentage_table = sales_percentage_table.sort_index()

# Function to apply gradient color from white to green based on value
def apply_green_gradient(data):
    styles = pd.DataFrame('', index=data.index, columns=data.columns)

    # Normalize data for coloring between 0% (white) to max % (green)
    max_value = data.max().max()  # Find max across all columns to normalize
    for column in data.columns:
        for index in data.index:
            value = data.loc[index, column]
            green_shade = int((value / max_value) * 255)  # Scale to get a value between 0 and 255 for green color intensity
            styles.at[index, column] = f'background-color: rgb({255 - green_shade}, 255, {255 - green_shade})'

    return styles

# Apply the green gradient function to the sales percentage table
styled_sales_percentage_table = sales_percentage_table.style.apply(apply_green_gradient, axis=None) \
                                                            .format("{:.2f}%")  # Formatting as percentage

# Render the table using Streamlit
st.dataframe(styled_sales_percentage_table, use_container_width=True)

# Add insight and recommendation for this section
st.markdown(""" Analyse : Le tableau montre le pourcentage des ventes réalisées par votre agence dans chaque commune au cours des trimestres disponibles. Le pourcentage est calculé comme la proportion des ventes de votre agence par rapport au total des ventes effectuées par votre agence sur l'ensemble des communes au cours d'un trimestre donné. Les valeurs sont affichées sous forme de pourcentages, ce qui permet de visualiser la part relative de chaque commune dans l'activité totale de votre agence. Le dégradé de couleur (du blanc au vert) facilite la distinction visuelle des meilleures performances.

**Comment lire le tableau :**

**Communes en lignes :** Chaque ligne correspond à une commune où des ventes ont été réalisées.
**Trimestres en colonnes :** Les colonnes correspondent aux différents trimestres de vente. Le label est de la forme "Année TTrimestre", par exemple "2022 T3" pour le troisième trimestre de 2022.
**Pourcentage des Ventes :** Chaque cellule représente le pourcentage de ventes réalisé par la commune pour un trimestre donné par rapport au total des ventes de l'agence sur toutes les communes ce même trimestre.
**Calculs :**

**Pourcentage de Ventes par Trimestre** = (Nombre de ventes de la commune / Total des ventes de l'agence ce trimestre) * 100.
**Dégradé de Couleur :** Plus la couleur est verte, plus la part de marché de la commune est importante. Cela permet d’identifier visuellement les communes qui ont la plus forte contribution.
**Recommandation :**

**Identifier les Communes Fortes :** Concentrez vos efforts sur les communes avec des teintes de vert foncé, car elles représentent une forte présence de votre agence.
Opportunités de Croissance : Pour les communes avec des teintes plus claires, explorez des stratégies pour améliorer la part de marché. """)



# --------------------------------------
# Step 0: Visualizations with Unique Keys
# --------------------------------------

# Function to generate unique keys
def generate_key(*args):
    return "_".join(map(str, args))


# --------------------------------------
# Step 2. Market Share Over Time by Commune (Line Chart)
# --------------------------------------
st.header("2. Part de Marché dans le Temps par Commune")

# Determine the earliest date where the agency sold a property
first_sale_date = df_agency['date'].min()

# Filter market share data from the first sale date onwards
market_share_filtered = market_share[market_share['date'] >= first_sale_date]

top_n = st.slider("Sélectionner les N meilleures Communes à afficher", min_value=1, max_value=20, value=5)
top_communes_over_time = market_share_filtered.groupby('commune')['market_share (%)'].mean().sort_values(ascending=False).head(top_n).index.tolist()

# Filter data for top N communes
market_share_top_n = market_share_filtered[market_share_filtered['commune'].isin(top_communes_over_time)]

# Plotting the line chart for market share over time
fig_market_share_time = px.line(
    market_share_top_n,
    x='date',
    y='market_share (%)',
    color='commune',
    markers=True,
    title='Part de Marché dans le Temps par les Communes Principales (Filtré aux Périodes de Ventes de l’Agence)',
    labels={'date': 'Date', 'market_share (%)': 'Part de Marché (%)', 'commune': 'Commune'}
)

# Add annotations for significant peaks
for commune in top_communes_over_time:
    commune_data = market_share_top_n[market_share_top_n['commune'] == commune]
    max_share = commune_data['market_share (%)'].max()
    max_date = commune_data[commune_data['market_share (%)'] == max_share]['date'].iloc[0]
    fig_market_share_time.add_annotation(
        x=max_date,
        y=max_share,
        text=f"Pic à {commune}",
        showarrow=True,
        arrowhead=1
    )

# Render the chart in Streamlit
st.plotly_chart(fig_market_share_time, use_container_width=True, key=generate_key('step6', 'market_share_over_time', 'line'))

# Insight and recommendation
st.markdown(""" Analyse : Le graphique montre l'évolution de la part de marché de votre agence par commune au fil du temps. Chaque ligne représente une commune particulière et l'évolution de sa part de marché à chaque trimestre. Les pics indiquent des périodes de forte part de marché tandis que les baisses peuvent signaler des pertes de compétitivité.

Comment lire le graphique :

Axe des X (horizontal) : Représente la date (trimestres successifs).
Axe des Y (vertical) : Représente la part de marché de l'agence, exprimée en pourcentage (%).
Lignes colorées : Chaque ligne représente une commune et montre comment la part de marché évolue trimestre par trimestre.
Calculs :

Part de Marché (%) = (Ventes de l'agence dans la commune / Ventes totales du marché dans la commune) * 100.
Annotations : Des annotations sont ajoutées pour signaler les pics importants (performances élevées). Cela vous aide à voir où l'agence a particulièrement bien performé.
Recommandation :

Analyser les Pics et Creux : Les pics montrent des périodes de forte performance, tandis que les creux peuvent indiquer des problèmes à traiter.
Reproduire le Succès : Utilisez ces tendances pour déterminer les stratégies qui ont fonctionné et les appliquer à d'autres communes. """)

# --------------------------------------
# Step 3. Quarterly Market Share by Commune (One Color per Quarter with Permanent Highlight)
# --------------------------------------
st.header("3. Part de Marché Trimestrielle par Commune (Une Couleur par Trimestre)")

# Calculate average market share per commune and quarter
market_share_quarterly = market_share.groupby(['commune', 'year', 'quarter']).agg({'market_share (%)': 'mean'}).reset_index()
market_share_quarterly['date'] = pd.to_datetime(market_share_quarterly['year'].astype(str) + '-Q' + market_share_quarterly['quarter'].astype(str))

# Sort by market share to highlight top-performing quarters across all communes
market_share_quarterly_sorted = market_share_quarterly.sort_values('market_share (%)', ascending=False).head(20)

# Create unique labels for each quarter and commune combination
market_share_quarterly_sorted['label'] = market_share_quarterly_sorted.apply(
    lambda row: f"{row['commune']} - {row['date'].strftime('%b %Y')}", axis=1
)

# Assign each quarter a unique color
market_share_quarterly_sorted['quarter_label'] = market_share_quarterly_sorted['year'].astype(str) + " T" + market_share_quarterly_sorted['quarter'].astype(str)

# Identify the latest two quarters
latest_dates = market_share_quarterly['date'].drop_duplicates().sort_values(ascending=False).head(2)

# Plot: Separate each quarter and each commune
fig_top20_quarters = px.bar(
    market_share_quarterly_sorted,
    x='label',
    y='market_share (%)',
    color='quarter_label',  # Color by quarter
    title='Part de Marché Trimestrielle par Commune (Top 20 par Trimestre et Commune)',
    labels={'market_share (%)': 'Part de Marché (%)', 'label': 'Commune - Trimestre', 'quarter_label': 'Trimestre'},
    color_discrete_sequence=px.colors.qualitative.Pastel
)

# Update x-axis labels to highlight latest two quarters by default
highlighted_labels = [
    f"<b style='font-size:16px'>{label}</b>" if row.date in latest_dates.values else label
    for label, row in zip(market_share_quarterly_sorted['label'], market_share_quarterly_sorted.itertuples())
]
fig_top20_quarters.update_xaxes(
    ticktext=highlighted_labels,
    tickvals=market_share_quarterly_sorted['label']
)

fig_top20_quarters.update_layout(
    xaxis={'categoryorder': 'total descending', 'tickangle': 45},
    yaxis_title='Part de Marché (%)',
    hovermode='x unified',
    annotations=[
        dict(
            text=(
                "Chaque barre représente un trimestre spécifique pour une commune particulière.\n"
                "Cette visualisation aide à identifier les trimestres les plus réussis pour chaque commune individuellement, avec une couleur distincte par trimestre."
            ),
            xref='paper',
            yref='paper',
            x=0.5,
            y=1.08,
            showarrow=False,
            font=dict(size=12),
            align='center'
        )
    ],
    height=600
)

fig_top20_quarters.update_traces(textposition='inside', marker_line_width=1.5)

# Render the plot
st.plotly_chart(fig_top20_quarters, use_container_width=True, key=generate_key('step7', 'separated_quarters_communes_market_share', 'highlight'))

# Add insight and recommendation for this section
st.markdown(""" Analyse : Le graphique en barres montre la part de marché trimestrielle de chaque commune, chaque barre représentant un trimestre spécifique. Les différentes couleurs aident à distinguer les trimestres, facilitant l'identification des performances saisonnières.

Comment lire le graphique :

Axe des X (horizontal) : Représente chaque commune et trimestre, avec un libellé indiquant la combinaison "Commune - Trimestre".
Axe des Y (vertical) : Indique la part de marché (%) pour chaque commune et trimestre.
Couleurs : Chaque couleur correspond à un trimestre donné, permettant une comparaison visuelle des différents trimestres au sein d'une même commune.
Calculs :

Part de Marché (%) = (Ventes de l'agence / Total des ventes du marché) * 100 pour chaque commune et trimestre.
Classement : Le graphique est trié pour montrer les périodes les plus performantes (les valeurs les plus élevées sont affichées en premier).
Recommandation :

Suivi de la Performance Saisonnière : Utilisez ces différentes couleurs pour comprendre comment la performance varie selon les trimestres.
Optimisation des Ressources : Planifiez vos actions marketing pour répliquer les succès des trimestres les plus performants. """)


# --------------------------------------
# Step 4. Quarterly Sales Volume Percentage by Commune (One Color per Quarter with Permanent Highlight)
# --------------------------------------

st.markdown("\n\n")  # Ajouter des sauts de ligne avant le début du chapitre
st.header("4. Pourcentage du Volume de Ventes Trimestriel par Commune (Une Couleur par Trimestre)")

# Calculate total sales volume per commune and quarter
sales_volume_quarterly = market_share.groupby(['commune', 'year', 'quarter']).agg({'nombre de ventes_agency': 'sum'}).reset_index()
sales_volume_quarterly['date'] = pd.to_datetime(sales_volume_quarterly['year'].astype(str) + '-Q' + sales_volume_quarterly['quarter'].astype(str))

# Calculate total sales across all communes for each quarter
total_sales_per_quarter = sales_volume_quarterly.groupby(['year', 'quarter'])['nombre de ventes_agency'].sum().reset_index()
total_sales_per_quarter['date'] = pd.to_datetime(total_sales_per_quarter['year'].astype(str) + '-Q' + total_sales_per_quarter['quarter'].astype(str))

# Merge to calculate percentage contribution of each commune
sales_volume_quarterly = sales_volume_quarterly.merge(total_sales_per_quarter, on=['year', 'quarter', 'date'], suffixes=('', '_total'))
sales_volume_quarterly['sales_volume_percentage'] = (sales_volume_quarterly['nombre de ventes_agency'] / sales_volume_quarterly['nombre de ventes_agency_total']) * 100

# Sort by sales volume percentage to highlight top-performing quarters across all communes
sales_volume_quarterly_sorted = sales_volume_quarterly.sort_values('sales_volume_percentage', ascending=False).head(20)

# Create unique labels for each quarter and commune combination
sales_volume_quarterly_sorted['label'] = sales_volume_quarterly_sorted.apply(
    lambda row: f"{row['commune']} - {row['date'].strftime('%b %Y')}", axis=1
)

# Assign each quarter a unique color
sales_volume_quarterly_sorted['quarter_label'] = sales_volume_quarterly_sorted['year'].astype(str) + " T" + sales_volume_quarterly_sorted['quarter'].astype(str)

# Identify the latest two quarters
latest_dates = sales_volume_quarterly['date'].drop_duplicates().sort_values(ascending=False).head(2)

# Always highlight the latest two quarters by updating their x-axis labels
highlighted_labels = [
    f"<b style='font-size:16px'>{label}</b>" if row.date in latest_dates.values else label
    for label, row in zip(sales_volume_quarterly_sorted['label'], sales_volume_quarterly_sorted.itertuples())
]

# Plot: Separate each quarter and each commune
fig_top20_quarters = px.bar(
    sales_volume_quarterly_sorted,
    x='label',
    y='sales_volume_percentage',
    color='quarter_label',  # Color by quarter
    title='Pourcentage du Volume de Ventes Trimestriel par Commune (Top 20 par Trimestre et Commune)',
    labels={'sales_volume_percentage': 'Volume des Ventes (%)', 'label': 'Commune - Trimestre', 'quarter_label': 'Trimestre'},
    color_discrete_sequence=px.colors.qualitative.Pastel
)

# Update x-axis labels to highlight latest two quarters
fig_top20_quarters.update_xaxes(
    ticktext=highlighted_labels,
    tickvals=sales_volume_quarterly_sorted['label']
)

fig_top20_quarters.update_layout(
    xaxis={'categoryorder': 'total descending', 'tickangle': 45},
    yaxis_title='Volume des Ventes (%)',
    hovermode='x unified',
    annotations=[
        dict(
            text=(
                "Chaque barre représente un trimestre spécifique pour une commune particulière.\n"
                "Cette visualisation aide à identifier les trimestres les plus fructueux pour chaque commune, chaque trimestre étant représenté par une couleur distincte."
            ),
            xref='paper',
            yref='paper',
            x=0.5,
            y=1.08,
            showarrow=False,
            font=dict(size=12),
            align='center'
        )
    ],
    height=600
)

fig_top20_quarters.update_traces(textposition='inside', marker_line_width=1.5)

# Display the plot
st.plotly_chart(fig_top20_quarters, use_container_width=True, key=generate_key('step7bis', 'separated_quarters_communes_sales_volume', 'highlight'))

# Add insights and recommendations
st.markdown("""
**Analyse :** Ce graphique en barres affiche les 20 meilleurs pourcentages de volume de ventes, séparés par chaque trimestre et chaque commune. Chaque trimestre est représenté par une couleur unique, permettant une différenciation facile et une meilleure compréhension des périodes les plus fructueuses en termes de volume de ventes.

**Comment lire le graphique :**

- **Axe des X (horizontal) :** Représente les communes et les trimestres.
- **Axe des Y (vertical) :** Représente le pourcentage du volume de ventes par trimestre.
- **Couleurs distinctes par trimestre :** Pour visualiser facilement les différences entre chaque période.

**Recommandation :**

- **Suivi de la Performance Saisonnière :** Utilisez les trimestres colorés pour analyser comment la performance fluctue au fil des saisons en termes de volume de ventes.
- **Optimisation des Campagnes :** Comprendre les trimestres les plus performants permet d'optimiser la planification des campagnes et des actions stratégiques.
- **Comparaison entre Trimestres :** Colorier chaque trimestre permet d'identifier les trimestres avec des performances similaires ou différentes entre les communes.
""")



# --------------------------------------
# Step 5. Sales Volume vs. Market Share by Commune (Bubble Chart with Logarithmic Scale)
# --------------------------------------
st.markdown("\n\n")  # Ajouter des sauts de ligne avant le début du chapitre
st.header("5. Volume des Ventes vs Part de Marché par Commune (Échelle Logarithmique)")

# Aggregate data for bubble chart
sales_vs_market_share = market_share.groupby('commune').agg({
    'nombre de ventes_market': 'sum',
    'nombre de ventes_agency': 'sum',
    'market_share (%)': 'mean'
}).reset_index()

# Plot with logarithmic axes
fig_bubble_log = px.scatter(
    sales_vs_market_share,
    x='nombre de ventes_market',
    y='nombre de ventes_agency',
    size='market_share (%)',
    color='market_share (%)',
    hover_name='commune',
    title='Volume des Ventes vs Part de Marché par Commune (Échelle Logarithmique)',
    labels={'nombre de ventes_market': 'Ventes Totales du Marché', 'nombre de ventes_agency': 'Ventes Agence', 'market_share (%)': 'Part de Marché (%)'},
    size_max=60
)

# Update layout to use logarithmic scale for x and y axes
fig_bubble_log.update_layout(
    xaxis_title='Ventes Totales du Marché (Échelle Logarithmique)',
    yaxis_title='Ventes Agence (Échelle Logarithmique)',
    hovermode='closest',
    xaxis=dict(type='log'),
    yaxis=dict(type='log')
)

st.plotly_chart(fig_bubble_log, use_container_width=True, key=generate_key('step8', 'sales_volume_market_share_log', 'bubble'))

st.markdown("""
**Analyse :** Le graphique à bulles avec des axes logarithmiques illustre la relation entre les ventes totales du marché et les ventes de votre agence dans les différentes communes, même lorsque les valeurs varient considérablement. Les bulles plus grandes indiquent une part de marché plus élevée.

**Comment lire le graphique :**

- **Axes Logarithmiques :** Les axes des ventes de marché et des ventes de l'agence sont en échelle logarithmique, ce qui permet de visualiser des différences très importantes entre les communes.
- **Taille des Bulles :** Plus la bulle est grande, plus la part de marché de la commune est importante.

**Recommandation :**

- **Cibler les Communes Potentielles :** Concentrez-vous sur les communes avec des ventes totales du marché élevées mais où votre agence a une faible présence.
- **Améliorer la Stratégie :** Utilisez des efforts de marketing plus ciblés pour améliorer vos parts de marché dans les communes sous-performantes.
""")



# --------------------------------------
# Step 6. Market Share by Commune Over Time (Side-by-Side Bar Chart with Total Market Share Line)
# --------------------------------------
st.markdown("\n\n")  # Ajouter des sauts de ligne avant le début du chapitre

st.header("6. Part de Marché par Commune au Fil du Temps (Graphique en Barres Comparatives avec Ligne de Part de Marché Totale)")

# Calculate average market share per commune and reuse for getting top 5 communes
avg_market_share_commune = market_share.groupby('commune').agg({'market_share (%)': 'mean'}).reset_index()

# Get top 5 communes by average market share
top5_communes_over_time = avg_market_share_commune.sort_values('market_share (%)', ascending=False).head(5)['commune'].tolist()

# Filter market_share data for the top 5 communes
market_share_top5 = market_share[market_share['commune'].isin(top5_communes_over_time)]

# Determine the earliest date where the agency sold a property
first_sale_date = df_agency['date'].min()
# Filter market share data from the first sale date onwards
market_share_top5_filtered = market_share_top5[market_share_top5['date'] >= first_sale_date]

# Calculate total market share percentage for each date
total_market_share = market_share_top5_filtered.groupby('date')['market_share (%)'].mean().reset_index()

# Plotting the side-by-side bar chart for individual communes and line for total market share
fig_combined = go.Figure()

# Add bars for individual communes for each date
for commune in market_share_top5_filtered['commune'].unique():
    commune_data = market_share_top5_filtered[market_share_top5_filtered['commune'] == commune]
    fig_combined.add_trace(
        go.Bar(
            x=commune_data['date'],
            y=commune_data['market_share (%)'],
            name=commune,
            hoverinfo='x+y+name'
        )
    )

# Add a line for average total market share percentage over time
fig_combined.add_trace(
    go.Scatter(
        x=total_market_share['date'],
        y=total_market_share['market_share (%)'],
        mode='lines+markers',
        name='Part de Marché Moyenne (%)',
        line=dict(color='black', width=3)
    )
)

# Update layout for better readability
fig_combined.update_layout(
    barmode='group',  # Use 'group' instead of 'stack' or 'overlay'
    title='Part de Marché par Commune au Fil du Temps (Top 5 Communes et Part de Marché Moyenne)',
    xaxis=dict(title='Date', tickformat='%b %Y'),
    yaxis=dict(title='Part de Marché (%)', tickformat=',.2f', range=[0, 100]),  # Set range from 0 to 100 to reflect percentages
    hovermode='x unified',
    height=600,
    legend_title_text='Communes',
    margin=dict(l=40, r=40, t=60, b=40),
)

# Add annotations to highlight peaks in total market share
max_total_share = total_market_share['market_share (%)'].max()
max_total_date = total_market_share[total_market_share['market_share (%)'] == max_total_share]['date'].iloc[0]

fig_combined.add_annotation(
    x=max_total_date,
    y=max_total_share,
    text=f"Pic de Part de Marché Moyenne",
    showarrow=True,
    arrowhead=2,
    font=dict(size=10),
    arrowcolor="green"
)

# Render the chart
st.plotly_chart(fig_combined, use_container_width=True, key=generate_key('step9', 'market_share_grouped', 'line_bar'))

# Insight and recommendation
st.markdown("""
**Analyse :** Le graphique montre comment la part de marché dans les 5 principales communes a évolué au fil du temps, à partir du premier trimestre où votre agence a vendu un bien. La ligne noire représente la part de marché moyenne combinée des 5 principales communes.

**Comment lire le graphique :**

- **Barres par Commune :** Chaque barre représente la part de marché trimestrielle d'une commune.
- **Ligne Noire :** La ligne représente la moyenne totale, indiquant les tendances générales de croissance ou de déclin.

**Recommandation :**

- **Identifier les Périodes Clés :** Concentrez-vous sur l'identification des périodes de croissance et de déclin.
- **Allocation Stratégique :** Allouez davantage de ressources aux communes où la part de marché est en hausse constante.
""")



# --------------------------------------
# Step 7. Sales Volume Heatmap by Commune (Folium Map with Hover Details)
# --------------------------------------
st.markdown("\n\n")  # Ajouter des sauts de ligne avant le début du chapitre

st.header("7. Carte de Chaleur du Volume de Ventes par Commune avec Détails au Survol")

if not df_agency_filtered.empty and 'latitude' in df_agency_filtered.columns and 'longitude' in df_agency_filtered.columns:
    sales_heatmap_data = df_agency_filtered[['latitude', 'longitude', 'nombre de ventes', 'commune']].dropna()
    if not sales_heatmap_data.empty:
        # Create a Folium map centered around the mean latitude and longitude
        map_center = [sales_heatmap_data['latitude'].mean(), sales_heatmap_data['longitude'].mean()]
        sales_heatmap = folium.Map(location=map_center, zoom_start=9, tiles='CartoDB positron')

        # Add the heatmap layer for sales data
        HeatMap(
            data=sales_heatmap_data[['latitude', 'longitude', 'nombre de ventes']].values.tolist(),
            radius=25,  # Increased radius for smoother blending
            blur=15,    # Increased blur for a more aesthetic spread
            max_zoom=12
        ).add_to(sales_heatmap)

        # Render the Folium map in Streamlit
        folium_static(sales_heatmap, width=700, height=500)

        st.markdown("""
        **Analyse :** La carte de chaleur visualise la répartition géographique du volume des ventes, avec des détails sur le nombre de ventes dans chaque commune.

        **Recommandation :**

        - **Ciblage Marketing :** Orientez vos ressources marketing vers les zones à forte densité de ventes.
        - **Opportunités d'Expansion :** Évaluez les zones ayant un volume de ventes moyen pour de futures opportunités de croissance.
        """)

    else:
        st.write("Aucune donnée de vente disponible pour tracer la carte de chaleur.")
else:
    st.write("Données insuffisantes pour générer la carte de chaleur du volume de ventes. Veuillez vérifier que les colonnes 'latitude' et 'longitude' sont présentes et contiennent des données valides.")

# --------------------------------------
# Step 8: Top Performing Communes by Market Share (Bar Chart)
# --------------------------------------
st.markdown("\n\n")  # Ajouter des sauts de ligne avant le début du chapitre

st.header("8. Communes les Plus Performantes par Part de Marché")

# Aggregate market share data to identify top-performing communes
avg_market_share_commune = market_share.groupby('commune').agg({'market_share (%)': 'mean'}).reset_index()
# Get top 10 communes by average market share
top_communes = avg_market_share_commune.sort_values('market_share (%)', ascending=False).head(10)

# Plot top-performing communes
fig_top_communes = px.bar(
    top_communes,
    x='commune',
    y='market_share (%)',
    title='Top 10 Communes les Plus Performantes par Part de Marché',
    labels={'market_share (%)': 'Part de Marché (%)', 'commune': 'Commune'},
    text='market_share (%)'
)
fig_top_communes.update_layout(
    xaxis_title='Commune',
    yaxis_title='Part de Marché (%)',
    hovermode='x unified'
)
fig_top_communes.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

st.plotly_chart(fig_top_communes, use_container_width=True, key=generate_key('step12', 'top_performing_communes'))

st.markdown("""
**Analyse :** Ce graphique compact montre la part de marché de votre agence par rapport à la moyenne du marché pour les six communes les plus vendues. Les zones vertes indiquent la surperformance de l'agence, tandis que les zones rouges montrent une sous-performance.

**Recommandation :**

- **Améliorer les Sous-performances :** Analysez les raisons des sous-performances et développez des stratégies pour améliorer.
- **Capitaliser sur les Forces :** Concentrez-vous sur les communes avec des performances élevées pour maintenir l'avantage.
- **Exploiter les Périodes de Pic :** Utilisez les périodes de pic pour comprendre ce qui fonctionne et reproduisez ce succès ailleurs.
""")

# --------------------------------------
# Step 9: Quarterly Market Share Trends by Sold Communes (Ultra Compact Version)
# --------------------------------------
st.markdown("\n\n")  # Ajouter des sauts de ligne avant le début du chapitre

st.header("9. Tendances Trimestrielles de Part de Marché par Communes Vendues (Version Ultra Compacte)")

# Filter communes to only those where properties were sold by the agency
sold_communes = df_agency['commune'].unique()
market_share_sold_communes = market_share[market_share['commune'].isin(sold_communes)]

# Get the most recent 8 periods (last 2 years)
latest_dates = market_share_sold_communes['date'].drop_duplicates().nlargest(8)
market_share_sold_communes = market_share_sold_communes[market_share_sold_communes['date'].isin(latest_dates)]

# Get top 6 communes by sales volume for visualization
top_6_communes = df_agency_filtered['commune'].value_counts().head(6).index
market_share_top_6 = market_share_sold_communes[market_share_sold_communes['commune'].isin(top_6_communes)]

# Create a compact multi-chart display for the top 6 communes in three columns
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2,
    cols=3,
    shared_xaxes=True,
    vertical_spacing=0.15,  # Adjusted spacing to make the chart more compact
    horizontal_spacing=0.1,
    subplot_titles=[f"Commune: {commune}" for commune in top_6_communes]
)

# Plot each commune in its own subplot
for i, commune in enumerate(top_6_communes):
    row = i // 3 + 1
    col = i % 3 + 1
    commune_data = market_share_top_6[market_share_top_6['commune'] == commune]
    overall_market_share = commune_data['market_share (%)'].mean()

    # Add line for your agency's market share with data labels
    fig.add_trace(go.Scatter(
        x=commune_data['date'],
        y=commune_data['market_share (%)'],
        mode='lines+markers+text',
        name=f'Part de Marché de l\'Agence à {commune}',
        line=dict(color='blue'),
        marker=dict(size=6),
        text=[f"{y:.1f}%" for y in commune_data['market_share (%)']],
        textposition='top center',
        hovertemplate='%{x}: %{y:.2f}% (Agence)'
    ), row=row, col=col)

    # Add line for market average for visual comparison with data labels
    fig.add_trace(go.Scatter(
        x=commune_data['date'],
        y=[overall_market_share] * len(commune_data),
        mode='lines',
        name=f'Moyenne du Marché à {commune}',
        line=dict(dash='dash', color='red'),
        hovertemplate='Moyenne Marché: %{y:.2f}%'
    ), row=row, col=col)

    # Add filled area for overperformance of the agency
    fig.add_trace(go.Scatter(
        x=commune_data['date'].tolist() + commune_data['date'].tolist()[::-1],
        y=[
            max(agency, overall_market_share) for agency in commune_data['market_share (%)']
        ] + [overall_market_share] * len(commune_data),
        fill='toself',
        fillcolor='rgba(34, 139, 34, 0.2)',
        line=dict(color='rgba(0, 0, 0, 0)'),
        name=f'Surperformance à {commune}',
        legendgroup=f'Surperformance {commune}',
        showlegend=(i == 0),
        hoverinfo='skip'
    ), row=row, col=col)

    # Add filled area for underperformance of the agency
    fig.add_trace(go.Scatter(
        x=commune_data['date'].tolist() + commune_data['date'].tolist()[::-1],
        y=[
            min(agency, overall_market_share) for agency in commune_data['market_share (%)']
        ] + [overall_market_share] * len(commune_data),
        fill='toself',
        fillcolor='rgba(220, 20, 60, 0.2)',
        line=dict(color='rgba(0, 0, 0, 0)'),
        name=f'Sous-performance à {commune}',
        legendgroup=f'Sous-performance {commune}',
        showlegend=(i == 0),
        hoverinfo='skip'
    ), row=row, col=col)

# Mise à jour du layout pour une meilleure lisibilité et compacité
fig.update_layout(
    height=1000,  # Adjusted height to fit more comfortably with three columns
    title_text="Tendances Trimestrielles de Part de Marché pour les 6 Communes les Plus Vendues (Comparaison Agence vs Marché - Dernières 8 Périodes)",
    xaxis_title='Date',
    yaxis_title='Part de Marché (%)',
    hovermode='x unified',
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5
    ),
    annotations=[
        dict(
            text=(
                "Comparaison de la part de marché de l'agence (Ligne Bleue) avec la moyenne du marché (Ligne Rouge Pointillée).<br>"
                "Les zones vertes représentent une surperformance de l'agence par rapport à la moyenne du marché.<br>"
                "Les zones rouges montrent une sous-performance par rapport à la moyenne."
            ),
            xref='paper',
            yref='paper',
            x=0.5,
            y=1.15,
            showarrow=False,
            font=dict(size=12, family='Arial, sans-serif'),
            align='center'
        )
    ]
)

# Rendu du graphique dans Streamlit
st.plotly_chart(fig, use_container_width=True, key=generate_key('step12b', 'top_6_communes_comparison_fill'))

st.markdown("""
**Analyse :** Ce graphique compact montre la part de marché de votre agence par rapport à la moyenne du marché pour les six communes les plus vendues. Les zones vertes indiquent la surperformance de l'agence, tandis que les zones rouges montrent une sous-performance.

**Recommandation :** 
- **Concentrez les efforts sur les sous-performances** : Pour les communes avec des zones rouges, analysez les raisons possibles et développez des stratégies pour améliorer la part de marché.
- **Renforcez la surperformance** : Dans les communes avec des zones vertes significatives, il est conseillé de capitaliser sur cette position forte avec des actions ciblées pour renforcer l'avantage.
- **Exploitez les périodes de pic** : Utilisez les périodes de pic indiquées pour mieux comprendre ce qui fonctionne et essayer de répliquer ce succès dans d'autres zones sous-performantes.
""")

# --------------------------------------
# Step 10 Market Share Analysis by Commune (Bubble Chart)
# --------------------------------------
st.markdown("\n\n")  # Ajouter des sauts de ligne avant le début du chapitre
st.header("10. Analyse de la Part de Marché par Commune")

# Fusionner les données de la part de marché moyenne pour chaque commune
# On part du principe que 'avg_market_share_commune' est déjà calculé et disponible

# Créer un graphique à bulles pour analyser la part de marché par commune
fig_market_share = px.scatter(
    avg_market_share_commune,
    x='commune',
    y='market_share (%)',
    size='market_share (%)',
    color='commune',
    title="Analyse de la Part de Marché par Commune",
    labels={'market_share (%)': 'Part de Marché (%)', 'commune': 'Commune'},
    size_max=60,
)

fig_market_share.update_layout(
    xaxis_title="Commune",
    yaxis_title='Part de Marché (%)',
    hovermode='closest',
    showlegend=False  # Supprimer la légende pour plus de clarté si nécessaire
)

st.plotly_chart(fig_market_share, use_container_width=True, key=generate_key('step12', 'market_share_by_commune'))

# Nouvelle analyse et recommandation
st.markdown("""
**Analyse :** Le graphique à bulles représente la part de marché de votre agence dans chaque commune. Chaque bulle représente une commune et sa taille indique la part de marché de l'agence dans cette commune. Plus la bulle est grande, plus votre agence détient une grande part de marché dans cette zone géographique.

Il est intéressant de noter les disparités de tailles de bulles entre les différentes communes. Cela vous permet d'identifier les communes dans lesquelles votre agence est dominante par rapport aux autres communes où la part de marché est relativement faible.

**Recommandation :**
- **Prioriser les Communes avec une Grande Part de Marché** : Pour les communes où votre part de marché est élevée (les plus grandes bulles), il serait judicieux de renforcer votre présence en consolidant vos actions marketing et en fidélisant davantage les clients actuels. Cela permettra de maintenir, voire d'augmenter, cette position dominante.
  
- **Analyser les Communes avec une Faible Part de Marché** : Pour les communes où votre part de marché est faible (les petites bulles), évaluez les facteurs qui pourraient expliquer ce manque de performance. Est-ce la concurrence qui est plus forte ? Est-ce une question de ciblage des clients ? Ces questions doivent guider vos décisions sur la manière d'aborder ces communes.
  
- **Exemples Pratiques :**
  - Si la commune **A** possède une grande bulle représentant une part de marché de **40 %**, il est crucial de continuer à renforcer votre stratégie dans cette zone, en investissant dans la publicité locale ou en établissant des partenariats avec d'autres acteurs locaux.
  - Dans le cas de la commune **B**, avec une bulle plus petite (par exemple **10 %** de part de marché), envisagez des stratégies pour augmenter votre présence, comme l'optimisation des campagnes numériques ciblées ou l'amélioration de la perception de votre marque.

L'objectif est de maximiser la présence dans les zones où votre agence est déjà bien positionnée tout en trouvant des opportunités de croissance dans les autres communes.
""")



# --------------------------------------
# Step 11. Sales Distribution by Property Type (Pie Chart)
# --------------------------------------
st.markdown("\n\n")  # Ajouter des sauts de ligne avant le début du chapitre

st.header("11. Répartition des Ventes par Type de Propriété")

# Assuming df_agency_filtered contains 'type de bâtiment'
if 'type de bâtiment' in df_market_filtered.columns:
    # Market sales by property type
    market_property_type = df_market_filtered.groupby('type de bâtiment').agg({'nombre de ventes': 'sum'}).reset_index()
    market_property_type.rename(columns={'nombre de ventes': 'nombre de ventes_market'}, inplace=True)

    if not market_property_type.empty:
        # Plot Market Sales Distribution
        fig_pie_market = px.pie(
            market_property_type,
            names='type de bâtiment',
            values='nombre de ventes_market',
            title='Répartition des Ventes sur le Marché par Type de Propriété',
            hole=0.3
        )
        st.plotly_chart(fig_pie_market, use_container_width=True, key=generate_key('step13', 'market_sales_distribution', 'pie'))

        st.markdown("""
        **Analyse :** Le diagramme circulaire montre la répartition des ventes sur le marché parmi différents types de propriétés, permettant d'identifier les types de propriétés les plus populaires.

        **Recommandation :** Concentrez-vous sur l'augmentation des inscriptions pour les types de propriétés ayant des ventes élevées, ou explorez des stratégies pour augmenter les ventes dans les segments sous-performants.
        """)
    else:
        st.write("Données insuffisantes pour tracer la Répartition des Ventes par Type de Propriété.")
else:
    st.write("Colonne 'type de bâtiment' non trouvée dans les données du marché.")
# --------------------------------------
# Step 12. Competitor Benchmark by Commune (Stacked Bar Chart with Checkbox Filter)
# --------------------------------------
st.markdown("\n\n")  # Ajouter des sauts de ligne avant le début du chapitre

st.header("12. Benchmark des Concurrents par Commune")

# Calculate total market sales by commune
total_sales_market = df_market_filtered.groupby('commune').agg({'nombre de ventes': 'sum'}).reset_index()
total_sales_market.rename(columns={'nombre de ventes': 'nombre de ventes_market'}, inplace=True)

# Calculate total agency sales by commune
total_sales_agency = df_agency_filtered.groupby('commune').agg({'nombre de ventes': 'sum'}).reset_index()
total_sales_agency.rename(columns={'nombre de ventes': 'nombre de ventes_agency'}, inplace=True)

# Merge market and agency sales data
competitor_sales = pd.merge(
    total_sales_market,
    total_sales_agency,
    on='commune',
    how='left'
).fillna({'nombre de ventes_agency': 0})

# Calculate competitor sales
competitor_sales['competitor_sales'] = competitor_sales['nombre de ventes_market'] - competitor_sales['nombre de ventes_agency']
competitor_sales['competitor_sales'] = competitor_sales['competitor_sales'].apply(lambda x: x if x >= 0 else 0)

# Checkbox for filtering "Mes Communes"
mes_communes = st.checkbox("Mes Communes", value=False)

# Filter for top 20 communes by market sales or only communes where the agency sold properties
if mes_communes:
    communes_with_sales = total_sales_agency[total_sales_agency['nombre de ventes_agency'] > 0]['commune'].unique()
    competitor_sales_filtered = competitor_sales[competitor_sales['commune'].isin(communes_with_sales)]
else:
    competitor_sales_filtered = competitor_sales.sort_values('nombre de ventes_market', ascending=False).head(20)

# Melt data for stacked bar chart
competitor_sales_melted = competitor_sales_filtered.melt(
    id_vars='commune',
    value_vars=['nombre de ventes_agency', 'competitor_sales'],
    var_name='agency_vs_competitors',
    value_name='sales'
)

# Rename for clarity
competitor_sales_melted['agency_vs_competitors'] = competitor_sales_melted['agency_vs_competitors'].map({
    'nombre de ventes_agency': 'Ventes de l\'Agence',
    'competitor_sales': 'Ventes des Concurrents'
})

if not competitor_sales_melted.empty:
    # Plot
    fig_competitor = px.bar(
        competitor_sales_melted,
        x='commune',
        y='sales',
        color='agency_vs_competitors',
        title='Ventes de l\'Agence vs Ventes des Concurrents par Commune (Top 20)' if not mes_communes else 'Ventes de l\'Agence vs Ventes des Concurrents (Mes Communes)',
        labels={'sales': 'Nombre de Ventes', 'agency_vs_competitors': 'Type de Ventes'},
        barmode='stack'
    )
    fig_competitor.update_layout(xaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig_competitor, use_container_width=True, key=generate_key('step14', 'competitor_benchmark', 'stacked_bar', 'mes_communes' if mes_communes else 'top_20'))

    st.markdown(""" Analyse : Le graphique à barres empilées montre les ventes de votre agence par rapport à celles des concurrents dans chaque commune. Cela permet de visualiser la position de l'agence sur chaque marché local.

    Comment lire le graphique :

    Axe des X (horizontal) : Représente chaque commune.
    Axe des Y (vertical) : Représente le nombre de ventes.
    Barres Empilées : Les différentes couleurs des barres représentent les ventes de votre agence et celles des concurrents.
    Calculs :

    Part des Concurrents = Ventes totales du marché - Ventes de l'agence.
    Recommandation :

    Analyser la Concurrence : Identifiez les communes où la concurrence est plus forte et ajustez vos stratégies pour augmenter votre part de marché, par exemple par des promotions ciblées ou des actions de communication spécifiques. """)
else:
    st.write("Aucune donnée disponible pour le Benchmark des Concurrents par Commune.")


# --------------------------------------
# Step 13. Sales Trends Over Time (Quarterly)
# --------------------------------------
st.markdown("\n\n")  # Ajouter des sauts de ligne avant le début du chapitre

st.header("13. Tendances des Ventes au Fil du Temps (Trimestriel)")

# Aggregate quarterly sales
agency_sales_time = df_agency_filtered.groupby(['year', 'quarter']).agg({'nombre de ventes': 'sum'}).reset_index()
market_sales_time = df_market_filtered.groupby(['year', 'quarter']).agg({'nombre de ventes': 'sum'}).reset_index()

# Convert year and quarter to datetime for consistency
agency_sales_time['date'] = pd.to_datetime(agency_sales_time['year'].astype(str) + '-Q' + agency_sales_time['quarter'].astype(str))
market_sales_time['date'] = pd.to_datetime(market_sales_time['year'].astype(str) + '-Q' + market_sales_time['quarter'].astype(str))

# Filter the data starting from the earliest date where the agency has sales
earliest_agency_date = agency_sales_time['date'].min()

# Filter both agency and market sales to start from the earliest agency sale date
agency_sales_time = agency_sales_time[agency_sales_time['date'] >= earliest_agency_date]
market_sales_time = market_sales_time[market_sales_time['date'] >= earliest_agency_date]

# Merge data on date to compare agency and market sales
sales_trends = pd.merge(
    agency_sales_time[['date', 'nombre de ventes']],
    market_sales_time[['date', 'nombre de ventes']],
    on='date',
    how='outer',
    suffixes=('_agency', '_market')
).fillna(0)

# Sort the values by date
sales_trends.sort_values('date', inplace=True)

# Calculate market share over time
sales_trends['market_share (%)'] = (sales_trends['nombre de ventes_agency'] / sales_trends['nombre de ventes_market']) * 100
sales_trends['market_share (%)'] = sales_trends['market_share (%)'].replace([np.inf, -np.inf], 0).fillna(0)

if not sales_trends.empty:
    # Plot the market share trends over time
    fig_sales_trends = go.Figure()
    fig_sales_trends.add_trace(go.Scatter(
        x=sales_trends['date'], 
        y=sales_trends['market_share (%)'], 
        mode='lines+markers', 
        name='Part de Marché (%)',
        line=dict(color='blue'),
        marker=dict(size=6),
        text=[f"{y:.1f}%" for y in sales_trends['market_share (%)']],
        textposition='top center'
    ))

    # Update layout for better readability
    fig_sales_trends.update_layout(
        title='Tendances de la Part de Marché au Fil du Temps (Trimestriel)',
        xaxis_title='Date',
        yaxis_title='Part de Marché (%)',
        hovermode='x unified',
        height=500  # Make the plot compact
    )

    # Add annotations for peaks and troughs
    sales_trends['is_peak'] = sales_trends['market_share (%)'] > sales_trends['market_share (%)'].quantile(0.75)
    sales_trends['is_trough'] = sales_trends['market_share (%)'] < sales_trends['market_share (%)'].quantile(0.25)

    for idx, row in sales_trends.iterrows():
        if row['is_peak']:
            fig_sales_trends.add_annotation(
                x=row['date'],
                y=row['market_share (%)'],
                text="Pic de Part de Marché",
                showarrow=True,
                arrowhead=1,
                arrowcolor="green",
                font=dict(size=10)
            )
        elif row['is_trough']:
            fig_sales_trends.add_annotation(
                x=row['date'],
                y=row['market_share (%)'],
                text="Creux de Part de Marché",
                showarrow=True,
                arrowhead=1,
                arrowcolor="red",
                font=dict(size=10)
            )

    # Render the plot in Streamlit
    st.plotly_chart(fig_sales_trends, use_container_width=True, key=generate_key('step15', 'market_share_trends'))

    # Insight and Recommendation
    st.markdown(""" Analyse : Le graphique linéaire illustre les tendances trimestrielles de la part de marché de votre agence comparée à celle du marché global. Cela vous permet de suivre l'évolution de votre position concurrentielle.

    Comment lire le graphique :

    Axe des X (horizontal) : Représente les dates par trimestre.
    Axe des Y (vertical) : Représente la part de marché (%) de l'agence.
    Ligne Bleue : Montre les fluctuations de la part de marché de l'agence au cours du temps, avec des annotations indiquant les pics et les creux.
    Calculs :

    Part de Marché (%) = (Ventes de l'agence / Ventes totales du marché) * 100 par trimestre.
    Recommandation :

    Identifier les Saisons de Succès : Utilisez les pics pour déterminer les périodes où vos stratégies ont été efficaces et planifiez des campagnes similaires pour les trimestres futurs.
    Traiter les Périodes de Faible Performance : Enquêtez sur les raisons des creux et ajustez les stratégies, comme des offres spéciales ou des campagnes publicitaires plus ciblées, pour améliorer la performance pendant ces périodes. """)
else:
    st.write("Aucune donnée disponible pour les Tendances Trimestrielles de la Part de Marché.")


# ----------------------------
# Step 14
# ----------------------------
st.markdown("\n\n")  # Ajouter des sauts de ligne avant le début du chapitre

# Calculs nécessaires

# Total des ventes de l'agence par trimestre
total_sales_quarter_agency = df_agency_filtered.groupby(['year', 'quarter']).agg(
    total_sales_agency=('nombre de ventes', 'sum')
).reset_index()

# Total des ventes du marché par trimestre
total_sales_quarter_market = df_market_filtered.groupby(['year', 'quarter']).agg(
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

# Calcul de la part de marché de l'agence par trimestre
sales_quarter['market_share (%)'] = (sales_quarter['total_sales_agency'] / sales_quarter['total_sales_market']) * 100
sales_quarter['market_share (%)'] = sales_quarter['market_share (%)'].replace([float('inf'), -float('inf')], 0).fillna(0)

# Tri des données à partir du trimestre le plus ancien
sales_quarter = sales_quarter.sort_values('date')

# Création du graphique en utilisant Plotly
fig = go.Figure()

# Ligne pour les ventes totales de l'agence
fig.add_trace(go.Scatter(
    x=sales_quarter['date'],
    y=sales_quarter['total_sales_agency'],
    mode='lines+markers',
    name='Ventes Totales Agence',
    line=dict(color='blue'),
    marker=dict(size=6),
    text=[f"{int(sales)} ventes" for sales in sales_quarter['total_sales_agency']],
    textposition='top center',
    hovertemplate='Trimestre: %{x}<br>Ventes Totales: %{y}'
))

# Ligne pour la part de marché de l'agence
fig.add_trace(go.Scatter(
    x=sales_quarter['date'],
    y=sales_quarter['market_share (%)'],
    mode='lines+markers',
    name='Part de Marché (%)',
    line=dict(color='green'),
    marker=dict(size=6),
    yaxis='y2',  # Utilisation du deuxième axe Y
    text=[f"{share:.2f}%" for share in sales_quarter['market_share (%)']],
    textposition='top center',
    hovertemplate='Trimestre: %{x}<br>Part de Marché: %{y:.2f}%'
))

# Mise à jour du layout du graphique
fig.update_layout(
    title='Performance des Trimestres (Ventes Totales et Part de Marché)',
    xaxis=dict(title='Date', tickformat='%b %Y'),
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
    height=600
)

# Annotations pour les meilleurs et pires trimestres en termes de part de marché
best_quarter = sales_quarter.loc[sales_quarter['market_share (%)'].idxmax()]
worst_quarter = sales_quarter.loc[sales_quarter['market_share (%)'].idxmin()]

fig.add_annotation(
    x=best_quarter['date'],
    y=best_quarter['market_share (%)'],
    text=f"Meilleur Trimestre (Part de Marché) : {best_quarter['market_share (%)']:.2f}%",
    showarrow=True,
    arrowhead=2,
    yshift=10,
    arrowcolor="green",
    font=dict(size=10)
)

fig.add_annotation(
    x=worst_quarter['date'],
    y=worst_quarter['market_share (%)'],
    text=f"Pire Trimestre (Part de Marché) : {worst_quarter['market_share (%)']:.2f}%",
    showarrow=True,
    arrowhead=2,
    yshift=-10,
    arrowcolor="red",
    font=dict(size=10)
)

# Annotations pour les meilleurs trimestres en termes de ventes totales
best_sales_quarter = sales_quarter.loc[sales_quarter['total_sales_agency'].idxmax()]
fig.add_annotation(
    x=best_sales_quarter['date'],
    y=best_sales_quarter['total_sales_agency'],
    text=f"Meilleur Trimestre (Ventes Totales) : {best_sales_quarter['total_sales_agency']} ventes",
    showarrow=True,
    arrowhead=2,
    yshift=10,
    arrowcolor="blue",
    font=dict(size=10)
)

# Affichage du graphique dans Streamlit
st.plotly_chart(fig, use_container_width=True)

# Ajout d'analyse et de recommandations
st.markdown("""
**Analyse :** Ce graphique présente deux perspectives importantes sur la performance des trimestres :
- **Ventes Totales de l'Agence :** La ligne bleue indique le volume total des ventes réalisées par votre agence pour chaque trimestre. Cela permet de voir l'évolution absolue des ventes.
- **Part de Marché :** La ligne verte représente la part de marché de votre agence par rapport aux ventes totales du marché pour chaque trimestre. Cela montre la performance relative de votre agence.

Les annotations mettent en évidence les trimestres avec les meilleures et les pires performances, en termes de part de marché et de ventes totales.

**Comment lire le graphique :**
- **Axe des X (horizontal) :** Représente les trimestres dans l'ordre chronologique.
- **Axe des Y (à gauche) :** Indique les ventes totales de l'agence pour chaque trimestre.
- **Axe des Y (à droite) :** Indique la part de marché en pourcentage.
- **Ligne Bleue :** Indique les ventes totales de l'agence.
- **Ligne Verte :** Indique la part de marché de l'agence par rapport au marché total.

**Recommandation :**
- **Renforcer les Stratégies Réussies :** Identifiez les trimestres où votre agence a obtenu la meilleure part de marché et analysez les actions qui ont été entreprises pour reproduire ce succès.
- **Améliorer les Performances lors des Trimestres Faibles :** Pour les trimestres avec une faible part de marché ou de faibles ventes, envisagez des actions correctives, telles que l'amélioration des offres, des promotions ciblées ou des campagnes marketing intensifiées.
- **Planification des Campagnes :** Les trimestres avec des ventes totales élevées montrent une forte demande. Concentrez vos efforts de vente et de marketing avant ces périodes pour maximiser les résultats.
""")
