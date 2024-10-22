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

# Streamlit configuration
st.set_page_config(
    layout="wide",
    page_title="🏠 Tableau de Bord d'Analyse de la Part de Marché Immobilière",
    page_icon=":house:"
)

# Title of the application
st.title("🏠 Tableau de Bord d'Analyse de la Part de Marché Immobilière")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --------------------------------------
# Step 1: Load the agency data from Streamlit secrets
# --------------------------------------
# Fetching the agency data from Streamlit secrets
data = st.secrets["data"]["my_agency_data"]
df_agency = pd.read_csv(StringIO(data))

# Display the agency data for verification
st.write(df_agency)

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
        st.error("Colonne 'date de transaction' non trouvée dans les données de l'agence.")
        st.stop()

    df_agency['year'] = df_agency['date'].dt.year
    df_agency['quarter'] = df_agency['date'].dt.quarter
    df_agency['month'] = df_agency['date'].dt.month

    # Split 'map' column into 'latitude' and 'longitude'
    if 'map' in df_agency.columns:
        try:
            df_agency[['latitude', 'longitude']] = df_agency['map'].str.strip().str.split(',', expand=True).astype(float)
        except Exception as e:
            st.error(f"Erreur lors du traitement de la colonne 'map' dans les données de l'agence : {e}")
            st.stop()
    else:
        st.error("Colonne 'map' non trouvée dans les données de l'agence.")
        st.stop()

    # Assuming each row is a transaction; set 'nombre de ventes' to 1
    df_agency['nombre de ventes'] = 1

    # Rename 'communes' to 'commune' for consistency
    if 'communes' in df_agency.columns:
        df_agency.rename(columns={'communes': 'commune'}, inplace=True)
    else:
        st.error("Colonne 'communes' non trouvée dans les données de l'agence.")
        st.stop()

    # Drop rows where 'commune' is missing
    if 'commune' in df_agency.columns:
        df_agency.dropna(subset=['commune'], inplace=True)
        df_agency.reset_index(drop=True, inplace=True)
    else:
        st.error("Colonne 'commune' non trouvée dans les données de l'agence.")
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

# Preprocess the data
df_market, df_agency = preprocess_data(df_market, df_agency)

# --------------------------------------
# Now continue with the rest of your analysis and visualization steps...
# --------------------------------------

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
# Step 1: Ventes de l'Agence par Commune et Trimestre
# --------------------------------------
st.header("1. Ventes de l'Agence par Commune et Trimestre")

# Filtrer les communes où l'agence a vendu des biens
sold_communes = df_agency_filtered['commune'].unique()

# Filtrer les données de l'agence pour les ventes dans les communes vendues
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

# Fonction pour appliquer un gradient de vert
def apply_green_gradient(data):
    styles = pd.DataFrame('', index=data.index, columns=data.columns)

    # Normaliser les données pour une couleur entre blanc et vert
    max_value = data.max().max()  # Trouver la valeur max pour normaliser
    for column in data.columns:
        for index in data.index:
            value = data.loc[index, column]
            green_shade = int((value / max_value) * 255)  # Échelle de couleur
            styles.at[index, column] = f'background-color: rgb({255 - green_shade}, 255, {255 - green_shade})'

    return styles

# Appliquer le gradient sur le tableau des pourcentages de ventes
styled_sales_percentage_table = sales_percentage_table.style.apply(apply_green_gradient, axis=None) \
                                                            .format("{:.2f}%")

# Afficher la table dans Streamlit
st.dataframe(styled_sales_percentage_table, use_container_width=True)

## --------------------------------------
# Step 2: Répartition en Pourcentage des Ventes par Commune au Dernier Trimestre
# --------------------------------------
st.header("2. Répartition en Pourcentage des Ventes par Commune au Dernier Trimestre")

# Convertir la colonne 'date' en datetime si nécessaire
df_agency_filtered['date'] = pd.to_datetime(df_agency_filtered['date'])

# Extraire l'année et le trimestre de la date
df_agency_filtered['year'] = df_agency_filtered['date'].dt.year
df_agency_filtered['quarter'] = df_agency_filtered['date'].dt.quarter

# Obtenir l'année et le trimestre les plus récents dans les données
latest_year = df_agency_filtered['year'].max()
latest_quarter = df_agency_filtered[df_agency_filtered['year'] == latest_year]['quarter'].max()

# Filtrer les données pour le dernier trimestre
df_last_quarter = df_agency_filtered[
    (df_agency_filtered['year'] == latest_year) &
    (df_agency_filtered['quarter'] == latest_quarter)
]

# Vérifier si des données sont disponibles pour le dernier trimestre
if df_last_quarter.empty:
    st.warning("Aucune donnée disponible pour le dernier trimestre.")
else:
    # Agréger les ventes totales de l'agence par commune pour le dernier trimestre
    sales_by_commune = df_last_quarter.groupby('commune').agg(
        total_sales=('nombre de ventes', 'sum')
    ).reset_index()

    # Calculer le total des ventes
    total_sales = sales_by_commune['total_sales'].sum()

    # Calculer le pourcentage de ventes par commune
    sales_by_commune['sales_percentage'] = (sales_by_commune['total_sales'] / total_sales) * 100

    # Créer le graphique en camembert
    fig_pie = px.pie(
        sales_by_commune,
        values='sales_percentage',
        names='commune',
        title=f"Répartition en Pourcentage des Ventes par Commune - {latest_year} T{latest_quarter}",
        hover_data=['total_sales'],
        labels={'total_sales': 'Nombre de Ventes', 'sales_percentage': 'Pourcentage des Ventes'}
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig_pie, use_container_width=True)

    # Analyse dynamique des données
    st.markdown("### Analyse et Implications Stratégiques")

    # Identifier les communes avec les plus fortes contributions
    top_communes = sales_by_commune.sort_values('sales_percentage', ascending=False).head(3)
    top_communes_list = top_communes['commune'].tolist()

    # Générer une analyse dynamique
    if top_communes_list:
        st.markdown(f"Les communes avec les plus fortes contributions aux ventes du dernier trimestre sont : **{', '.join(top_communes_list)}**. Cela indique que ces zones sont particulièrement actives et pourraient offrir des opportunités pour renforcer davantage notre présence.")

    # Recommandations stratégiques
    st.markdown("### Recommandations Stratégiques")

    st.markdown("""
    - **Capitaliser sur les Communes Performantes** : Continuer à investir dans les communes les plus performantes pour consolider notre position dominante.
    - **Explorer les Opportunités dans les Communes Sous-Représentées** : Pour les communes avec une faible part de ventes, envisager des stratégies pour augmenter la visibilité et attirer de nouveaux clients.
    - **Analyser les Facteurs de Succès** : Comprendre ce qui fonctionne dans les communes performantes (marketing, relations clients, offres spéciales) et appliquer ces stratégies à d'autres zones.
    """)

    # Transparence pour solliciter des retours des parties prenantes
    st.markdown("### Transparence : Retour et Discussions")

    st.markdown(f"""
    Ce camembert nous permet de visualiser la répartition des ventes par commune pour le dernier trimestre ({latest_year} T{latest_quarter}). Il est crucial de comprendre pourquoi certaines communes performent mieux que d'autres.

    - **Questions à considérer** :
        - Les communes les plus performantes bénéficient-elles de campagnes marketing spécifiques ?
        - Y a-t-il des facteurs externes (économiques, démographiques) qui influencent ces résultats ?
        - Comment pouvons-nous reproduire le succès de ces communes dans d'autres zones ?

    Vos retours et observations sont essentiels pour ajuster nos stratégies et maximiser notre performance sur l'ensemble du territoire.
    """)


# --------------------------------------
# Section 14: Évolution de la Part de Marché par Commune
# --------------------------------------


st.header("14. Évolution de la Part de Marché par Commune")

# Question stratégique
st.markdown("""
### Question Clé : **Comment nos ventes évoluent-t-elles dans chaque commune par rapport au marché au fil du temps ?**
Cette analyse vise à comprendre **dans quelle mesure** nos ventes et notre part de marché changent dans chaque commune, afin de guider nos décisions stratégiques.
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

# Étape 2: Sélectionner les Top N communes
top_n = st.slider("Sélectionner le nombre de communes à afficher", min_value=1, max_value=10, value=6)
top_communes = market_share_filtered.groupby('commune')['nombre de ventes_agency'].sum().sort_values(ascending=False).head(top_n).index.tolist()

st.write(f"**Communes sélectionnées :** {', '.join(top_communes)}")

filtered_sales = market_share_filtered[market_share_filtered['commune'].isin(top_communes)].copy()

if filtered_sales.empty:
    st.warning("Aucune donnée disponible pour les communes sélectionnées.")
    st.stop()

# Étape 3: Préparer les données pour les graphiques
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

for commune in top_communes:
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
    height=250 * (top_n // 2 + top_n % 2)  # Ajuster la hauteur en fonction du nombre de facettes
)

fig_line.update_layout(
    hovermode='x unified',
    legend_title='Type de Ventes',
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig_line, use_container_width=True)


# Étape 7: Analyse et Recommandations
st.markdown("""
### Analyse et Implications Stratégiques :
- **Performance Supérieure au Marché** : Indique où nos stratégies fonctionnent bien.
- **Performance Inférieure au Marché** : Signale des domaines nécessitant une attention ou des ajustements.
- **Performance Alignée avec le Marché** : Représente des zones de stabilité et des opportunités potentielles de croissance.
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

# Recommandations stratégiques
st.markdown("### Recommandations Stratégiques :")

recommendations_outperforming = [
    "🔝 **Capitaliser sur le Succès** : Continuez à investir dans la commune {commune} pour maintenir et renforcer notre position dominante.",
    "🚀 **Renforcer les Stratégies Gagnantes** : Poursuivez les initiatives actuelles dans la commune {commune} qui montrent une forte croissance.",
    "🌟 **Maximiser les Opportunités** : Exploitez la dynamique positive dans la commune {commune} en augmentant notre présence et nos actions marketing.",
    "📈 **Soutenir la Croissance** : Maintenez et développez nos efforts dans la commune {commune} pour capitaliser sur cette tendance ascendante.",
    "💼 **Renforcer l'Engagement Local** : Intensifiez nos actions dans la commune {commune} pour consolider notre succès et attirer davantage de clients."
]

recommendations_underperforming = [
    "⚠️ **Réévaluer les Stratégies** : Analysez les causes de la sous-performance dans la commune {commune} et ajustez nos approches marketing et commerciales.",
    "🔍 **Investiguer les Défis** : Identifiez les obstacles dans la commune {commune} et développez des plans d'action ciblés pour inverser la tendance.",
    "📉 **Stimuler la Croissance** : Implémentez des initiatives spécifiques dans la commune {commune} pour améliorer notre part de marché.",
    "🛠️ **Adapter les Approches** : Révisez nos stratégies dans la commune {commune} pour mieux répondre aux besoins et attentes du marché local.",
    "📊 **Optimiser les Efforts** : Renforcez notre présence et nos campagnes dans la commune {commune} pour remédier à la baisse observée."
]

recommendations_matching = [
    "🔄 **Maintenir la Stabilité** : Continuez les efforts actuels dans la commune {commune} pour conserver notre position alignée avec le marché.",
    "🛠️ **Optimiser les Processus** : Identifiez des moyens d'améliorer encore nos opérations dans la commune {commune} pour stimuler une croissance future.",
    "💡 **Explorer de Nouvelles Opportunités** : Cherchez des initiatives innovantes dans la commune {commune} pour dépasser les performances du marché.",
    "📈 **Stimuler la Croissance** : Introduisez de nouvelles stratégies dans la commune {commune} pour dynamiser davantage notre part de marché.",
    "🌱 **Développer de Nouvelles Initiatives** : Lancez des projets spécifiques dans la commune {commune} pour encourager une croissance continue."
]

def add_recommendations(communes, recommendations_list):
    for commune in communes:
        recommendation = np.random.choice(recommendations_list).format(commune=commune)
        st.markdown(f"- {recommendation}")

add_recommendations(outperforming_communes, recommendations_outperforming)
add_recommendations(underperforming_communes, recommendations_underperforming)
add_recommendations(matching_communes, recommendations_matching)

# Transparence et Sollicitation de Retours
st.markdown("""
### Transparence : Retour et Discussions
Cette section analyse l'évolution de notre part de marché par commune par rapport au marché au fil du temps. Les tendances observées indiquent où nous excellons et où des améliorations sont nécessaires.

- **Questions à considérer** :
    - Dans les communes où nous surperformons, quelles stratégies spécifiques ont conduit à cette croissance ?
    - Pour les communes en sous-performance, quels facteurs externes ou internes pourraient être responsables ?
    - Comment pouvons-nous adapter nos stratégies pour mieux aligner notre croissance avec celle du marché ?

**Action Demandée** : Veuillez fournir vos retours et observations pour affiner nos stratégies et maximiser notre impact sur le marché.
""")


# --------------------------------------
# Step 0: Visualizations with Unique Keys
# --------------------------------------

# Function to generate unique keys
def generate_key(*args):
    return "_".join(map(str, args))


# --------------------------------------
# Step 2. Part de Marché dans le Temps par Commune (Graphique en Lignes)
# --------------------------------------
st.header("2. Part de Marché dans le Temps par Commune")

# Question stratégique pour guider cette section
st.markdown("### Question Clé : Comment la performance de notre agence évolue-t-elle dans les principales communes au fil du temps ?")

# Déterminer la date où l'agence a réalisé sa première vente
first_sale_date = df_agency['date'].min()

# Filtrer les données à partir de la date de la première vente de l'agence
market_share_filtered = market_share[market_share['date'] >= first_sale_date]

# Sélection du nombre de communes à afficher
top_n = st.slider("Sélectionner les N meilleures Communes à afficher", min_value=1, max_value=20, value=5)
top_communes_over_time = market_share_filtered.groupby('commune')['market_share (%)'].mean().sort_values(ascending=False).head(top_n).index.tolist()

# Filtrer les données pour les communes sélectionnées
market_share_top_n = market_share_filtered[market_share_filtered['commune'].isin(top_communes_over_time)]

# Création du graphique en lignes pour la part de marché dans le temps
fig_market_share_time = px.line(
    market_share_top_n,
    x='date',
    y='market_share (%)',
    color='commune',
    markers=True,
    title='Évolution de la Part de Marché dans les Communes Principales (Depuis la Première Vente de l’Agence)',
    labels={'date': 'Date', 'market_share (%)': 'Part de Marché (%)', 'commune': 'Commune'}
)

# Ajouter des annotations pour les pics significatifs
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

# Affichage du graphique dans Streamlit
st.plotly_chart(fig_market_share_time, use_container_width=True, key=generate_key('step6', 'market_share_over_time', 'line'))

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
for commune in top_communes_over_time:
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

# Recommandations stratégiques pour pousser à l'action
st.markdown("""### Recommandations Stratégiques :
- **Capitaliser sur la Croissance** : Pour les communes en croissance, continuez à investir et à renforcer les stratégies qui portent leurs fruits.
- **Réagir aux Baisses** : Pour les communes en baisse, identifiez les causes possibles et envisagez des actions ciblées pour inverser la tendance.
- **Exploiter la Stabilité** : Pour les communes stables, explorez des opportunités pour stimuler la croissance ou consolider votre position.
""")

# Transparence pour solliciter des retours des parties prenantes
feedback_text = "### Transparence : Retour et Discussions\n"
feedback_text += "Cet aperçu montre l'évolution de la part de marché de notre agence dans les principales communes. "

if increasing_communes:
    communes_list = ', '.join([f"**{c}**" for c in increasing_communes])
    feedback_text += f"Des communes comme {communes_list} affichent une progression notable – devrions-nous continuer à investir ici ou diversifier nos efforts ? "

if decreasing_communes:
    communes_list = ', '.join([f"**{c}**" for c in decreasing_communes])
    feedback_text += f"Les baisses à {communes_list} sont préoccupantes — devrions-nous explorer plus en détail ce qui se passe et adapter nos stratégies ? "

if stable_communes:
    communes_list = ', '.join([f"**{c}**" for c in stable_communes])
    feedback_text += f"Les communes {communes_list} montrent une stabilité – existe-t-il des opportunités pour stimuler davantage la croissance ? "

feedback_text += "\nVos retours sont essentiels pour ajuster nos prochaines actions."

st.markdown(feedback_text)


# --------------------------------------
# Step 3: Part de Marché Trimestrielle par Commune
# --------------------------------------
st.header("3. Part de Marché Trimestrielle par Commune")

# Question commerciale guidant la section
st.markdown("### Question Clé : Quels trimestres et quelles communes affichent les meilleures performances en termes de part de marché, et comment pouvons-nous exploiter ces informations pour renforcer notre position sur le marché ?")

# Calculer la part de marché moyenne par commune et trimestre
market_share_quarterly = market_share.groupby(['commune', 'year', 'quarter']).agg({'market_share (%)': 'mean'}).reset_index()
market_share_quarterly['date'] = pd.to_datetime(market_share_quarterly['year'].astype(str) + '-Q' + market_share_quarterly['quarter'].astype(str))

# Identifier les trimestres les plus récents
latest_quarters = market_share_quarterly['date'].drop_duplicates().sort_values(ascending=False).head(4)

# Filtrer les données pour les 4 derniers trimestres
recent_market_share = market_share_quarterly[market_share_quarterly['date'].isin(latest_quarters)]

# Calculer la part de marché moyenne sur les 4 derniers trimestres par commune
average_market_share = recent_market_share.groupby('commune').agg({'market_share (%)': 'mean'}).reset_index()
average_market_share = average_market_share.sort_values('market_share (%)', ascending=False)

# Identifier les communes avec la meilleure part de marché moyenne
top_communes = average_market_share.head(7)['commune'].tolist()

# Filtrer les données pour les communes les plus performantes
top_communes_data = recent_market_share[recent_market_share['commune'].isin(top_communes)]

# Créer des étiquettes pour les trimestres
top_communes_data['quarter_label'] = top_communes_data['year'].astype(str) + " T" + top_communes_data['quarter'].astype(str)

# Simplifier la visualisation : Afficher un graphique compact
fig_market_share = px.bar(
    top_communes_data,
    x='commune',
    y='market_share (%)',
    color='quarter_label',
    barmode='group',
    title='Part de Marché des Top Communes sur les 4 Derniers Trimestres',
    labels={'market_share (%)': 'Part de Marché (%)', 'commune': 'Commune', 'quarter_label': 'Trimestre'},
    color_discrete_sequence=px.colors.qualitative.Pastel
)

fig_market_share.update_layout(
    xaxis_title='Commune',
    yaxis_title='Part de Marché (%)',
    legend_title='Trimestre',
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig_market_share, use_container_width=True)

# Analyse des Tendances et Calculs
st.subheader("Analyse des Performances des Communes les Plus Performantes")

analysis_data = []

for commune in top_communes:
    commune_data = top_communes_data[top_communes_data['commune'] == commune].copy()
    avg_market_share = commune_data['market_share (%)'].mean()
    latest_market_share = commune_data.iloc[-1]['market_share (%)']
    
    # Calcul du changement de part de marché entre le dernier et l'avant-dernier trimestre
    if len(commune_data) >= 2:
        previous_market_share = commune_data.iloc[-2]['market_share (%)']
        market_share_change = latest_market_share - previous_market_share
    else:
        market_share_change = None
    
    # Déterminer la tendance
    if market_share_change is not None:
        if market_share_change > 0:
            trend = "Hausse"
            symbol = "⬆️"
            # Phrases pour tendance positive
            phrases = [
                f"La part de marché à {commune} a augmenté de {market_share_change:+.2f}% au dernier trimestre, atteignant {latest_market_share:.2f}%.",
                f"Nous observons une croissance de {market_share_change:+.2f}% de la part de marché à {commune}, maintenant à {latest_market_share:.2f}%.",
                f"Amélioration notable à {commune} avec une part de marché en hausse de {market_share_change:+.2f}% pour atteindre {latest_market_share:.2f}%."
            ]
        elif market_share_change < 0:
            trend = "Baisse"
            symbol = "⬇️"
            # Phrases pour tendance négative
            phrases = [
                f"La part de marché à {commune} a diminué de {market_share_change:+.2f}%, tombant à {latest_market_share:.2f}%.",
                f"Réduction de notre part de marché à {commune} de {market_share_change:+.2f}%, désormais à {latest_market_share:.2f}%.",
                f"Déclin observé à {commune} avec une baisse de {market_share_change:+.2f}% de notre part de marché, atteignant {latest_market_share:.2f}%."
            ]
        else:
            trend = "Stable"
            symbol = "➡️"
            # Phrases pour tendance neutre
            phrases = [
                f"La part de marché à {commune} est restée stable à {latest_market_share:.2f}%.",
                f"Aucune variation notable de la part de marché à {commune}, maintenue à {latest_market_share:.2f}%.",
                f"La part de marché à {commune} demeure inchangée à {latest_market_share:.2f}%."
            ]
        # Sélectionner une phrase aléatoire pour varier les recommandations
        phrase = random.choice(phrases)
    else:
        trend = "N/A"
        symbol = "❓"
        phrase = f"Données insuffisantes pour déterminer la tendance récente à {commune}."
    
    analysis_data.append({
        'Commune': commune,
        'Dernière Part de Marché (%)': f"{latest_market_share:.2f}",
        'Changement Récent (%)': f"{market_share_change:+.2f}%" if market_share_change is not None else "N/A",
        'Tendance': f"{symbol} {trend}",
        'Analyse et Recommandation': phrase
    })

# Convertir les données d'analyse en DataFrame
analysis_df = pd.DataFrame(analysis_data)

# Afficher la table d'analyse compacte
st.table(analysis_df)

# --------------------------------------
# Step 4. Pourcentage du Volume de Ventes Trimestriel par Commune (Une Couleur par Trimestre)
# --------------------------------------

st.header("4. Pourcentage du Volume de Ventes Trimestriel par Commune (Une Couleur par Trimestre)")

# Calculer le volume total des ventes par commune et par trimestre
sales_volume_quarterly = market_share.groupby(['commune', 'year', 'quarter']).agg({'nombre de ventes_agency': 'sum'}).reset_index()
sales_volume_quarterly['date'] = pd.to_datetime(sales_volume_quarterly['year'].astype(str) + '-Q' + sales_volume_quarterly['quarter'].astype(str))

# Calculer le volume total des ventes pour toutes les communes pour chaque trimestre
total_sales_per_quarter = sales_volume_quarterly.groupby(['year', 'quarter'])['nombre de ventes_agency'].sum().reset_index()
total_sales_per_quarter['date'] = pd.to_datetime(total_sales_per_quarter['year'].astype(str) + '-Q' + sales_volume_quarterly['quarter'].astype(str))

# Fusionner pour calculer la contribution en pourcentage de chaque commune
sales_volume_quarterly = sales_volume_quarterly.merge(total_sales_per_quarter, on=['year', 'quarter', 'date'], suffixes=('', '_total'))
sales_volume_quarterly['sales_volume_percentage'] = (sales_volume_quarterly['nombre de ventes_agency'] / sales_volume_quarterly['nombre de ventes_agency_total']) * 100

# Trier par pourcentage de volume de ventes pour mettre en évidence les trimestres les plus performants
sales_volume_quarterly_sorted = sales_volume_quarterly.sort_values('sales_volume_percentage', ascending=False).head(20)

# Créer des étiquettes uniques pour chaque combinaison de commune et de trimestre
sales_volume_quarterly_sorted['label'] = sales_volume_quarterly_sorted.apply(
    lambda row: f"{row['commune']} - {row['date'].strftime('%b %Y')}", axis=1
)

# Attribuer une couleur unique à chaque trimestre
sales_volume_quarterly_sorted['quarter_label'] = sales_volume_quarterly_sorted['year'].astype(str) + " T" + sales_volume_quarterly_sorted['quarter'].astype(str)

# Identifier les deux derniers trimestres
latest_dates = sales_volume_quarterly['date'].drop_duplicates().sort_values(ascending=False).head(2)

# Mettre en évidence les deux derniers trimestres en mettant à jour leurs étiquettes sur l'axe des x
highlighted_labels = [
    f"<b style='font-size:16px'>{label}</b>" if row.date in latest_dates.values else label
    for label, row in zip(sales_volume_quarterly_sorted['label'], sales_volume_quarterly_sorted.itertuples())
]

# Tracer : Séparer chaque trimestre et chaque commune
fig_top20_quarters = px.bar(
    sales_volume_quarterly_sorted,
    x='label',
    y='sales_volume_percentage',
    color='quarter_label',  # Couleur par trimestre
    title='Pourcentage du Volume de Ventes Trimestriel par Commune (Top 20 par Trimestre et Commune)',
    labels={'sales_volume_percentage': 'Volume des Ventes (%)', 'label': 'Commune - Trimestre', 'quarter_label': 'Trimestre'},
    color_discrete_sequence=px.colors.qualitative.Pastel
)

# Mettre à jour les étiquettes de l'axe x pour mettre en évidence les deux derniers trimestres
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
                "Chaque barre représente un trimestre spécifique pour une commune particulière.<br>"
                "Cette visualisation aide à identifier les trimestres les plus performants pour chaque commune,<br>"
                "chaque trimestre étant représenté par une couleur distincte."
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

# Afficher le graphique
st.plotly_chart(fig_top20_quarters, use_container_width=True, key=generate_key('step7bis', 'separated_quarters_communes_sales_volume', 'highlight'))

# Amélioration des analyses et recommandations pour s'adapter à tout résultat
st.markdown("""
**Analyse :**

Ce graphique met en évidence les 20 meilleures contributions en pourcentage du volume de ventes trimestriel par commune. En observant les communes qui apparaissent dans ce top 20, il est possible d'identifier les zones géographiques et les périodes qui ont le plus contribué au volume de ventes total.

**Comment lire le graphique :**

- **Axe des X (horizontal) :** Chaque étiquette correspond à une commune pour un trimestre donné.
- **Axe des Y (vertical) :** Indique le pourcentage du volume total de ventes que représente chaque commune pour le trimestre considéré.
- **Couleurs par trimestre :** Chaque trimestre est représenté par une couleur unique, ce qui facilite la comparaison entre les périodes.

**Recommandations :**

- **Identifier les Opportunités de Croissance :** Repérez les communes qui ont une forte contribution au volume de ventes pour déterminer où concentrer vos efforts futurs.
- **Analyser les Performances Saisonnières :** Observez les variations de performance d'une commune à l'autre selon les trimestres pour adapter vos stratégies marketing et commerciales en conséquence.
- **Allouer les Ressources Efficacement :** Utilisez ces informations pour décider où investir en termes de personnel, de budget marketing et d'autres ressources afin de maximiser le rendement.
- **Planifier les Actions Marketing :** Développez des campagnes ciblées pour les trimestres et les communes qui montrent un potentiel élevé, afin de capitaliser sur les tendances positives.
""")

# --------------------------------------
# Section 5: Volume des Ventes vs Part de Marché par Commune
# --------------------------------------


st.markdown("\n\n")  # Ajouter des sauts de ligne avant le début du chapitre
st.header("5. Volume des Ventes vs Part de Marché par Commune")

# Question commerciale guidant la section
st.markdown("""
### Question Clé : **Comment le volume total des ventes du marché se compare-t-il à nos ventes par commune, et où devrions-nous concentrer nos efforts pour augmenter notre part de marché ?**
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

# Étape 2: Sélectionner les Top N communes
top_n = st.slider("Sélectionner le nombre de communes à afficher", min_value=1, max_value=20, value=20)
top_communes = market_share_filtered.groupby('commune')['nombre de ventes_agency'].sum().sort_values(ascending=False).head(top_n).index.tolist()

st.write(f"**Communes sélectionnées :** {', '.join(top_communes)}")

filtered_sales = market_share_filtered[market_share_filtered['commune'].isin(top_communes)].copy()

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

# Graphique à bulles affichant toutes les communes
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
        'Haute Part de Marché': 'green',
        'Faible Part de Marché': 'red'
    }
)

fig_bubble.update_layout(
    xaxis_title='Ventes Totales du Marché',
    yaxis_title='Ventes de l\'Agence',
    legend_title='Performance',
    hovermode='closest',
    height=500,  # Réduire la hauteur pour une meilleure compacité
    margin=dict(l=50, r=50, t=50, b=50)  # Réduire les marges pour économiser de l'espace
)

st.plotly_chart(fig_bubble, use_container_width=True)

# Analyse des Communes Clés
st.subheader("Analyse des Communes Clés")

# Sélectionner les top 15 communes par part de marché (ou ajuster si top_n < 15)
top_15_communes = sales_vs_market_share.head(20).copy()

# Présenter les analyses sous forme de tableau sans recommandations
analysis_data = []

for index, row in top_15_communes.iterrows():
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

# Recommandations Stratégiques
st.subheader("Recommandations Stratégiques")

# Générer les recommandations basées sur l'analyse des données
recommendations_high = top_15_communes[top_15_communes['Performance'] == 'Haute Part de Marché']
recommendations_low = top_15_communes[top_15_communes['Performance'] == 'Faible Part de Marché']

# Générer le texte des recommandations
recommendations_text = "En nous basant sur l'analyse ci-dessus, voici les recommandations pour les communes clés :\n\n"

# Communes avec Haute Part de Marché
if not recommendations_high.empty:
    recommendations_text += "- **Communes avec Haute Part de Marché :**\n"
    for index, row in recommendations_high.iterrows():
        commune = row['commune']
        part_de_marche = row['market_share (%)']
        recommendations = [
            f"**{commune}** : Maintenez nos efforts pour conserver notre position dominante avec une part de marché de **{part_de_marche:.2f}%**.",
            f"**{commune}** : Continuez à renforcer notre présence pour maintenir une part de marché élevée de **{part_de_marche:.2f}%**.",
            f"**{commune}** : Consolidez nos stratégies actuelles pour préserver une part de marché solide de **{part_de_marche:.2f}%**."
        ]
        recommendation = random.choice(recommendations)  # Sélectionner une phrase aléatoire pour varier les recommandations
        recommendations_text += f"  - {recommendation}\n"

# Communes avec Faible Part de Marché
if not recommendations_low.empty:
    recommendations_text += "\n- **Communes avec Faible Part de Marché :**\n"
    for index, row in recommendations_low.iterrows():
        commune = row['commune']
        part_de_marche = row['market_share (%)']
        recommendations = [
            f"**{commune}** : Opportunité à saisir pour augmenter notre part de marché actuellement à **{part_de_marche:.2f}%**.",
            f"**{commune}** : Potentiel de croissance avec une part de marché de **{part_de_marche:.2f}%** - renforcez nos efforts commerciaux.",
            f"**{commune}** : Augmentez notre présence marketing pour capter une plus grande part de marché de **{part_de_marche:.2f}%**."
        ]
        recommendation = random.choice(recommendations)  # Sélectionner une phrase aléatoire pour varier les recommandations
        recommendations_text += f"  - {recommendation}\n"

# Ajouter une note sur la période couverte
recommendations_text += "\n_**Note** : Cette analyse couvre l'ensemble des périodes durant lesquelles nous avons été actifs. Elle offre une vue globale de nos performances par commune depuis le début de nos activités jusqu'à la dernière période de vente._"

# Afficher les recommandations
st.markdown(recommendations_text)

# Transparence et Sollicitation de Retours
st.markdown("""
### Transparence : Retour et Discussions
Cette section analyse l'évolution de notre part de marché par commune par rapport au marché au fil du temps. Les tendances observées indiquent où nous excellons et où des améliorations sont nécessaires.

- **Questions à considérer** :
    - Dans les communes où nous surperformons, quelles stratégies spécifiques ont conduit à cette croissance ?
    - Pour les communes en sous-performance, quels facteurs externes ou internes pourraient être responsables ?
    - Comment pouvons-nous adapter nos stratégies pour mieux aligner notre croissance avec celle du marché ?

**Action Demandée** : Veuillez fournir vos retours et observations pour affiner nos stratégies et maximiser notre impact sur le marché.
""")

# --------------------------------------
# Step 6: Part de Marché par Commune au Fil du Temps
# --------------------------------------
st.markdown("\n\n")  # Ajouter des sauts de ligne avant le début du chapitre
st.header("6. Part de Marché par Commune au Fil du Temps")

# Question commerciale guidant la section
st.markdown("### Question Clé : Comment la part de marché de chaque commune a-t-elle évolué au fil du temps, et quelles stratégies pouvons-nous adopter pour optimiser notre position ?")

# Calculer la part de marché moyenne par commune
avg_market_share_commune = market_share.groupby('commune').agg({'market_share (%)': 'mean'}).reset_index()

# Obtenir les top 15 communes par part de marché moyenne
top15_communes_over_time = avg_market_share_commune.sort_values('market_share (%)', ascending=False).head(15)['commune'].tolist()

# Filtrer les données de market_share pour les top 15 communes
market_share_top15 = market_share[market_share['commune'].isin(top15_communes_over_time)]

# Déterminer la première date où l'agence a vendu un bien
first_sale_date = df_agency['date'].min()

# Filtrer les données de market_share à partir de la première date de vente
market_share_top15_filtered = market_share_top15[market_share_top15['date'] >= first_sale_date]

# Calculer la part de marché totale moyenne pour chaque date
total_market_share = market_share_top15_filtered.groupby('date')['market_share (%)'].mean().reset_index()

# Créer le graphique combiné : Barres pour les communes et ligne pour la part de marché totale
fig_combined = go.Figure()

# Ajouter les barres pour chaque commune
for commune in market_share_top15_filtered['commune'].unique():
    commune_data = market_share_top15_filtered[market_share_top15_filtered['commune'] == commune]
    fig_combined.add_trace(
        go.Bar(
            x=commune_data['date'],
            y=commune_data['market_share (%)'],
            name=commune,
            hoverinfo='x+y+name'
        )
    )

# Ajouter la ligne pour la part de marché moyenne
fig_combined.add_trace(
    go.Scatter(
        x=total_market_share['date'],
        y=total_market_share['market_share (%)'],
        mode='lines+markers',
        name='Part de Marché Moyenne (%)',
        line=dict(color='black', width=3),
        marker=dict(size=6)
    )
)

# Mettre à jour la mise en page du graphique
fig_combined.update_layout(
    barmode='group',
    title='Part de Marché par Commune au Fil du Temps (Top 15 Communes et Part de Marché Moyenne)',
    xaxis=dict(title='Date', tickformat='%b %Y'),
    yaxis=dict(title='Part de Marché (%)', tickformat=',.2f', range=[0, 100]),
    hovermode='x unified',
    height=600,
    legend_title_text='Communes',
    margin=dict(l=40, r=40, t=60, b=40),
)

# Ajouter des annotations pour les pics de part de marché moyenne
max_total_share = total_market_share['market_share (%)'].max()
max_total_date = total_market_share[total_market_share['market_share (%)'] == max_total_share]['date'].iloc[0]

fig_combined.add_annotation(
    x=max_total_date,
    y=max_total_share,
    text="Pic de Part de Marché Moyenne",
    showarrow=True,
    arrowhead=2,
    font=dict(size=10),
    arrowcolor="green"
)

# Afficher le graphique
st.plotly_chart(fig_combined, use_container_width=True, key='step6_market_share_grouped')

# Analyse des Communes Clés
st.subheader("Analyse des Communes Clés")

# Sélectionner les top 15 communes par part de marché
top_15_communes = market_share_top15_filtered.groupby('commune').agg({
    'market_share (%)': 'mean'
}).reset_index().sort_values('market_share (%)', ascending=False).head(15)

# Présenter les analyses sous forme de tableau compact
analysis_data = []

for index, row in top_15_communes.iterrows():
    commune = row['commune']
    avg_part_de_marche = row['market_share (%)']
    
    analysis_data.append({
        'Commune': commune,
        'Part de Marché Moyenne (%)': f"{avg_part_de_marche:.2f}"
    })

# Convertir les données d'analyse en DataFrame
analysis_df = pd.DataFrame(analysis_data)

# Afficher la table d'analyse
st.table(analysis_df)

# Recommandations Stratégiques
st.subheader("Recommandations Stratégiques")

# Générer les recommandations basées sur l'analyse des données
recommendations_high = top_15_communes[top_15_communes['market_share (%)'] >= top_15_communes['market_share (%)'].mean()]
recommendations_low = top_15_communes[top_15_communes['market_share (%)'] < top_15_communes['market_share (%)'].mean()]

# Générer le texte des recommandations
recommendations_text = "En nous basant sur l'analyse ci-dessus, voici les recommandations pour les communes clés :\n\n"

# Communes avec Haute Part de Marché
if not recommendations_high.empty:
    recommendations_text += "- **Communes avec Haute Part de Marché :**\n"
    for index, row in recommendations_high.iterrows():
        commune = row['commune']
        part_de_marche = row['market_share (%)']
        recommendations = [
            f"**{commune}** : Maintenir nos efforts pour conserver notre position dominante avec une part de marché de **{part_de_marche:.2f}%**.",
            f"**{commune}** : Continuer à renforcer notre présence pour maintenir une part de marché élevée de **{part_de_marche:.2f}%**.",
            f"**{commune}** : Consolider nos stratégies actuelles pour préserver une part de marché solide de **{part_de_marche:.2f}%**."
        ]
        recommendation = random.choice(recommendations)  # Sélectionner une phrase aléatoire pour varier les recommandations
        recommendations_text += f"  - {recommendation}\n"

# Communes avec Faible Part de Marché
if not recommendations_low.empty:
    recommendations_text += "\n- **Communes avec Faible Part de Marché :**\n"
    for index, row in recommendations_low.iterrows():
        commune = row['commune']
        part_de_marche = row['market_share (%)']
        recommendations = [
            f"**{commune}** : Opportunité à saisir pour augmenter notre part de marché actuellement à **{part_de_marche:.2f}%**.",
            f"**{commune}** : Potentiel de croissance avec une part de marché de **{part_de_marche:.2f}%** - renforcer nos efforts commerciaux.",
            f"**{commune}** : Augmenter notre présence marketing pour capter une plus grande part de marché de **{part_de_marche:.2f}%**."
        ]
        recommendation = random.choice(recommendations)  # Sélectionner une phrase aléatoire pour varier les recommandations
        recommendations_text += f"  - {recommendation}\n"

# Ajouter une note sur la période couverte
recommendations_text += "\n_**Note** : Cette analyse couvre l'ensemble des périodes durant lesquelles nous avons été actifs. Elle offre une vue globale de nos performances par commune depuis le début de nos activités._"

# Afficher les recommandations
st.markdown(recommendations_text)

# --------------------------------------
# Step 7: Carte de Chaleur du Volume de Ventes par Commune (Folium Map avec Détails au Survol)
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
        sales_heatmap = folium.Map(location=map_center, zoom_start=9, tiles='CartoDB positron')
        
        # Préparer les données pour la HeatMap
        heat_data = sales_by_commune[['latitude', 'longitude', 'nombre de ventes']].values.tolist()
        
        # Ajouter la couche heatmap pour les données de vente
        HeatMap(
            data=heat_data,
            radius=25,  # Rayon augmenté pour un mélange plus fluide
            blur=15,    # Flou augmenté pour une répartition plus esthétique
            max_zoom=12
        ).add_to(sales_heatmap)
        
        # Ajouter des popups avec les détails au survol
        for _, row in sales_by_commune.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=f"Commune: {row['commune']}<br>Ventes: {row['nombre de ventes']}",
                color='green',
                fill=True,
                fill_color='green'
            ).add_to(sales_heatmap)
        
        # Rendre la carte Folium dans Streamlit
        folium_static(sales_heatmap, width=700, height=500)
        
    else:
        st.warning("Aucune donnée de vente agrégée disponible pour tracer la carte de chaleur.")
else:
    st.error("Données insuffisantes pour générer la carte de chaleur du volume de ventes. Veuillez vérifier que les colonnes 'latitude', 'longitude', 'nombre de ventes' et 'commune' sont présentes et contiennent des données valides.")

# --------------------------------------
# Step 8: Communes les Plus Performantes par Part de Marché (Bar Chart)
# --------------------------------------
st.markdown("\n\n")  # Ajouter des sauts de ligne avant le début du chapitre

st.header("8. Communes les Plus Performantes par Part de Marché")

# Vérifier si 'market_share_filtered' est défini et contient les colonnes nécessaires
if 'market_share_filtered' not in locals():
    st.error("Le DataFrame 'market_share_filtered' n'est pas défini. Veuillez vous assurer que les Steps 5 et 6 sont exécutés correctement.")
    st.stop()

required_columns = {'commune', 'market_share (%)'}
if not required_columns.issubset(market_share_filtered.columns):
    st.error(f"Le DataFrame 'market_share_filtered' doit contenir les colonnes suivantes : {required_columns}")
    st.stop()

# Sélectionner les Top N communes par part de marché moyenne
default_top_n = 10  # Valeur par défaut pour le top N
top_n_step8 = st.slider(
    "Sélectionner le nombre de communes à afficher",
    min_value=1,
    max_value=20,
    value=default_top_n,
    key='step8_top_n'
)

# Agréger les données pour calculer la part de marché moyenne par commune
avg_market_share_commune_step8 = market_share_filtered.groupby('commune').agg({
    'market_share (%)': 'mean'
}).reset_index()

# Trier les communes par part de marché moyenne décroissante et sélectionner le top N
top_communes_step8 = avg_market_share_commune_step8.sort_values('market_share (%)', ascending=False).head(top_n_step8)

# Créer le graphique à barres pour les communes les plus performantes
fig_top_communes_step8 = px.bar(
    top_communes_step8,
    x='commune',
    y='market_share (%)',
    title=f'Top {top_n_step8} Communes les Plus Performantes par Part de Marché',
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
    margin=dict(l=50, r=50, t=50, b=50)
)

# Afficher les valeurs de part de marché sur les barres
fig_top_communes_step8.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

# Afficher le graphique dans Streamlit
st.plotly_chart(fig_top_communes_step8, use_container_width=True, key='step8_top_performing_communes')

# Analyse et Recommandations
st.markdown(f"""
**Analyse :** Ce graphique présente les {top_n_step8} communes où votre agence détient la plus haute part de marché moyenne depuis la première vente. Les barres vertes plus foncées indiquent une surperformance plus élevée.

**Recommandation :**

- **Améliorer les Sous-performances :** Bien que ces communes soient performantes, continuez à analyser les données pour identifier toute opportunité d'amélioration supplémentaire.
- **Capitaliser sur les Forces :** Concentrez-vous sur ces communes pour maintenir et renforcer votre position dominante.
- **Exploiter les Périodes de Pic :** Analysez les périodes où la part de marché est à son apogée pour reproduire ces succès dans d'autres zones.
""")


# --------------------------------------
# Step 9: Tendances Trimestrielles de Part de Marché par Communes Vendues (Version Ultra Compacte)
# --------------------------------------
st.header("9. Tendances Trimestrielles de Part de Marché par Communes Vendues (Version Ultra Compacte)")

# Filtrer les communes où l'agence a vendu des biens
sold_communes = df_agency['commune'].unique()
market_share_sold_communes = market_share[market_share['commune'].isin(sold_communes)]

# Convertir la colonne 'date' en datetime si ce n'est pas déjà fait
market_share_sold_communes['date'] = pd.to_datetime(market_share_sold_communes['date'])

# Obtenir les 8 périodes les plus récentes (derniers 2 ans)
latest_dates = market_share_sold_communes['date'].drop_duplicates().nlargest(8)
market_share_recent = market_share_sold_communes[market_share_sold_communes['date'].isin(latest_dates)]

# Identifier les 6 communes principales par volume de ventes
top_6_communes = df_agency_filtered['commune'].value_counts().head(6).index
market_share_top_6 = market_share_recent[market_share_recent['commune'].isin(top_6_communes)]

# Créer une disposition de sous-graphiques compacte avec 2 lignes et 3 colonnes
fig = make_subplots(
    rows=2,
    cols=3,
    shared_xaxes=True,
    vertical_spacing=0.15,
    horizontal_spacing=0.05,
    subplot_titles=[f"Commune: {commune}" for commune in top_6_communes]
)

# Tracer les tendances de part de marché pour chaque commune
for i, commune in enumerate(top_6_communes):
    row = i // 3 + 1
    col = i % 3 + 1
    commune_data = market_share_top_6[market_share_top_6['commune'] == commune].sort_values('date')
    
    # Vérifier si les données sont suffisantes
    if commune_data.empty:
        continue
    
    # Calculer la moyenne du marché pour la commune
    market_avg = commune_data['market_share (%)'].mean()
    
    # Tracer la part de marché de l'agence
    fig.add_trace(
        go.Scatter(
            x=commune_data['date'],
            y=commune_data['market_share (%)'],
            mode='lines+markers',
            name='Agence',
            line=dict(color='blue'),
            marker=dict(size=6),
            hovertemplate='%{x|%Y-%m-%d}: %{y:.2f}% (Agence)'
        ),
        row=row,
        col=col
    )
    
    # Tracer la moyenne du marché
    fig.add_trace(
        go.Scatter(
            x=commune_data['date'],
            y=[market_avg] * len(commune_data),
            mode='lines',
            name='Moyenne du Marché',
            line=dict(color='red', dash='dash'),
            hovertemplate='Moyenne Marché: %{y:.2f}%'
        ),
        row=row,
        col=col
    )
    
    # Ajouter des zones de surperformance et de sous-performance
    fig.add_trace(
        go.Scatter(
            x=commune_data['date'],
            y=np.maximum(commune_data['market_share (%)'], market_avg),
            mode='none',
            fill='toself',
            fillcolor='rgba(34, 139, 34, 0.2)',  # Vert clair
            hoverinfo='skip',
            showlegend=False
        ),
        row=row,
        col=col
    )
    
    fig.add_trace(
        go.Scatter(
            x=commune_data['date'],
            y=np.minimum(commune_data['market_share (%)'], market_avg),
            mode='none',
            fill='toself',
            fillcolor='rgba(220, 20, 60, 0.2)',  # Rouge clair
            hoverinfo='skip',
            showlegend=False
        ),
        row=row,
        col=col
    )
    
    # Annotation pour indiquer la surperformance et la sous-performance
    fig.add_annotation(
        x=0.5,
        y=1.05,
        xref=f'x domain',
        yref=f'y domain',
        text=f"<b>Surperformance</b> (Vert) et <b>Sous-performance</b> (Rouge)",
        showarrow=False,
        font=dict(size=10, color="black"),
        xanchor='center',
        yanchor='bottom',
        row=row,
        col=col
    )

# Mise à jour du layout pour une meilleure lisibilité et compacité
fig.update_layout(
    height=800,  # Ajusté pour une meilleure disposition
    title_text="Tendances Trimestrielles de Part de Marché pour les 6 Communes les Plus Vendues (Comparaison Agence vs Marché - Dernières 8 Périodes)",
    hovermode='x unified',
    showlegend=False,  # Légende globale désactivée pour éviter la surcharge
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
            y=1.02,
            showarrow=False,
            font=dict(size=12, family='Arial, sans-serif'),
            align='center'
        )
    ]
)

# Afficher le graphique dans Streamlit
st.plotly_chart(fig, use_container_width=True)

# **Remarque** : Pour une meilleure lisibilité, toutes les recommandations et analyses ont été intégrées dans les annotations et la disposition visuelle du graphique.


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
# Step 12: Benchmark des Concurrents par Commune (avec Filtre de Date Ajusté)
# --------------------------------------
st.header("12. Benchmark des Concurrents par Commune")

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
mes_communes = st.checkbox("Mes Communes", value=False)

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
        title='Ventes de l\'Agence vs Ventes des Concurrents par Commune (Top 20)' if not mes_communes else 'Ventes de l\'Agence vs Ventes des Concurrents (Mes Communes)',
        labels={'Nombre de Ventes': 'Nombre de Ventes', 'commune': 'Commune'},
        barmode='stack'
    )
    fig_competitor.update_layout(xaxis={'categoryorder': 'total descending'})

    # Afficher le graphique
    st.plotly_chart(fig_competitor, use_container_width=True)

    # Observations et recommandations compactes
    st.markdown("### Observations et Recommandations")

    # Analyse interne pour déterminer les communes clés
    top_competitor_communes = competitor_sales_filtered.sort_values('ventes_concurrents', ascending=False)['commune'].tolist()
    underperforming_communes = competitor_sales_filtered[competitor_sales_filtered['ventes_agence'] < competitor_sales_filtered['ventes_concurrents']]['commune'].tolist()
    top_agency_communes = competitor_sales_filtered.sort_values('ventes_agence', ascending=False)['commune'].tolist()

    # Observations
    st.markdown(f"**Communes avec forte concurrence :** {', '.join(top_competitor_communes[:3])}")
    st.markdown(f"**Communes où l'agence sous-performe :** {', '.join(underperforming_communes[:3])}")
    st.markdown(f"**Communes où l'agence performe bien :** {', '.join(top_agency_communes[:3])}")

    # Recommandations
    st.markdown("**Recommandations :**")
    st.markdown("- **Accentuer les efforts commerciaux** dans les communes où la concurrence est forte et où l'agence sous-performe.")
    st.markdown("- **Maintenir et renforcer la présence** dans les communes où l'agence performe bien pour conserver l'avantage compétitif.")
    st.markdown("- **Analyser les stratégies des concurrents** pour adapter nos approches marketing et commerciales.")
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
