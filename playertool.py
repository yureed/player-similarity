import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import ftfy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mplsoccer import Radar, grid
from sklearn.metrics import silhouette_score

from mplsoccer import PyPizza
from mplsoccer.utils import FontManager

# Load custom fonts for radar chart
URL1 = 'https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/SourceSerifPro-Regular.ttf'
serif_regular = FontManager(URL1)
URL2 = 'https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/SourceSerifPro-ExtraLight.ttf'
serif_extra_light = FontManager(URL2)
URL3 = 'https://raw.githubusercontent.com/google/fonts/main/ofl/rubikmonoone/RubikMonoOne-Regular.ttf'
rubik_regular = FontManager(URL3)
URL4 = 'https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Thin.ttf'
robotto_thin = FontManager(URL4)
URL5 = 'https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/RobotoSlab%5Bwght%5D.ttf'
robotto_bold = FontManager(URL5)

# Load data
dataf = pd.read_csv('Final FBRef 2024-2025.csv')

# Fix encoding issues in player and squad names
def fix_encoding(text):
    return ftfy.fix_text(text)

dataf['Player'] = dataf['Player'].apply(fix_encoding)
dataf['Squad'] = dataf['Squad'].apply(fix_encoding)

# Remove goalkeepers, players with age 0, and rows with missing 'Main Position'
dataf = dataf[~dataf['Pos'].str.contains('GK')]
dataf = dataf[dataf['Age'] != 0]
dataf = dataf.dropna(subset=['Main Position'])

# Calculate '90s' based on minutes played
dataf['90s'] = dataf['Min'] / 90

# Filter columns for per-90 and adjusted metrics
per90_columns = [col for col in dataf.columns if 'Per90' in col]
adjusted_columns = [col for col in dataf.columns if 'pAdj' in col]
selected_columns = per90_columns + adjusted_columns

templates = {
    'Poacher (FW)': [
        'GoalsPer90', 'ShotsPer90', 'SoTPer90', 'G/ShPer90', 
        'G/SoTPer90', 'PKPer90', 'xGPer90', 'npxGPer90', 
        'G-xGPer90', 'npG-xGPer90', 'TouchesPer90', 'AttPenTouchPer90', 
        'ProgPassesRecPer90', 'pAdjDrbTklPer90'
    ],
    'Complete Forward (FW)': [
        'GoalsPer90', 'AssistsPer90', 'ShotsPer90', 'SoTPer90', 
        'G/ShPer90', 'G/SoTPer90', 'PKPer90', 'xGPer90', 
        'npxGPer90', 'PassesCompletedPer90', 'PassesAttemptedPer90', 
        'KeyPassesPer90', 'SuccDrbPer90', 'TouchesPer90', 'AttPenTouchPer90'
    ],
    'Advanced Forward (FW)': [
        'GoalsPer90', 'ShotsPer90', 'SoTPer90', 'G/ShPer90', 
        'G/SoTPer90', 'PKPer90', 'xGPer90', 'npxGPer90', 
        'G-xGPer90', 'npG-xGPer90', 'SuccDrbPer90', 'TouchesPer90', 
        'AttPenTouchPer90', 'ProgPassesRecPer90'
    ],
    'Inside Forward (FW)': [
        'GoalsPer90', 'ShotsPer90', 'SoTPer90', 'G/ShPer90', 
        'G/SoTPer90', 'xGPer90', 'npxGPer90', 'AssistsPer90', 
        'SuccDrbPer90', 'TouchesPer90', 'AttPenTouchPer90', 'ProgPassesRecPer90', 
        'KeyPassesPer90'
    ],
    'Winger (FW)': [
        'AssistsPer90', 'CrsPer90', 'SuccDrbPer90', 'TouchesPer90', 
        'AttPenTouchPer90', 'KeyPassesPer90', 'PassesCompletedPer90', 
        'PassesAttemptedPer90', 'CrsPenAreaCmpPer90', 'ProgPassesPer90', 
        'PenAreaCmpPer90'
    ],
    'Box-to-Box Midfielder (MF)': [
        'GoalsPer90', 'AssistsPer90', 'PassesCompletedPer90', 
        'PassesAttemptedPer90', 'KeyPassesPer90', 'SuccDrbPer90', 
        'TklPer90', 'IntPer90', 'TouchesPer90', 'ProgPassesPer90', 
        'ProgCarriesPer90', 'CarriesToFinalThirdPer90'
    ],
    'Deep-Lying Playmaker (MF)': [
        'PassesCompletedPer90', 'PassesAttemptedPer90', 'KeyPassesPer90', 
        'TouchesPer90', 'ProgPassesPer90', 'SwitchesPer90', 
        'PenAreaCmpPer90', 'SuccDrbPer90', 'IntPer90', 'TklPer90'
    ],
    'Advanced Playmaker (MF)': [
        'AssistsPer90', 'KeyPassesPer90', 'PassesCompletedPer90', 
        'PassesAttemptedPer90', 'SuccDrbPer90', 'TouchesPer90', 
        'ProgPassesPer90', 'PenAreaCmpPer90', 'ProgCarriesPer90', 
        'CarriesToFinalThirdPer90'
    ],
    'Defensive Midfielder (MF)': [
        'TklPer90', 'IntPer90', 'PassesCompletedPer90', 
        'PassesAttemptedPer90', 'TouchesPer90', 'BlocksPer90', 
        'ClrPer90', 'RecovPer90', 'ProgPassesPer90', 'ProgCarriesPer90'
    ],
    'Roaming Playmaker (MF)': [
        'AssistsPer90', 'KeyPassesPer90', 'PassesCompletedPer90', 
        'PassesAttemptedPer90', 'SuccDrbPer90', 'TouchesPer90', 
        'ProgPassesPer90', 'ProgCarriesPer90', 'CarriesToFinalThirdPer90', 
        'CarriesToPenAreaPer90'
    ],
    'Wing-Back (DF)': [
        'AssistsPer90', 'CrsPer90', 'SuccDrbPer90', 'TouchesPer90', 
        'TklPer90', 'IntPer90', 'ClrPer90', 'PassesCompletedPer90', 
        'PassesAttemptedPer90', 'KeyPassesPer90', 'CrsPenAreaCmpPer90', 
        'ProgPassesPer90'
    ],
    'Inverted Full-Back (DF)': [
        'PassesCompletedPer90', 'PassesAttemptedPer90', 'KeyPassesPer90', 
        'SuccDrbPer90', 'TouchesPer90', 'TklPer90', 'IntPer90', 
        'ProgPassesPer90', 'SwitchesPer90'
    ],
    'Ball-Playing Defender (DF)': [
        'PassesCompletedPer90', 'PassesAttemptedPer90', 'KeyPassesPer90', 
        'SuccDrbPer90', 'TouchesPer90', 'TklPer90', 'IntPer90', 
        'ClrPer90', 'BlocksPer90', 'RecovPer90', 'ProgPassesPer90'
    ],
    'No-Nonsense Centre-Back (DF)': [
        'TklPer90', 'IntPer90', 'ClrPer90', 'BlocksPer90', 
        'RecovPer90', 'AerialWinsPer90', 'AerialLossPer90', 'PassesCompletedPer90'
    ],
    'Libero (DF)': [
        'TklPer90', 'IntPer90', 'PassesCompletedPer90', 
        'PassesAttemptedPer90', 'KeyPassesPer90', 'SuccDrbPer90', 
        'TouchesPer90', 'ProgPassesPer90', 'ClrPer90', 'BlocksPer90', 
        'RecovPer90'
    ],
    'Segundo Volante (MF)': [
        'GoalsPer90', 'AssistsPer90', 'PassesCompletedPer90', 
        'PassesAttemptedPer90', 'KeyPassesPer90', 'SuccDrbPer90', 
        'TouchesPer90', 'TklPer90', 'IntPer90', 'ProgPassesPer90', 
        'ProgCarriesPer90'
    ],
    'Mezzala (MF)': [
        'AssistsPer90', 'GoalsPer90', 'KeyPassesPer90', 'PassesCompletedPer90', 
        'PassesAttemptedPer90', 'SuccDrbPer90', 'TouchesPer90', 
        'ProgPassesPer90', 'ProgCarriesPer90', 'CarriesToFinalThirdPer90'
    ],
    'False Nine (FW)': [
        'AssistsPer90', 'GoalsPer90', 'KeyPassesPer90', 'PassesCompletedPer90', 
        'PassesAttemptedPer90', 'SuccDrbPer90', 'TouchesPer90', 
        'ProgPassesPer90', 'ProgCarriesPer90', 'CarriesToPenAreaPer90', 
        'ProgPassesRecPer90'
    ]
}

dataf = dataf[~dataf['Main Position'].str.strip().str.lower().isin(['attack', 'defence', 'midfield','goalkeeper'])]

# Calculate '90s' based on minutes played and add it as a new column
dataf['90s'] = dataf['Min'] / 90

# Adjust per 90 columns by dividing by the calculated '90s'
for column in dataf.columns:
    if 'Per90' in column and column != '90s':
        dataf[column] = dataf[column] / dataf['90s']

# Adjusting the tool to use 'Main Position' instead of 'Pos'
tool_choice = st.sidebar.radio("Choose Tool", options=["Similarity Checker", "Scouting Tool"])

# Main logic, assuming templates were already adjusted
if tool_choice == "Similarity Checker":
    competition_options = ['All Competitions'] + list(dataf['Comp'].unique())
    selected_competitions = st.sidebar.multiselect("Select Competitions", competition_options, default='All Competitions')

    if 'All Competitions' in selected_competitions:
        filtered_data = dataf
    else:
        filtered_data = dataf[dataf['Comp'].isin(selected_competitions)]
    all_players = dataf['Player'].unique().tolist()  # List of all players
    excluded_players = st.sidebar.multiselect("Exclude Players", all_players, default=[])
    
    # Apply the filter to exclude selected players
    filtered_data = filtered_data[~filtered_data['Player'].isin(excluded_players)]

    # Filtering based on 'Main Position' and user selections
    positions = dataf['Main Position'].unique().tolist()
    selected_positions = st.sidebar.multiselect('Select Positions', positions, default=positions)
    
    min_90s = st.sidebar.slider('Minimum 90s played', int(dataf['90s'].min()), int(dataf['90s'].max()), int(dataf['90s'].min()))
    min_age, max_age = st.sidebar.slider('Age Range', int(dataf['Age'].min()), int(dataf['Age'].max()),
                                         (int(dataf['Age'].min()), int(dataf['Age'].max())))

    player_options = [f"{row['Player']} ({row['Squad']})" for idx, row in filtered_data.iterrows()]
    selected_player = st.sidebar.selectbox('Select Player', player_options)
    player_name, player_club = selected_player.split(' (')
    player_club = player_club[:-1]

    template_options = list(templates.keys())
    selected_template = st.sidebar.selectbox('Select Template', template_options)
    selected_columns = templates[selected_template]
    selected_columns = st.sidebar.multiselect('Select Columns', selected_columns, default=templates[selected_template])

    # Function to find similar players
    def find_similar_players(player_name, player_club, positions, min_90s, min_age, max_age, selected_columns, dataf):
        player_data = dataf[(dataf['Player'] == player_name) & (dataf['Squad'] == player_club)][selected_columns]
        df = dataf[
            dataf['Main Position'].isin(positions) &
            (dataf['90s'] >= min_90s) &
            (dataf['Age'] >= min_age) &
            (dataf['Age'] <= max_age) &
            ~((dataf['Player'] == player_name) & (dataf['Squad'] == player_club))
        ]

        if df.empty:
            return None, None, None

        df = df.dropna(subset=selected_columns)

        scaler = StandardScaler()
        metrics_data_scaled = scaler.fit_transform(df[selected_columns])
        player_data_scaled = scaler.transform(player_data)

        cosine_sim_matrix = cosine_similarity(metrics_data_scaled, player_data_scaled)
        similarity_scores = cosine_sim_matrix.flatten()
        similar_players_indices = np.argsort(similarity_scores)[::-1]

        return df, similar_players_indices, similarity_scores

    # Find and display similar players
    if st.sidebar.button('Find Similar Players'):
        df, similar_players_indices, similarity_scores = find_similar_players(
            player_name, player_club, selected_positions, min_90s, min_age, max_age, selected_columns, filtered_data
        )

        if similar_players_indices is not None:
            num_similar_players = min(10, len(similar_players_indices))
            st.write(f"Players similar to {player_name} from {player_club}:")
            for i in range(num_similar_players):
                similar_player_index = similar_players_indices[i]
                similarity_score = similarity_scores[similar_player_index]
                similar_player_name = df.iloc[similar_player_index]['Player']
                similar_player_club = df.iloc[similar_player_index]['Squad']
                st.write(f"{i+1}. {similar_player_name} ({similar_player_club}) (Similarity Score: {similarity_score:.3f})")
             # Get the most similar player (the first in the list)
            most_similar_player_index = similar_players_indices[0]
            most_similar_player_score = similarity_scores[most_similar_player_index]
            most_similar_player_name = df.iloc[most_similar_player_index]['Player']
            most_similar_player_club = df.iloc[most_similar_player_index]['Squad']
            params = selected_columns

            # Calculate min and max for selected columns
            low = [df[col].min() for col in selected_columns]
            high = [df[col].max() for col in selected_columns]
            most_similar_player_metrics = df.iloc[most_similar_player_index][selected_columns].values

            # Radar chart setup
            radar = Radar(params, low, high,
                          lower_is_better=[],
                          round_int=[False] * len(params),
                          num_rings=4,
                          ring_width=1, center_circle_radius=1)

            fig, axs = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                            title_space=0, endnote_space=0, grid_key='radar', axis=False)

            radar.setup_axis(ax=axs['radar'], facecolor='black')
            rings_inner = radar.draw_circles(ax=axs['radar'], facecolor='orange', edgecolor='black')
            radar_output = radar.draw_radar(most_similar_player_metrics, ax=axs['radar'],
                                            kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6})

            range_labels = radar.draw_range_labels(ax=axs['radar'], fontsize=23, color='white')
            param_labels = radar.draw_param_labels(ax=axs['radar'], fontsize=25, color='white')

            title_text = axs['title'].text(0.5, 0.5, f"{most_similar_player_score} ({most_similar_player_club})", fontsize=25,
                                           fontproperties=robotto_bold.prop, color='white',
                                           ha='center', va='center')
            fig.set_facecolor('#121212')

            st.pyplot(fig)
elif tool_choice == "Scouting Tool":
    # Filters and options for Scouting Tool
    competition_options = ['All Competitions'] + list(dataf['Comp'].unique())
    selected_competitions = st.sidebar.multiselect("Select Competitions", competition_options, default='All Competitions')

    if 'All Competitions' in selected_competitions:
        filtered_data = dataf
    else:
        filtered_data = dataf[dataf['Comp'].isin(selected_competitions)]
    all_players = dataf['Player'].unique().tolist()  # List of all players
    excluded_players = st.sidebar.multiselect("Exclude Players", all_players, default=[])
    
    # Apply the filter to exclude selected players
    filtered_data = filtered_data[~filtered_data['Player'].isin(excluded_players)]

    # Filtering based on 'Main Position' and user selections
    positions = dataf['Main Position'].unique().tolist()
    selected_positions = st.sidebar.multiselect('Select Positions', positions, default=positions)
    min_90s = st.sidebar.slider('Minimum 90s played', int(dataf['90s'].min()), int(dataf['90s'].max()), int(dataf['90s'].min()))
    min_age, max_age = st.sidebar.slider('Age Range', int(dataf['Age'].min()), int(dataf['Age'].max()),
                                         (int(dataf['Age'].min()), int(dataf['Age'].max())))

    selected_template = st.sidebar.selectbox('Select Template', list(templates.keys()))
    selected_columns = list(set(st.sidebar.multiselect('Select Columns', selected_columns, default=templates[selected_template])))

    # Assign weights for each selected column once outside the 'Find Top Players' button click
    weights = {}
    st.sidebar.write("### Assign weights to each metric")
    for col in selected_columns:
        if col not in weights:
            weights[col] = st.sidebar.slider(f"Weight for {col}", 0.0, 1.0, 0.5)

    def calculate_percentiles(df, columns):
        return df[columns].rank(pct=True).multiply(100).round()  # Added rounding here

    # Function to calculate weighted scores using PCA and Grid Search for weight tuning
    def calculate_pca_weighted_scores(df, columns, weights):
        # Rescale weights to sum to 1
        weights = np.array(weights)
        weights /= weights.sum()
        
        # Apply weights directly to scaled metrics before PCA
        scaler = StandardScaler()
        metrics_data_scaled = scaler.fit_transform(df[columns])
        weighted_metrics = metrics_data_scaled * weights  # Element-wise multiplication
        
        # Perform PCA on weighted data
        pca = PCA(n_components=0.95)
        pca_data = pca.fit_transform(weighted_metrics)
        
        # Calculate final scores based on PCA components
        pca_weights = np.ones(pca_data.shape[1])  # Equal weights for PCA components
        weighted_pca_scores = np.dot(pca_data, pca_weights)
        
        # Normalize scores for comparison
        min_score, max_score = weighted_pca_scores.min(), weighted_pca_scores.max()
        normalized_scores = (weighted_pca_scores - min_score) / (max_score - min_score) * 100 if max_score != min_score else weighted_pca_scores
        
        return normalized_scores
        
    # Function for clustering players by similarity before ranking
    def cluster_players(df, columns):
        scaler = StandardScaler()
        metrics_data_scaled = scaler.fit_transform(df[columns])
        
        # Identify optimal number of clusters based on Silhouette Score
        best_k, best_score = 2, -1
        for k in range(2, 10):  # Range of clusters to try
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(metrics_data_scaled)
            score = silhouette_score(metrics_data_scaled, labels)
            if score > best_score:
                best_k, best_score = k, score
        
        # Final clustering with optimal k
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(metrics_data_scaled)
        return df
    
    # Updated display function for the pizza plot with threshold lines
    def display_pizza_plot(player_name, player_club, df, columns, percentiles, thresholds=None):
        # Shorten or abbreviate longer labels if necessary
        short_columns = [col[:10] + '...' if len(col) > 10 else col for col in columns]
        
        # Adjust font size based on length of labels
        font_size = 10 if max(len(col) for col in short_columns) > 10 else 12
        
        # Get player percentiles
        player_percentiles = percentiles[(df['Player'] == player_name) & (df['Squad'] == player_club)].iloc[0].round()
        
        # Create pizza plot with customized label formatting
        baker = PyPizza(
            params=short_columns,
            background_color="#121212",
            straight_line_color="#222222",
            straight_line_lw=1,
            last_circle_lw=1.5,
            last_circle_color="#121212",
            other_circle_lw=1,
            other_circle_color="#222222",
        )
        
        fig, ax = baker.make_pizza(
            player_percentiles.values,
            figsize=(8, 8),
            color_blank_space="same",
            slice_colors=["#00f2c1"] * len(short_columns),
            value_colors=["white"] * len(short_columns),
            value_bck_colors=["#121212"] * len(short_columns),
            kwargs_slices=dict(edgecolor="#222222", linewidth=1),
            kwargs_params=dict(color="white", fontsize=font_size),  # Removed rotation here
            kwargs_values=dict(color="white", fontsize=11, fontweight="bold", zorder=3),
        )
        
        # Adjust label rotation separately
        for text in ax.texts:
            text.set_rotation(45)
        
        # Display player name and club at the top
        fig.text(0.5, 0.97, f"{player_name} ({player_club})", size=16, color="white", ha="center", fontweight="bold")
        plt.show()

    
    # Main function to find top players
    if st.sidebar.button('Find Top Players'):
        weights_array = np.array([weights[col] for col in selected_columns])  # Get slider values once, avoiding duplicates
        weights_array /= weights_array.sum()  
        
        filtered_df = filtered_data[
            (filtered_data['Main Position'].isin(selected_positions)) &
            (filtered_data['90s'] >= min_90s) &
            (filtered_data['Age'] >= min_age) & 
            (filtered_data['Age'] <= max_age)
        ].dropna(subset=selected_columns)
        
        if not filtered_df.empty:
            # Calculate percentiles for the selected columns
            percentiles = calculate_percentiles(filtered_df, selected_columns)
            
            # Clustering players based on selected metrics
            filtered_df = cluster_players(filtered_df, selected_columns)
            
            st.write("Top 10 Players based on scouting criteria (Clustered):")
            
            # Calculate PCA weighted scores
            normalized_scores = calculate_pca_weighted_scores(filtered_df, selected_columns, weights_array)
            
            # Sort and display top players
            top_players = filtered_df.assign(Score=normalized_scores).sort_values(by='Score', ascending=False)
            top_players = top_players.head(10)
            
            for i, (idx, row) in enumerate(top_players.iterrows(), 1):
                player_name, player_club, score = row['Player'], row['Squad'], row['Score']
                st.write(f"{i}. {player_name} ({player_club}) - Score: {score:.2f}")
            
            # Display pizza plot for top player
            top_player = top_players.iloc[0]
            top_player_name, top_player_club = top_player['Player'], top_player['Squad']
            
            st.write(f"### Pizza Plot for Top Player: {top_player_name} ({top_player_club})")
            
            # Thresholds can represent benchmark percentiles for each metric
            thresholds = [75] * len(selected_columns)  # example benchmark value
            display_pizza_plot(top_player_name, top_player_club, filtered_df, selected_columns, percentiles, thresholds)
        else:
            st.write("No players found meeting the criteria.")
