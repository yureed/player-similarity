import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import ftfy
from mplsoccer import Radar, grid
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
tool_choice = st.sidebar.radio("Choose Tool", options=["Similarity Checker", "Scouting Tool", "Analyze One Player"])

# Main logic, assuming templates were already adjusted
if tool_choice == "Similarity Checker":
    competition_options = ['All Competitions'] + list(dataf['Comp'].unique())
    selected_competitions = st.sidebar.multiselect("Select Competitions", competition_options, default='All Competitions')

    if 'All Competitions' in selected_competitions:
        filtered_data = dataf
    else:
        filtered_data = dataf[dataf['Comp'].isin(selected_competitions)]

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
elif tool_choice == "Scouting Tool":
    # Filters and options for Scouting Tool
    competition_options = ['All Competitions'] + list(dataf['Comp'].unique())
    selected_competitions = st.sidebar.multiselect("Select Competitions", competition_options, default='All Competitions')

    if 'All Competitions' in selected_competitions:
        filtered_data = dataf
    else:
        filtered_data = dataf[dataf['Comp'].isin(selected_competitions)]

    # Filtering based on 'Main Position' and user selections
    positions = dataf['Main Position'].unique().tolist()
    selected_positions = st.sidebar.multiselect('Select Positions', positions, default=positions)
    min_90s = st.sidebar.slider('Minimum 90s played', int(dataf['90s'].min()), int(dataf['90s'].max()), int(dataf['90s'].min()))
    min_age, max_age = st.sidebar.slider('Age Range', int(dataf['Age'].min()), int(dataf['Age'].max()),
                                         (int(dataf['Age'].min()), int(dataf['Age'].max())))

    selected_template = st.sidebar.selectbox('Select Template', list(templates.keys()))
    selected_columns = st.sidebar.multiselect('Select Columns', selected_columns, default=templates[selected_template])

    # Add weights for each selected column
    weights = {}
    st.sidebar.write("### Assign weights to each metric")
    for col in selected_columns:
        weights[col] = st.sidebar.slider(f"Weight for {col}", 0.0, 1.0, 0.5)

    # Function to find top players based on criteria
    def find_weighted_top_players(selected_positions, min_90s, min_age, max_age, selected_columns, weights, dataf):
        df = dataf[
            dataf['Main Position'].isin(selected_positions) &
            (dataf['90s'] >= min_90s) &
            (dataf['Age'] >= min_age) &
            (dataf['Age'] <= max_age)
        ]

        if df.empty:
            return None, None, None

        df = df.dropna(subset=selected_columns)

        scaler = StandardScaler()
        metrics_data_scaled = scaler.fit_transform(df[selected_columns])

        # Adjust metrics data by weights
        weighted_metrics = metrics_data_scaled * np.array([weights[col] for col in selected_columns])
        raw_scores = np.sum(weighted_metrics, axis=1)  # Calculate a combined score

        # Normalize scores to be out of 100
        max_score = raw_scores.max()
        normalized_scores = (raw_scores / max_score) * 100 if max_score != 0 else raw_scores

        sorted_indices = np.argsort(normalized_scores)[::-1]
        return df, sorted_indices, normalized_scores

    # Display Top Players
    if st.sidebar.button('Find Top Players'):
        df, sorted_indices, normalized_scores = find_weighted_top_players(
            selected_positions, min_90s, min_age, max_age, selected_columns, weights, filtered_data
        )

        if sorted_indices is not None:
            st.write("Top 10 Players based on scouting criteria:")
            for i in range(min(10, len(sorted_indices))):
                idx = sorted_indices[i]
                score = normalized_scores[idx]
                player_name = df.iloc[idx]['Player']
                player_club = df.iloc[idx]['Squad']
                st.write(f"{i+1}. {player_name} ({player_club}) - Score: {score:.2f}")
            # Radar chart for the top player
            top_player_name = df.iloc[sorted_indices[0]]['Player']
            top_player_club = df.iloc[sorted_indices[0]]['Squad']
            top_player_metrics = df.iloc[sorted_indices[0]][selected_columns]

            params = selected_columns

            # Calculate min and max for selected columns
            low = [df[col].min() for col in selected_columns]
            high = [df[col].max() for col in selected_columns]

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
            radar_output = radar.draw_radar(top_player_metrics, ax=axs['radar'],
                                            kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6})

            range_labels = radar.draw_range_labels(ax=axs['radar'], fontsize=23, color='white')
            param_labels = radar.draw_param_labels(ax=axs['radar'], fontsize=25, color='white')

            title_text = axs['title'].text(0.5, 0.5, f"{top_player_name} ({top_player_club})", fontsize=25,
                                           fontproperties=robotto_bold.prop, color='white',
                                           ha='center', va='center')
            fig.set_facecolor('#121212')

            st.pyplot(fig)
        else:
            st.write("No players found meeting the criteria.")
elif tool_choice == "Analyze One Player":
    # Competition filter
    competition_options = ['All Competitions'] + list(dataf['Comp'].unique())
    selected_competitions = st.sidebar.multiselect("Select Competitions", competition_options, default='All Competitions')

    # Filter data based on competition selection
    if 'All Competitions' in selected_competitions:
        filtered_data = dataf
    else:
        filtered_data = dataf[dataf['Comp'].isin(selected_competitions)]

    # Player selection from filtered data
    player_options = [f"{row['Player']} ({row['Squad']})" for idx, row in filtered_data.iterrows()]
    selected_player = st.sidebar.selectbox('Select Player', player_options)
    player_name, player_club = selected_player.split(' (')
    player_club = player_club[:-1]

    # Age range and minimum 90s played filter
    selected_age_range = st.sidebar.slider('Select Age Range', int(filtered_data['Age'].min()), int(filtered_data['Age'].max()),
                                           (int(filtered_data['Age'].min()), int(filtered_data['Age'].max())))
    min_90s = st.sidebar.slider('Minimum 90s played', int(filtered_data['90s'].min()), int(filtered_data['90s'].max()), int(filtered_data['90s'].min()))

    # Template and columns selection for radar chart
    template_options = list(templates.keys())
    selected_template = st.sidebar.selectbox('Select Template', template_options)
    selected_columns = st.sidebar.multiselect('Select Columns', selected_columns, default=templates[selected_template])

    # Apply all filters for selected player and criteria
    player_data = filtered_data[
        (filtered_data['Age'] >= selected_age_range[0]) &
        (filtered_data['Age'] <= selected_age_range[1]) &
        (filtered_data['90s'] >= min_90s)
    ]
    st.write('min_90s')
    if player_data.empty:
        st.write(f"No data available for {player_name} meeting the selected criteria.")
    else:
        st.write(f"Displaying radar chart for {player_name} ({player_club})")

        # Prepare data for radar chart
        params = selected_columns
        player_metrics = player_data.iloc[0][selected_columns].values

        # Calculate min and max for radar chart based on currently filtered data
        low = [filtered_data[selected_columns].min()[col] for col in selected_columns]
        high = [filtered_data[selected_columns].max()[col] for col in selected_columns]

        # Radar chart setup
        radar = Radar(params, low, high,
                      lower_is_better=[],
                      round_int=[False] * len(params),
                      num_rings=4,
                      ring_width=1, center_circle_radius=1)

        fig, axs = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                        title_space=0, endnote_space=0, grid_key='radar', axis=False)

        radar.setup_axis(ax=axs['radar'], facecolor='black')
        rings_inner = radar.draw_circles(ax=axs['radar'], facecolor='#ffb2b2', edgecolor='#fc5f5f')
        radar_output = radar.draw_radar(player_metrics, ax=axs['radar'],
                                        kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6})

        range_labels = radar.draw_range_labels(ax=axs['radar'], fontsize=23, color='white')
        param_labels = radar.draw_param_labels(ax=axs['radar'], fontsize=25, color='white')

        # Set title for radar chart
        title_text = axs['title'].text(0.5, 0.5, f"{player_name} ({player_club})", fontsize=25,
                                       fontproperties=robotto_bold.prop, color='white',
                                       ha='center', va='center')
        fig.set_facecolor('#121212')

        # Display the radar chart
        st.pyplot(fig)


