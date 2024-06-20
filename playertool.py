import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import ftfy
from mplsoccer import Radar, grid
from mplsoccer.utils import FontManager

# Read the CSV file
dataf = pd.read_csv('Final FBRef 2023-2024.csv')

# Function to fix text encoding issues
def fix_encoding(text):
    return ftfy.fix_text(text)

# Apply the fix to the 'Player' and 'Squad' columns
dataf['Player'] = dataf['Player'].apply(fix_encoding)
dataf['Squad'] = dataf['Squad'].apply(fix_encoding)

# Create the '90s' column by dividing 'Min' by 90
dataf['90s'] = dataf['Min'] / 90

# List of all columns except the excluded ones
all_columns = [col for col in dataf.columns if col not in ['Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age', 'Born', 'MP', 'Starts', 'Min']]

# Templates for different positions
templates = {
    'Attacking Winger (FW)': [
        'Goals', 'Shots', 'Shots on Target', 'Assists', 'Key Passes', 'Crosses', 'Succ Drb', 'AerialWins', 'Prog Carries', 'Carries To Pen Area'
    ],
    'Central Forward (FW)': [
        'Goals', 'Shots', 'Shots on Target', 'Assists', 'Key Passes', 'AerialWins', 'Succ Drb', 'Att Pen Touch', 'xG', 'Penalties'
    ],
    'Central Midfielder (MF)': [
        'Passes Completed', 'Key Passes', 'Assists', 'Tackles Won', 'Int', 'Prog Pass Dist', 'Prog Carries', 'Shots', 'Goals', 'Touches'
    ],
    'Defensive Midfielder (MF)': [
        'Tackle', 'Int', 'Clr', 'Blocks', 'Passes Completed', 'Passes Attempted', 'Prog Pass Dist', 'Recov', 'AerialWins', 'Touches'
    ],
    'Attacking Midfielder (MF)': [
        'Goals', 'Assists', 'Key Passes', 'Shots', 'Shots on Target', 'Succ Drb', 'Prog Carries', 'Passes into Penalty Area', 'xA', 'xG'
    ],
    'Fullback (DF)': [
        'Tackle', 'Int', 'Clr', 'Blocks', 'Crosses', 'Passes Completed', 'Passes Attempted', 'Succ Drb', 'Prog Pass Dist', 'Touches'
    ],
    'Center Back (DF)': [
        'Tackle', 'Int', 'Clr', 'Blocks', 'AerialWins', 'Passes Completed', 'Passes Attempted', 'Prog Pass Dist', 'Touches', 'Recov'
    ]
}

# Normalize all columns except '90s' by dividing by '90s'
for column in all_columns:
    if column != '90s':  # Ensure '90s' column is not normalized
        dataf[column] = dataf[column] / dataf['90s']

# Function to find similar players
def find_similar_players(player_name, player_club, positions, min_90s, selected_columns, dataf):
    # Check if the player exists in the data
    if not ((dataf['Player'] == player_name) & (dataf['Squad'] == player_club)).any():
        return None, None
    
    # Extract the data for the given player
    player_data = dataf[(dataf['Player'] == player_name) & (dataf['Squad'] == player_club)][selected_columns]
    
    # Filter players based on position, minutes played, and age criteria, excluding the given player
    df = dataf[dataf['Pos'].apply(lambda x: any(pos in x for pos in positions)) & 
               (dataf['90s'] >= min_90s) & 
               ~((dataf['Player'] == player_name) & (dataf['Squad'] == player_club))]
    
    # Check if the filtered DataFrame is empty
    if df.empty:
        return None, None
    
    # Remove rows with missing values in selected columns
    df = df.dropna(subset=selected_columns)
    
    # Standardize the data (mean=0, std=1)
    scaler = StandardScaler()
    metrics_data_scaled = scaler.fit_transform(df[selected_columns])
    player_data_scaled = scaler.transform(player_data)
    
    # Calculate cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(metrics_data_scaled, player_data_scaled)
    
    # Get similarity scores for all players compared to the selected player
    similarity_scores = cosine_sim_matrix.flatten()
    
    # Sort players based on similarity scores in descending order
    similar_players_indices = np.argsort(similarity_scores)[::-1]
    
    return df, similar_players_indices, similarity_scores

# Streamlit App
st.title('Player Similarity Finder')

# Player selection
player_options = [f"{row['Player']} ({row['Squad']})" for idx, row in dataf.iterrows()]
selected_player = st.selectbox('Select Player', player_options)

# Extract player name and club
player_name, player_club = selected_player.split(' (')
player_club = player_club[:-1]  # Remove trailing ')'

# Position selection (allow multiple positions)
positions = ['DF', 'MF', 'FW']
selected_positions = st.multiselect('Select Positions', positions, default=positions)

# 90s filter with dynamic min and max values
min_90s_value = int(dataf['90s'].min())
max_90s_value = int(dataf['90s'].max())
min_90s = st.slider('Minimum 90s played', min_value=min_90s_value, max_value=max_90s_value, value=min_90s_value)

# Template selection
template_options = list(templates.keys())
selected_template = st.selectbox('Select Template', template_options)

# Get the selected template columns
selected_columns = templates[selected_template]

# Column selection
selected_columns = st.multiselect('Select Columns', all_columns, default=selected_columns)

# Find similar players
if st.button('Find Similar Players'):
    df, similar_players_indices, similarity_scores = find_similar_players(player_name, player_club, selected_positions, min_90s, selected_columns, dataf)

    if similar_players_indices is not None:
        num_similar_players = min(10, len(similar_players_indices))  # Number of similar players to retrieve
        st.write(f"Players similar to {player_name} from {player_club}:")
        similar_players = []
        for i in range(num_similar_players):
            similar_player_index = similar_players_indices[i]
            similarity_score = similarity_scores[similar_player_index]
            similar_player_name = df.iloc[similar_player_index]['Player']
            similar_player_club = df.iloc[similar_player_index]['Squad']
            similar_players.append(f"{similar_player_name} ({similar_player_club})")
            st.write(f"{i+1}. {similar_player_name} ({similar_player_club}) (Similarity Score: {similarity_score:.3f})")
        
        # Radar chart selection
        selected_similar_player = st.selectbox('Select a player for radar chart comparison', similar_players)
        if selected_similar_player:
            selected_similar_player_name, selected_similar_player_club = selected_similar_player.split(' (')
            selected_similar_player_club = selected_similar_player_club[:-1]  # Remove trailing ')'

            # Extract metrics for the most similar player and the given player
            most_similar_player_metrics = df[(df['Player'] == selected_similar_player_name) & (df['Squad'] == selected_similar_player_club)].iloc[0][selected_columns]
            given_player_metrics = dataf[(dataf['Player'] == player_name) & (dataf['Squad'] == player_club)].iloc[0][selected_columns]

            # Parameters (metrics) for the radar chart
            params = selected_columns

            player_data_full = dataf[(dataf['Player'] == player_name) & (dataf['Squad'] == player_club)]

            # Concatenate the filtered dataframe with the player data to include it back
            df_with_player = pd.concat([df, player_data_full])

            # Lower and upper boundaries for the statistics
            low = [df_with_player[col].min() for col in selected_columns]
            high = [df_with_player[col].max() for col in selected_columns]

            # Create the radar chart
            radar = Radar(params, low, high,
                          # whether to round any of the labels to integers instead of decimal places
                          round_int=[False]*len(params),
                          num_rings=4,  # the number of concentric circles (excluding center circle)
                          ring_width=1, center_circle_radius=1)

            URL1 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/'
                    'SourceSerifPro-Regular.ttf')
            serif_regular = FontManager(URL1)
            URL2 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/'
                    'SourceSerifPro-ExtraLight.ttf')
                        serif_extra_light = FontManager(URL2)
            URL3 = ('https://raw.githubusercontent.com/google/fonts/main/ofl/rubikmonoone/'
                    'RubikMonoOne-Regular.ttf')
            rubik_regular = FontManager(URL3)
            URL4 = 'https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Thin.ttf'
            robotto_thin = FontManager(URL4)
            URL5 = ('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
                    'RobotoSlab%5Bwght%5D.ttf')
            robotto_bold = FontManager(URL5)

            fig, axs = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                            title_space=0, endnote_space=0, grid_key='radar', axis=False)

            # Plot radar
            radar.setup_axis(ax=axs['radar'], facecolor='black')  # Format axis as a radar
            rings_inner = radar.draw_circles(ax=axs['radar'], facecolor='orange', edgecolor='black')
            radar_output = radar.draw_radar_compare(given_player_metrics, most_similar_player_metrics, ax=axs['radar'],
                                                    kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6},
                                                    kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
            radar_poly, radar_poly2, vertices1, vertices2 = radar_output
            range_labels = radar.draw_range_labels(ax=axs['radar'], fontsize=23, color='white')
            param_labels = radar.draw_param_labels(ax=axs['radar'], fontsize=25, color='white')
            axs['radar'].scatter(vertices1[:, 0], vertices1[:, 1],
                                 c='#00f2c1', edgecolors='#6d6c6d', marker='o', s=150, zorder=2)
            axs['radar'].scatter(vertices2[:, 0], vertices2[:, 1],
                                 c='#d80499', edgecolors='#6d6c6d', marker='o', s=150, zorder=2)

            # Adding the endnote and title text (these axes range from 0-1, i.e. 0, 0 is the bottom left)
            title1_text = axs['title'].text(0.01, 0.65, player_name, fontsize=25, color='#01c49d',
                                            fontproperties=robotto_bold.prop, ha='left', va='center')
            title2_text = axs['title'].text(0.01, 0.25, player_club, fontsize=20,
                                            fontproperties=robotto_thin.prop,
                                            ha='left', va='center', color='#01c49d')
            title3_text = axs['title'].text(0.99, 0.65, selected_similar_player_name, fontsize=25,
                                            fontproperties=robotto_bold.prop,
                                            ha='right', va='center', color='#d80499')
            title4_text = axs['title'].text(0.99, 0.25, selected_similar_player_club, fontsize=20,
                                            fontproperties=robotto_thin.prop,
                                            ha='right', va='center', color='#d80499')
            fig.set_facecolor('#121212')

            st.pyplot(fig)

           
