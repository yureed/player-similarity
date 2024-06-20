import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import ftfy

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
    'Central Midfielder (MF)': [
        'Passes Completed', 'Key Passes', 'Assists', 'Tackles Won', 'Int', 'Prog Pass Dist', 'Prog Carries', 'Shots', 'Goals', 'Touches'
    ],
    'Defender (DF)': [
        'Tackle', 'Int', 'Clr', 'Blocks', 'AerialWins', 'Passes Completed', 'Passes Attempted', 'Succ Drb', 'Prog Pass Dist', 'Touches'
    ]
}

# Normalize all columns by dividing by '90s'
for column in all_columns:
    dataf[column] = dataf[column] / dataf['90s']

# Function to find similar players
def find_similar_players(player_name, player_club, position, min_90s, selected_columns, dataf):
    # Check if the player exists in the data
    if not ((dataf['Player'] == player_name) & (dataf['Squad'] == player_club)).any():
        return None, None
    
    # Extract the data for the given player
    player_data = dataf[(dataf['Player'] == player_name) & (dataf['Squad'] == player_club)][selected_columns]
    
    # Filter players based on position, minutes played, and age criteria, excluding the given player
    df = dataf[(dataf['Pos'].str.contains(position)) & 
               (dataf['90s'] >= min_90s) & 
               ~((dataf['Player'] == player_name) & (dataf['Squad'] == player_club))]
    
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

# Position selection
positions = ['DF', 'MF', 'FW']
selected_position = st.selectbox('Select Position', positions)

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
    df, similar_players_indices, similarity_scores = find_similar_players(player_name, player_club, selected_position, min_90s, selected_columns, dataf)

    if similar_players_indices is not None:
        num_similar_players = min(10, len(similar_players_indices))  # Number of similar players to retrieve
        st.write(f"Players similar to {player_name} from {player_club}:")
        for i in range(num_similar_players):
            similar_player_index = similar_players_indices[i]
            similarity_score = similarity_scores[similar_player_index]
            similar_player_name = df.iloc[similar_player_index]['Player']
            similar_player_club = df.iloc[similar_player_index]['Squad']
            st.write(f"{i+1}. {similar_player_name} ({similar_player_club}) (Similarity Score: {similarity_score:.3f})")
    else:
        st.write(f"Player {player_name} from {player_club} not found in the dataset.")

