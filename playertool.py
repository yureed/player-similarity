import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import ftfy
from mplsoccer import Radar, grid
from mplsoccer.utils import FontManager


dataf = pd.read_csv('Final FBRef 2023-2024.csv')

def fix_encoding(text):
    return ftfy.fix_text(text)

dataf['Player'] = dataf['Player'].apply(fix_encoding)
dataf['Squad'] = dataf['Squad'].apply(fix_encoding)



dataf = dataf[~dataf['Pos'].str.contains('GK')]
dataf = dataf[dataf['Age'] != 0]


dataf['90s'] = dataf['Min'] / 90


all_columns = [col for col in dataf.columns if col not in ['Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Born', 'MP', 'Starts', 'Min']]

templates = {
    'Poacher (FW)': [
        'Goals', 'Shots', 'Shots on Target', 'Goals per Shot', 
        'Goals per Shot on Target', 'Penalties', 'xG', 
        'npxG', 'G-xG', 'npG-xG', 'Touches', 'Att Pen Touch', 
        'Prog Passes Rec', 'Drb Tkl%'
    ],
    'Complete Forward (FW)': [
        'Goals', 'Assists', 'Shots', 'Shots on Target', 
        'Goals per Shot', 'Goals per Shot on Target', 'Penalties', 
        'xG', 'npxG', 'Passes Completed', 'Passes Attempted', 
        'Key Passes', 'Succ Drb', 'Touches', 'Att Pen Touch'
    ],
    'Advanced Forward (FW)': [
        'Goals', 'Shots', 'Shots on Target', 'Goals per Shot', 
        'Goals per Shot on Target', 'Penalties', 'xG', 
        'npxG', 'G-xG', 'npG-xG', 'Succ Drb', 'Touches', 
        'Att Pen Touch', 'Prog Passes Rec'
    ],
    'Inside Forward (FW)': [
        'Goals', 'Shots', 'Shots on Target', 'Goals per Shot', 
        'Goals per Shot on Target', 'xG', 'npxG', 'Assists', 
        'Succ Drb', 'Touches', 'Att Pen Touch', 'Prog Passes Rec', 
        'Key Passes'
    ],
    'Winger (FW)': [
        'Assists', 'Crosses', 'Succ Drb', 'Touches', 'Att Pen Touch', 
        'Key Passes', 'Passes Completed', 'Passes Attempted', 
        'Crosses into Penalty Area', 'Prog Passes', 'Passes into Penalty Area'
    ],
    'Box-to-Box Midfielder (MF)': [
        'Goals', 'Assists', 'Passes Completed', 'Passes Attempted', 
        'Key Passes', 'Succ Drb', 'Tackles Won', 'Int', 
        'Touches', 'Prog Passes', 'Prog Carries', 'Carries To Fina lThird'
    ],
    'Deep-Lying Playmaker (MF)': [
        'Passes Completed', 'Passes Attempted', 'Key Passes', 
        'Touches', 'Prog Passes', 'Switches', 'Passes into Penalty Area', 
        'Succ Drb', 'Int', 'Tackles Won'
    ],
    'Advanced Playmaker (MF)': [
        'Assists', 'Key Passes', 'Passes Completed', 'Passes Attempted', 
        'Succ Drb', 'Touches', 'Prog Passes', 'Passes into Penalty Area', 
        'Prog Carries', 'Carries To Fina lThird'
    ],
    'Defensive Midfielder (MF)': [
        'Tackles Won', 'Int', 'Passes Completed', 'Passes Attempted', 
        'Touches', 'Blocks', 'Clr', 'Recov', 'Prog Passes', 
        'Prog Carries'
    ],
    'Roaming Playmaker (MF)': [
        'Assists', 'Key Passes', 'Passes Completed', 'Passes Attempted', 
        'Succ Drb', 'Touches', 'Prog Passes', 'Prog Carries', 
        'Carries To Fina lThird', 'Carries To Pen Area'
    ],
    'Wing-Back (DF)': [
        'Assists', 'Crosses', 'Succ Drb', 'Touches', 'Tackles Won', 
        'Int', 'Clr', 'Passes Completed', 
        'Passes Attempted', 'Key Passes', 'Crosses into Penalty Area', 
        'Prog Passes'
    ],
    'Inverted Full-Back (DF)': [
        'Passes Completed', 'Passes Attempted', 'Key Passes', 
        'Succ Drb', 'Touches', 'Tackles Won', 'Int', 
        'Prog Passes', 'Switches'
    ],
    'Ball-Playing Defender (DF)': [
        'Passes Completed', 'Passes Attempted', 'Key Passes', 
        'Succ Drb', 'Touches', 'Tackles Won', 'Int', 
        'Clr', 'Blocks', 'Recov', 'Prog Passes'
    ],
    'No-Nonsense Centre-Back (DF)': [
        'Tackles Won', 'Int', 'Clr', 'Blocks', 
        'Recov', 'AerialWins', 'AerialLoss', 'Passes Completed'
    ],
    'Libero (DF)': [
        'Tackles Won', 'Int', 'Passes Completed', 'Passes Attempted', 
        'Key Passes', 'Succ Drb', 'Touches', 'Prog Passes', 'Clr', 
        'Blocks', 'Recov'
    ],
    'Segundo Volante (MF)': [
        'Goals', 'Assists', 'Passes Completed', 'Passes Attempted', 
        'Key Passes', 'Succ Drb', 'Touches', 'Tackles Won', 'Int', 
        'Prog Passes', 'Prog Carries'
    ],
    'Mezzala (MF)': [
        'Assists', 'Goals', 'Key Passes', 'Passes Completed', 
        'Passes Attempted', 'Succ Drb', 'Touches', 'Prog Passes', 
        'Prog Carries', 'Carries To Fina lThird'
    ],
    'False Nine (FW)': [
        'Assists', 'Goals', 'Key Passes', 'Passes Completed', 
        'Passes Attempted', 'Succ Drb', 'Touches', 'Prog Passes', 
        'Prog Carries', 'Carries To Pen Area', 'Prog Passes Rec'
    ]
}
# Adjust columns for per-90 calculations
for column in all_columns:
    if column != '90s' and column != 'Age':
        dataf[column] = dataf[column] / dataf['90s']

# Sidebar for tool selection
tool_choice = st.sidebar.radio("Choose Tool", options=["Similarity Checker", "Scouting Tool", "Analyze One Player"])

# Conditional logic based on tool choice
if tool_choice == "Similarity Checker":
    # Filters and options for Similarity Checker
    competition_options = ['All Competitions'] + list(dataf['Comp'].unique())
    selected_competitions = st.sidebar.multiselect("Select Competitions", competition_options, default='All Competitions')

    if 'All Competitions' in selected_competitions:
        filtered_data = dataf
    else:
        filtered_data = dataf[dataf['Comp'].isin(selected_competitions)]

    positions = ['DF', 'MF', 'FW']
    selected_positions = st.sidebar.multiselect('Select Positions', positions, default=positions)
    min_90s = st.sidebar.slider('Minimum 90s played', int(dataf['90s'].min()), int(dataf['90s'].max()), int(dataf['90s'].min()))
    min_age, max_age = st.sidebar.slider('Age Range', int(dataf['Age'].min()), int(dataf['Age'].max()),
                                         (int(dataf['Age'].min()), int(dataf['Age'].max())))

    player_options = [f"{row['Player']} ({row['Squad']})" for idx, row in filtered_data.iterrows()]
    selected_player = st.sidebar.selectbox('Select Player', player_options)
    player_name, player_club = selected_player.split(' (')
    player_club = player_club[:-1]  # Remove trailing ')'

    template_options = list(templates.keys())
    selected_template = st.sidebar.selectbox('Select Template', template_options)
    selected_columns = templates[selected_template]

    # Function to find similar players without weights
    def find_similar_players(player_name, player_club, positions, min_90s, min_age, max_age, selected_columns, dataf):
        if not ((dataf['Player'] == player_name) & (dataf['Squad'] == player_club)).any():
            return None, None, None

        player_data = dataf[(dataf['Player'] == player_name) & (dataf['Squad'] == player_club)][selected_columns]
        df = dataf[
            dataf['Pos'].apply(lambda x: any(pos in x for pos in positions)) &
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

    # Similarity Checker Output
    if st.sidebar.button('Find Similar Players'):
        df, similar_players_indices, similarity_scores = find_similar_players(
            player_name, player_club, selected_positions, min_90s, min_age, max_age, selected_columns, filtered_data)

        if similar_players_indices is not None:
            num_similar_players = min(10, len(similar_players_indices))
            st.write(f"Players similar to {player_name} from {player_club}:")
            for i in range(num_similar_players):
                similar_player_index = similar_players_indices[i]
                similarity_score = similarity_scores[similar_player_index]
                similar_player_name = df.iloc[similar_player_index]['Player']
                similar_player_club = df.iloc[similar_player_index]['Squad']
                st.write(f"{i+1}. {similar_player_name} ({similar_player_club}) (Similarity Score: {similarity_score:.3f})")

            # Radar chart for most similar player
            most_similar_player_name = df.iloc[similar_players_indices[0]]['Player']
            most_similar_player_club = df.iloc[similar_players_indices[0]]['Squad']
            most_similar_player_metrics = df[(df['Player'] == most_similar_player_name) & (
                df['Squad'] == most_similar_player_club)].iloc[0][selected_columns]
            given_player_metrics = dataf[(dataf['Player'] == player_name) & (
                dataf['Squad'] == player_club)].iloc[0][selected_columns]

            params = selected_columns

            player_data_full = dataf[(dataf['Player'] == player_name) & (dataf['Squad'] == player_club)]
            df_with_player = pd.concat([df, player_data_full])

            # Lower and upper boundaries for the statistics
            low = [df_with_player[col].min() for col in selected_columns]
            high = [df_with_player[col].max() for col in selected_columns]

            # List of columns where lower values are better
            lower_is_better = ['Drb Past', 'Err', 'Carry Mistakes', 'Dispossessed', 'Yellows', 'Reds',
                               'Yellow2', 'Fls', 'Off', 'AerialLoss']
            lower_columns = [col for col in selected_columns if col in lower_is_better]

            # Radar chart setup (using your old radar chart code)
            radar = Radar(params, low, high,
                          lower_is_better=lower_columns,
                          round_int=[False] * len(params),
                          num_rings=4,
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

            radar.setup_axis(ax=axs['radar'], facecolor='black')
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

            title1_text = axs['title'].text(0.01, 0.65, player_name, fontsize=25, color='#01c49d',
                                            fontproperties=robotto_bold.prop, ha='left', va='center')
            title2_text = axs['title'].text(0.01, 0.25, player_club, fontsize=20,
                                            fontproperties=robotto_thin.prop,
                                            ha='left', va='center', color='#01c49d')
            title3_text = axs['title'].text(0.99, 0.65, most_similar_player_name, fontsize=25,
                                            fontproperties=robotto_bold.prop,
                                            ha='right', va='center', color='#d80499')
            title4_text = axs['title'].text(0.99, 0.25, most_similar_player_club, fontsize=20,
                                            fontproperties=robotto_thin.prop,
                                            ha='right', va='center', color='#d80499')
            fig.set_facecolor('#121212')

            st.pyplot(fig)
        else:
            st.write(f"No players found meeting the criteria.")

elif tool_choice == "Scouting Tool":
    # Filters and options for Scouting Tool
    competition_options = ['All Competitions'] + list(dataf['Comp'].unique())
    selected_competitions = st.sidebar.multiselect("Select Competitions", competition_options, default='All Competitions')

    if 'All Competitions' in selected_competitions:
        filtered_data = dataf
    else:
        filtered_data = dataf[dataf['Comp'].isin(selected_competitions)]

    positions = ['DF', 'MF', 'FW']
    selected_positions = st.sidebar.multiselect('Select Positions', positions, default=positions)
    min_90s = st.sidebar.slider('Minimum 90s played', int(dataf['90s'].min()), int(dataf['90s'].max()), int(dataf['90s'].min()))
    min_age, max_age = st.sidebar.slider('Age Range', int(dataf['Age'].min()), int(dataf['Age'].max()),
                                         (int(dataf['Age'].min()), int(dataf['Age'].max())))

    selected_template = st.sidebar.selectbox('Select Template', list(templates.keys()))
    selected_columns = st.sidebar.multiselect('Select Columns', all_columns, default=templates[selected_template])

    # Add weights for each selected column
    weights = {}
    st.sidebar.write("### Assign weights to each metric")
    for col in selected_columns:
        weights[col] = st.sidebar.slider(f"Weight for {col}", 0.0, 1.0, 0.5)

    # Scouting Tool Function: Find top players based on criteria
    def find_weighted_top_players(selected_positions, min_90s, min_age, max_age, selected_columns, weights, dataf):
        df = dataf[
            dataf['Pos'].apply(lambda x: any(pos in x for pos in selected_positions)) &
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

            # Radar chart setup (using your old radar chart code)
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

    competition_options = ['All Competitions'] + list(dataf['Comp'].unique())
    selected_competitions = st.sidebar.multiselect("Select Competitions", competition_options, default='All Competitions')

    if 'All Competitions' in selected_competitions:
        filtered_data = dataf
    else:
        filtered_data = dataf[dataf['Comp'].isin(selected_competitions)]

    player_options = [f"{row['Player']} ({row['Squad']})" for idx, row in filtered_data.iterrows()]
    selected_player = st.sidebar.selectbox('Select Player', player_options)
    player_name, player_club = selected_player.split(' (')
    player_club = player_club[:-1]  # Remove trailing ')'


    selected_age_range = st.sidebar.slider('Select Age Range', int(dataf['Age'].min()), int(dataf['Age'].max()),
                                           (int(dataf['Age'].min()), int(dataf['Age'].max())))
    min_90s = st.sidebar.slider('Minimum 90s played', int(dataf['90s'].min()), int(dataf['90s'].max()), int(dataf['90s'].min()))

    # Template selection and column customization
    template_options = list(templates.keys())
    selected_template = st.sidebar.selectbox('Select Template', template_options)
    selected_columns = st.sidebar.multiselect('Select Columns', all_columns, default=templates[selected_template])


    player_data = dataf[
        (dataf['Player'] == player_name) &
        (dataf['Squad'] == player_club) &
        (dataf['Age'] >= selected_age_range[0]) &
        (dataf['Age'] <= selected_age_range[1]) &
        (dataf['90s'] >= min_90s)
    ]

    if player_data.empty:
        st.write(f"No data available for {player_name} meeting the selected criteria.")
    else:
        st.write(f"Displaying radar chart for {player_name} ({player_club})")

        params = selected_columns
        player_metrics = player_data.iloc[0][selected_columns]

        # Calculate min and max for selected columns across all players
        low = [dataf[col].min() for col in selected_columns]
        high = [dataf[col].max() for col in selected_columns]

        # Radar chart setup (using your old radar chart code)
        radar = Radar(params, low, high,
                      lower_is_better=[],
                      round_int=[False] * len(params),
                      num_rings=4,
                      ring_width=1, center_circle_radius=1)

        fig, axs = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                        title_space=0, endnote_space=0, grid_key='radar', axis=False)

        radar.setup_axis(ax=axs['radar'], facecolor='black')
        rings_inner = radar.draw_circles(ax=axs['radar'], facecolor='orange', edgecolor='black')
        radar_output = radar.draw_radar(player_metrics, ax=axs['radar'],
                                        kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6})

        range_labels = radar.draw_range_labels(ax=axs['radar'], fontsize=23, color='white')
        param_labels = radar.draw_param_labels(ax=axs['radar'], fontsize=25, color='white')

        title_text = axs['title'].text(0.5, 0.5, f"{player_name} ({player_club})", fontsize=25,
                                       fontproperties=robotto_bold.prop, color='white',
                                       ha='center', va='center')
        fig.set_facecolor('#121212')

        st.pyplot(fig)
