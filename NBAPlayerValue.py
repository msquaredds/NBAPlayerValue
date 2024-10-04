import pandas as pd
import streamlit as st


def main():
    st.set_page_config(page_title="NBA Player Value", layout="wide")

    title_cols = st.columns(3)
    with title_cols[1]:
        title_writing = "NBA Player Value"
        title_format = (
            f'<p style="text-align: center; font-size: 40px; '
            f'font-weight: bold;">{title_writing}</p>')
        st.markdown(title_format, unsafe_allow_html=True)

    ######################################################################
    # Overview
    ######################################################################
    st.subheader("Overview")
    overview_cols = st.columns([.05, .95])
    with overview_cols[1]:
        st.write("This page investigates NBA player value through two lenses, "
                 "using statistical methods: "
                 "\n\n"
                 "1. Game Value - Measured in contribution to point "
                 "differential."
                 "\n\n"
                 "2. Monetary Value - Measured in dollars."
                 "\n\n"
                 "\n\n")

    ######################################################################
    # Team Stats
    ######################################################################
    st.subheader("Game Value")
    game_value_cols = st.columns([.05, .95])
    with game_value_cols[1]:
        st.write("To determine Game Value, we ultimately want to know how "
                 "likely a player is to help their team win. This could be "
                 "measured by determining the probability of a team winning "
                 "given a player's performance or we can approximate it by "
                 "looking at the contribution to point differential. Point "
                 "differential seems more intuitive, so we use that here."
                 "\n\n"
                 "\n\n")
        st.write("Build Up Steps (details are below): "
                 "\n\n"
                 "1. Regress team point differentials on team stats, one stat "
                 "at a time. This gives us a first interesting look at how "
                 "valuable each stat is. It helps convert things like assists "
                 "into point differential. And we cannot say that even points "
                 "are exactly one-to-one - if a player scores two points, "
                 "that gives the other team some chance to score two points "
                 "as well."
                 "\n\n"
                 "2. Create a holistic model that can include all the stats. "
                 "We will only consider stats that are individually "
                 "significant and will then remove any highly correlated "
                 "variables and finally use stepwise regression to get the "
                 "best model."
                 "\n\n"
                 "3. Apply this model to player stats to get an estimate of "
                 "their value."
                 "\n\n"
                 "\n\n")
        st.write("A few top-level notes on our data and assumptions "
                 "(we will explain further assumptions as we get into the "
                 "details):"
                 "\n\n"
                 "1. We are using box score data from the 2023-2024 NBA "
                 "season. More seasons would give us more data, but there is "
                 "also the possibility that the game changes over time. This "
                 "attempts to balance the two."
                 "\n\n"
                 "2. We assume that stats are, on average, equally important "
                 "for each team and player."
                 "\n\n"
                 "3. We assume that the player stats are representative of "
                 "how good the player actually is. As an extreme counter "
                 "example, the best player on a team could be benched all "
                 "season, but we can't know their true value from this data "
                 "exercise since we rely on their stats.")

    team_box_score_data = pd.read_excel("NBAPlayerValueData.xlsx",
                                        "TeamBoxScores", header=None)


if __name__ == '__main__':
    main()