from selectors import SelectSelector

import pandas as pd
import streamlit as st

import GlobalVariables as gv

from statsmodels import api as sm


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
    # Team Stats - Intro
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
        with st.expander("Build Up Steps"):
            st.write("1. Regress team point differentials on team stats, one "
                     "stat at a time. This gives us a first interesting look "
                     "at how valuable each stat is. It helps convert things "
                     "like assists into point differential. And we cannot "
                     "say that even points are exactly one-to-one - if a "
                     "player scores two points, that gives the other team "
                     "some chance to score two points as well."
                     "\n\n"
                     "2. Create a holistic model that can include all the "
                     "stats. We will only consider stats that are "
                     "individually significant and will then remove any "
                     "highly correlated variables and finally use stepwise "
                     "regression to get the best model."
                     "\n\n"
                     "3. Apply this model to player stats to get an estimate "
                     "of their value.")
        with st.expander("Assumptions"):
            st.write("1. We are using box score data from the 2023-2024 NBA "
                     "season. More seasons would give us more data, but there "
                     "is also the possibility that the game changes over "
                     "time. This attempts to balance the two."
                     "\n\n"
                     "2. We assume that stats are, on average, equally "
                     "important for each team and player."
                     "\n\n"
                     "3. We assume that the player stats are representative "
                     "of how good the player actually is. As an extreme "
                     "counter example, the best player on a team could be "
                     "benched all season, but we can't know their true value "
                     "from this data exercise since we rely on their stats.")

    ######################################################################
    # Team Stats - Analysis
    ######################################################################
    team_box_score_data = pd.read_excel("NBAPlayerValueData.xlsx",
                                        "TeamBoxScores")
    # rename columns to replace % with _P and / with _ and any other
    # specifics
    team_box_score_data.columns = [
        col if col not in gv.COL_NAME_REPLACEMENTS.keys()
        else gv.COL_NAME_REPLACEMENTS[col]
        for col in team_box_score_data.columns]
    team_box_score_data.columns = [col.replace("%", "_P").replace("/", "_")
                                   for col in team_box_score_data.columns]
    with game_value_cols[1]:
        with st.expander("Team Box Score Data"):
            st.dataframe(team_box_score_data)

    # narrow down the data to just the point differential and the columns
    # that we want to test
    dependent_data = team_box_score_data[[gv.DEPENDENT_BOX_COL]]
    independent_data = team_box_score_data[gv.INDEPENDENT_BOX_COLS]
    # make sure data is numeric
    dependent_data = pd.to_numeric(dependent_data)
    st.write(dependent_data)
    for col in independent_data.columns:
        independent_data[col] = pd.to_numeric(independent_data[col])
        st.write(independent_data[col])

    # we want to keep the intercept, coefficient, p-value and r-squared
    results = {}
    for col in independent_data.columns:
        # add a constant
        X = sm.add_constant(independent_data[col])
        st.write(dependent_data)
        st.write(X)
        model = sm.OLS(dependent_data, X).fit()
        # use a robust covariance matrix
        model = model.get_robustcov_results()
        results[col] = {
            "intercept": model.params[0],
            "coefficient": model.params[1],
            "p-value": model.pvalues[1],
            "r-squared": model.rsquared
        }

    results_df = pd.DataFrame(results).T
    with game_value_cols[1]:
        st.dataframe(results_df)



if __name__ == '__main__':
    main()