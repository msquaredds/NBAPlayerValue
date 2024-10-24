import math
import pandas as pd
import streamlit as st

import GlobalVariables as gv

from plotly import graph_objects as go
from statsmodels import api as sm

from RegressionAnalysis import RegressionAnalysis


def main():
    st.set_page_config(page_title="NBA Player Value", layout="wide",
                       page_icon="🏀")

    title_cols = st.columns(3)
    with title_cols[1]:
        title_writing = "🏀 NBA Player Value 🏀"
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
    st.subheader("Game Value ⭐")
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
    # Data Pull & Cleaning
    ######################################################################
    team_box_score_data = pd.read_excel("NBAPlayerValueData.xlsx",
                                        "TeamBoxScores")
    with game_value_cols[1]:
        with st.expander("Team Box Score Data"):
            st.dataframe(team_box_score_data)

    # rename columns to be more readable
    team_box_score_data.columns = [
        col if col not in gv.COL_NAME_REPLACEMENTS.keys()
        else gv.COL_NAME_REPLACEMENTS[col]
        for col in team_box_score_data.columns]
    # narrow down the data to just the point differential and the columns
    # that we want to test
    dependent_data = team_box_score_data[gv.DEPENDENT_BOX_COL]
    independent_data = team_box_score_data[gv.INDEPENDENT_BOX_COLS]
    # make sure data is numeric
    dependent_data = pd.to_numeric(dependent_data)
    for col in independent_data.columns:
        # replace any dash with zero
        independent_data[col] = independent_data[col].replace("-", 0)
        independent_data[col] = pd.to_numeric(independent_data[col])

    ######################################################################
    # Game Value - Individual Analysis
    ######################################################################

    # we want to keep the intercept, coefficient, p-value and r-squared
    results = {}
    for col in independent_data.columns:
        # add a constant
        X = sm.add_constant(independent_data[col])
        # we want to remove any influential points, so we re-run the model
        # until there are none
        re_run = True
        while re_run:
            model = sm.OLS(dependent_data, X).fit()
            # use a robust covariance matrix
            model = model.get_robustcov_results()
            # check for influential points
            infl = model.get_influence()
            cooks_distance, cooks_p = infl.cooks_distance
            # remove any influential points
            if any(cooks_p < 0.05):
                # find the most influential point for any points with a
                # p-value less than 0.05
                infl_points = [(d, i) for d, p, i in
                               enumerate(zip(cooks_distance, cooks_p))
                               if p < 0.05]
                infl_points.sort(reverse=True)
                X = X.drop(infl_points[0][1])
            else:
                re_run = False

        results[col] = {
            "Sensitivity": model.params[1],
            "Correlation": math.sqrt(model.rsquared) * 100,
            "P-Value": model.pvalues[1],
            "R-Squared": model.rsquared * 100
        }

    # sort columns alphabetically
    results = {k: results[k] for k in sorted(results)}
    results_df = pd.DataFrame(results).T

    with game_value_cols[1]:
        st.markdown("#### Single Stat Analysis")
    ind_stats_header_cols = st.columns([.05, .25, .20, .05, .20, .05, .20])
    with ind_stats_header_cols[2]:
        st.markdown("##### Sensitivity")
    with ind_stats_header_cols[4]:
        st.markdown("##### Correlation")
    with ind_stats_header_cols[6]:
        st.markdown("##### P-Value")

    individual_stats_cols = st.columns([.05, .25, .20, .05, .20, .05, .20])
    with individual_stats_cols[1]:
        st.dataframe(
            results_df,
            height=565,
            column_config={
                "Sensitivity": st.column_config.NumberColumn(
                    width='small',
                    help="The change in point differential for a one unit "
                         "change in the stat. For example, if assist sensitivity "
                         "is 1.3, then for every assist a player has, they are "
                         "expected to contribute 1.3 points to the point "
                         "differential. This is the coefficient of the "
                         "independent variable in the regression model.",
                    format="%.2f"),
                "Correlation": st.column_config.NumberColumn(
                    width='small', format="%.0f%%"),
                "P-Value": st.column_config.NumberColumn(
                    width='small',
                    help="The probability that the coefficient is actually "
                         "zero, in which case the variable is not "
                         "significant. So a lower value is better here. If "
                         "this is less than 0.05, we can be 95% confident "
                         "that the coefficient is not zero. In statistics, "
                         "this is the standard for determining whether a "
                         "variable matters.",
                    format="%.4f"),
                "R-Squared": st.column_config.NumberColumn(
                    width='small',
                    help="The percentage of the variance in the dependent "
                         "variable that is explained by the independent "
                         "variable. For example, if the assist r-squared is "
                         "18%, then 18% of the point differential can be "
                         "explained by the number of assists. This is the "
                         "square of the correlation.",
                    format="%.0f%%")})
    with individual_stats_cols[2]:
        # sort the columns so the highest sensitivity is first
        results_df = results_df.sort_values("Sensitivity", ascending=True)
        fig = go.Figure(go.Bar(x=results_df["Sensitivity"],
                               y=results_df.index,
                               orientation='h',
                               marker_color="blue",
                               xhoverformat=".2f"))
        fig.update_layout(font_size=14,
                          height=585,
                          margin=dict(l=40, r=0, t=0, b=20))
        st.plotly_chart(fig, theme=None)
    with individual_stats_cols[4]:
        # sort the columns so the highest correlation is first
        results_df = results_df.sort_values("Correlation", ascending=True)
        fig = go.Figure(go.Bar(x=results_df["Correlation"],
                               y=results_df.index,
                               orientation='h',
                               marker_color="green",
                               xhoverformat=".0f",))
        fig.update_layout(font_size=14,
                          height=585,
                          margin=dict(l=40, r=0, t=0, b=20),
                          xaxis_ticksuffix="%")
        st.plotly_chart(fig, theme=None)
    with individual_stats_cols[6]:
        # sort the columns so the highest p-value is first
        results_df = results_df.sort_values("P-Value", ascending=False)
        fig = go.Figure(go.Bar(x=results_df["P-Value"],
                               y=results_df.index,
                               orientation='h',
                               marker_color="lightcoral",
                               xhoverformat=".4f",))
        fig.update_layout(font_size=14,
                          height=585,
                          margin=dict(l=40, r=0, t=0, b=20))
        st.plotly_chart(fig, theme=None)

    single_notes_cols = st.columns([.05, .95])
    with single_notes_cols[1]:
        st.markdown("###### Notes:")
        st.write("1. The sensitivity is the coefficient of the variable in "
                 "the regression. We use the term senstivity here because it "
                 "is the change in point differential for a one unit change "
                 "in the stat and more intuitive than coefficient.")

    ######################################################################
    # Game Value - Multivariate Analysis
    ######################################################################

    multi_analysis_cols = st.columns([.05, .95])
    with multi_analysis_cols[1]:
        st.markdown("#### Multiple Stat Analysis")

    # define the regression analysis engine for all regressions here
    regr_engine = RegressionAnalysis(indep_in=independent_data,
                                     depen_in=dependent_data,
                                     vif_cutoff=gv.VIF_CUTOFF)

    # run the regression
    error_str = regr_engine.ols_regression(gv.MIN_OBSERVATION_MULTIPLIER)

    # stop the program if the regression wasn't successful
    if error_str is not None:
        st.error(error_str)
        st.stop()

    # show the results and covariance robust results
    st.write(regr_engine.res.summary())


if __name__ == '__main__':
    main()