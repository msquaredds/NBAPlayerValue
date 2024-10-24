"""
Runs different types of regressions.
:class RegressionAnalysis: All the different regression types.

Created by: Alex Melesko
Date: 5/24/2022
"""

import numpy as np
import pandas as pd

from statsmodels import api as sm
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.stats.outliers_influence import \
    variance_inflation_factor as vif
from typing import Tuple, Union


class RegressionAnalysis(object):
    """
    All the different regression types.
    :method ols_regression: Runs an ordinary least squares regression with
        a number of robustness steps. This includes removing independent
        data with too little data, removing influence points, removing
        collinear independent data, and removing independent data that's
        not significant.
    """

    def __init__(self,
                 indep_in: Union[pd.DataFrame, pd.Series],
                 depen_in: pd.Series,
                 vif_cutoff: float,
                 indep: Union[pd.DataFrame, pd.Series] = None,
                 depen: pd.Series = None,
                 has_const: bool = False,
                 multi_handler: str = 'multi',
                 res: RegressionResults = None,
                 bad_individual_vars: list = None,
                 influence_points: list = None,
                 missing_data_vars: list = None,
                 vif_drop: list = None,
                 pval_drop: list = None,
                 indep_out: Union[pd.DataFrame, pd.Series] = None,
                 depen_out: pd.Series = None):
        """
        :param indep_in: The original independent variable(s) in the
            regression.
        :param depen_in: The original dependent variable in the
            regression.
        :param vif_cutoff: The cutoff for the Variance Inflation Factor.
            Any factor with a VIF above this cutoff will be removed.
        :param indep: The updated independent variable(s) in the
            regression.
        :param depen: The updated dependent variable in the regression.
        :param has_const: Whether the current version of the independent
            data has a constant included.
        :param multi_handler: Helps orchestrate the multifactor
            regression. Can either be 'multi' (for a new multifactor
            regression), 'prescreen' (for when screening individual
            factors in a multifactor regression), 'postscreen' (for when
            running a multifactor regression after individual factors have
            been screened) and 'none' (for when running single factor
            regressions or when pre-screening factors in a multifactor
            regression).
        :param res: The results of fitting a model.
        :param bad_individual_vars: A set of variables that are dropped
            from a multifactor regression when we do single factor
            regressions to pre-screen.
        :param influence_points: Points that are influential to a
            regression. Will record: [the location in the set of
            observations, x-value and y-value] for each.
        :param missing_data_vars: A list of variables that are removed,
            if any, due to having too many missing values.
        :param vif_drop: A list of variables that are removed, if any,
            due to having too high of a Variance Inflation Factor.
        :param pval_drop: A list of variables that are removed, if any,
            due to having too high of a p-value (> 0.05).
        :param indep_out: The independent data that is ultimately used for
            a regression, potentially after changes.
        :param depen_out: The dependent data that is ultimately used for
            a regression, potentially after changes.
        """
        self.indep_in = indep_in
        self.depen_in = depen_in
        self.vif_cutoff = vif_cutoff
        self.indep = indep
        self.depen = depen
        self.has_const = has_const
        self.multi_handler = multi_handler
        self.res = res
        self.bad_individual_vars = bad_individual_vars
        self.influence_points = influence_points
        self.missing_data_vars = missing_data_vars
        self.vif_drop = vif_drop
        self.pval_drop = pval_drop
        self.indep_out = indep_out
        self.depen_out = depen_out

    def _drop_na(self):
        """Drop any rows with missing data in the independent or dependent
        data sets."""
        # combine data
        all_regr_data = pd.concat([self.depen, self.indep], axis=1)

        # remove the rows, reset the independent and dependent data
        all_regr_data = all_regr_data.dropna().reset_index(drop=True)
        self.depen = all_regr_data.iloc[:, 0]
        self.indep = all_regr_data.iloc[:, 1:]

    def _small_data_handling(self) -> str:
        """
        Remove the independent data series with the most missing data.
        :return error_str: If the data is already too small - we
            only have one non-intercept independent variable - return an
            error message. Will also return an error message if we don't
            actually have any missing data. Will return None if there is
            no error.
        """

        # determine the minimum number of independent variables we
        # can have, depending on whether we have a constant
        if self.has_const is True:
            min_num_indep_vars = 2
        else:
            min_num_indep_vars = 1

        # remove independent data with the most missing values as
        # long as we have the intercept plus one additional factor
        if self.indep.shape[1] > min_num_indep_vars:
            # count the number of missing data points in the
            # original independent data
            missing_count = self.indep_in.isna().sum()
            max_loc_missing = np.argmax(missing_count)
            # error out if we don't have any missing data
            if max(missing_count) == 0:
                error_str = \
                    "There are actually no missing data points, " \
                    "so either a) you are starting with less " \
                    "observations than variables times your minimum " \
                    "data factor, or there's an error in the code, " \
                    "which should be reported to the developer."
                return error_str
            else:
                # record any columns we're dropping
                missing_data_col = self.indep_in.columns[max_loc_missing]
                if self.missing_data_vars is None:
                    self.missing_data_vars = [missing_data_col]
                else:
                    self.missing_data_vars.append(missing_data_col)
                # remove the column with the most missing data
                self.indep_in.drop(missing_data_col, axis=1, inplace=True)
                # return nothing so we know to re-run this method
                return None
        else:
            error_str = \
                "There are less data points for this model than the number " \
                "of factors times the min data multiplier, so the model is " \
                "intractable. Consider reducing the number of factors or " \
                "removing factors with large amounts of missing data."
            if self.influence_points is not None:
                error_str = " ".join(["After removing influence points,",
                                      error_str])
            return error_str

    def _reset_vars(self) -> None:
        """
        Reset the class variables. This is necessary in cases where we
            re-start the regression with a new set of independent data,
            for example.
        """
        self.has_const = False
        self.res = None
        self.influence_points = None

    def _small_data_orchestration(self, regr_inputs: dict) -> str:
        """
        Orchestrate the handling of small data sets.
        :param regr_inputs: The set of inputs we need to re-run
            the ols_regression if necessary. These include: data_min,
            missing, robust_covar, remove_influence.
        :return error_str: If there is an error, return the error.
        """
        error_str = self._small_data_handling()
        # either error out or re-run this method
        if error_str is None:
            self._reset_vars()
            error_str = self.ols_regression(
                data_min=regr_inputs['data_min'],
                missing=regr_inputs['missing'],
                add_constant=True,
                robust_covar=regr_inputs['robust_covar'],
                remove_influence=regr_inputs['remove_influence'])
        else:
            return error_str

    def _run_single_regr_in_multi(self, first_non_const_indep: int,
                                  regr_inputs: dict,
                                  multifactor_keep: list) -> Tuple[str, list]:
        """
        Runs single factor regressions for each independent variable in a
            multi-factor regression and then keeps just those that are
            significant.
        :param first_non_const_indep: The index of the first non-constant
            independent variable (1 with intercept, 0 without).
        :param regr_inputs: The set of inputs we need to re-run
            the ols_regression if necessary. These include: data_min,
            missing, robust_covar, remove_influence.
        :param multifactor_keep: Which variables to keep.
        :return error_str: If there is an error, return the error.
        :return multifactor_keep: The list of variables to keep.
        """
        for curr_var in self.indep.columns[first_non_const_indep:]:
            self._reset_vars()
            error_str = self.ols_regression(
                data_min=regr_inputs['data_min'],
                missing=regr_inputs['missing'],
                add_constant=True,
                robust_covar=regr_inputs['robust_covar'],
                remove_influence=regr_inputs['remove_influence'],
                indep_override=self.indep_in[curr_var])

            # check if we want to keep the factor based on it having
            # a p-value of < 0.05
            if error_str is None and \
                    self.res.pvalues[first_non_const_indep] < 0.05:
                multifactor_keep.append(curr_var)
            else:
                self.bad_individual_vars.append(curr_var)

        return error_str, multifactor_keep

    def _run_multi_subset(self, multifactor_keep: list,
                          regr_inputs: dict) -> str:
        """
        Re-set the independent variables and re-run with multi_handler set
            to 'postscreen'
        :param regr_inputs: The set of inputs we need to re-run
            the ols_regression if necessary. These include: data_min,
            missing, robust_covar, remove_influence.
        :param multifactor_keep: Which variables to keep.
        :return error_str: If there is an error, return the error.
        """
        if multifactor_keep:
            self.indep_in = self.indep_in[multifactor_keep]
            self.multi_handler = 'postscreen'
            error_str = self.ols_regression(
                data_min=regr_inputs['data_min'],
                missing=regr_inputs['missing'],
                add_constant=True,
                robust_covar=regr_inputs['robust_covar'],
                remove_influence=regr_inputs['remove_influence'])
        else:
            error_str = ("No individual factors were significant in the "
                         "multifactor regression.")

        return error_str

    def _multi_screen_individual(self, regr_inputs: dict) -> str:
        """
        Deal with the 'multi' section of the multi_handler, which screens
            individual factors and then re-runs the regression with only
            the significant factors.
        :param regr_inputs: The set of inputs we need to re-run
            the ols_regression if necessary. These include: data_min,
            missing, robust_covar, remove_influence.
        :return error_str: If there is an error, return the error.
        """
        # determine if we have a constant
        if self.has_const is True:
            first_non_const_indep = 1
        else:
            first_non_const_indep = 0

        self.multi_handler = 'prescreen'
        multifactor_keep = []
        self.bad_individual_vars = []

        # run the regression for all individual factors
        error_str, multifactor_keep = self._run_single_regr_in_multi(
            first_non_const_indep=first_non_const_indep,
            regr_inputs=regr_inputs, multifactor_keep=multifactor_keep)

        # re-set the independent variables and re-run with
        # multi_handler set to 'postscreen'
        error_str = self._run_multi_subset(multifactor_keep, regr_inputs)

        return error_str

    def _record_influence_points(self, min_loc_p: int) -> None:
        """
        Record the influence points, including their location in the
            original list and their x and y values.
        :param min_loc_p: The location of the influence point.
        """
        # determine if we have a constant
        if self.has_const is True:
            first_non_const_indep = 1
        else:
            first_non_const_indep = 0

        indep_influence = self.indep.iloc[min_loc_p, first_non_const_indep:]
        depen_influence = self.depen.iloc[min_loc_p]

        if self.influence_points is None:
            self.influence_points = [[min_loc_p, indep_influence,
                                      depen_influence]]
        else:
            min_loc_p_influence = min_loc_p
            # if we already removed points earlier in the data than the
            # current influence point, we need to account for that when
            # setting the location
            for metric_list in self.influence_points:
                curr_infl_loc = metric_list[0]
                if curr_infl_loc <= min_loc_p:
                    min_loc_p_influence += 1

            self.influence_points.append([min_loc_p_influence, indep_influence,
                                          depen_influence])

    def _remove_influence_points(self) -> bool:
        """
        Remove influence points from a set of data based on the results
            of a regression, using Cook's Distance.
        :return: Returns True if any points were dropped.
        """

        # check on influential points
        influence = self.res.get_influence()
        cooks_distance, cooks_p = influence.cooks_distance
        # if any, remove the most influential and re-run
        if any(cooks_p < 0.05):
            # find the most influential and record it
            min_loc_p = np.argmin(cooks_p)
            self._record_influence_points(min_loc_p)

            # create the new data
            self.indep = self.indep.drop(min_loc_p).reset_index(drop=True)
            self.depen = self.depen.drop(min_loc_p).reset_index(drop=True)

            return True

        else:
            return False

    def _influence_orchestration(self, regr_inputs: dict) -> str:
        """
        Orchestrate the handling of influence points.
        :param regr_inputs: The set of inputs we need to re-run
            the ols_regression if necessary. These include: data_min,
            missing, add_constant, robust_covar, remove_influence.
        :return error_str: If there is an error, return the error. If not,
            but the multi-handler is in 'prescreen' or 'none' modes,
            return 'return_none' as an indicator to the calling function.
        """
        points_removed = self._remove_influence_points()
        # if we removed any points, re-run
        if points_removed:
            error_str = self.ols_regression(
                data_min=regr_inputs['data_min'],
                missing=regr_inputs['missing'],
                add_constant=False,
                robust_covar=regr_inputs['robust_covar'],
                remove_influence=regr_inputs['remove_influence'],
                indep_override=self.indep,
                depen_override=self.depen)
            return error_str
        elif self.multi_handler == 'prescreen':
            return 'return_none'
        elif self.multi_handler == 'none':
            self.indep_out = self.indep
            self.depen_out = self.depen
            return 'return_none'
        else:
            return None

    def _find_highest_pval_factor(self, first_non_const_indep: int,
                                  vif_data: list) -> int:
        """For factors with high VIF, find the one with the highest
        p-value."""
        indep_pvals = self.res.pvalues[first_non_const_indep:]
        pval_order = np.argsort(indep_pvals)[::-1]
        for curr_index in pval_order:
            if vif_data[curr_index] > self.vif_cutoff:
                return curr_index

    def _handle_high_vif(self, first_non_const_indep: int,
                         vif_data: list, indep_vif: pd.DataFrame,
                         regr_inputs: dict) -> str:
        """
        Remove the factor with the highest p-value in the subset that are
            above the VIF cutoff and re-run the regression.
        :param first_non_const_indep: The index of the first non-constant
            independent variable (1 with intercept, 0 without).
        :param vif_data: The VIF by factor.
        :param indep_vif: The independent variables to look at for VIF,
            which just removes the constant.
        :param regr_inputs: The set of inputs we need to re-run
            the ols_regression if necessary. These include: data_min,
            missing, robust_covar, remove_influence. Add constant here
            will be set to True.
        :return error_str: If there is an error, return the error.
        """
        # find the factor with the highest p-value in the subset of
        # factors with high VIF
        max_loc_vif = self._find_highest_pval_factor(first_non_const_indep,
                                                     vif_data)

        # record any columns we're dropping
        vif_data_col = indep_vif.columns[max_loc_vif]
        if self.vif_drop is None:
            self.vif_drop = [vif_data_col]
        else:
            self.vif_drop.append(vif_data_col)
        # remove the column with the highest VIF
        self.indep_in.drop(vif_data_col, axis=1, inplace=True)

        self._reset_vars()
        error_str = self.ols_regression(
            data_min=regr_inputs['data_min'],
            missing=regr_inputs['missing'],
            add_constant=True,
            robust_covar=regr_inputs['robust_covar'],
            remove_influence=regr_inputs['remove_influence'])

        return error_str

    def _vif_orchestration(self, first_non_const_indep: int,
                           regr_inputs: dict) -> str:
        """
        Orchestrate the handling of VIF (variance inflation factor).
        :param first_non_const_indep: The index of the first non-constant
            independent variable (1 with intercept, 0 without).
        :param regr_inputs: The set of inputs we need to re-run
            the ols_regression if necessary. These include: data_min,
            missing, add_constant, robust_covar, remove_influence.
        :return error_str: If there is an error, return the error.
        """
        # grab all non-intercept factors
        indep_vif = self.indep.iloc[:, first_non_const_indep:]
        # calculate their VIFs
        vif_data = [vif(indep_vif, i) for i in range(len(indep_vif.columns))]

        # remove factors with high VIF
        if max(vif_data) > self.vif_cutoff:
            error_str = self._handle_high_vif(first_non_const_indep,
                                              vif_data, indep_vif, regr_inputs)
        else:
            self.indep_out = self.indep
            self.depen_out = self.depen
            error_str = None

        return error_str

    def _multi_screen_collinearity(self, regr_inputs: dict) -> str:
        """
        Deal with the high collinearity (VIF) between factors as
            part of the 'postscreen' section of the multi_handler.
        :param regr_inputs: The set of inputs we need to re-run
            the ols_regression if necessary. These include: data_min,
            missing, robust_covar, remove_influence.
        :return error_str: If there is an error, return the error.
        """
        # determine if we have a constant
        if self.has_const is True:
            first_non_const_indep = 1
        else:
            first_non_const_indep = 0

        # look at the VIF as long as we have the intercept plus one
        # additional factor
        if self.indep.shape[1] > first_non_const_indep + 1:
            error_str = self._vif_orchestration(first_non_const_indep,
                                                regr_inputs)
        else:
            self.indep_out = self.indep
            self.depen_out = self.depen
            error_str = None

        return error_str

    def _error_too_few_factors(self):
        error_str = ("After removing factors with too little data, factors "
            "that aren't independently significant, factors that are too "
            "highly correlated with other factors and removing the remaining "
            "factors with little significance, no factors remain. "
            "Multifactor regression isn't a perfect science, so if you have "
            "significant individual factors, you should look at those. If "
            "not, it might be that none of the factors you tested are useful.")
        return error_str

    def _drop_insignificant_factor(self, indep_pvals: list,
                                   first_non_const_indep: int,
                                   regr_inputs: dict) -> str:
        """
        Drop the factor with the highest p-value and re-run the
            regression.
        :param indep_pvals: The set of p-values for the independent
            variables in the same order as the variables.
        :param first_non_const_indep: The index of the first non-constant
            independent variable (1 with intercept, 0 without).
        :param regr_inputs: The set of inputs we need to re-run
            the ols_regression if necessary. These include: data_min,
            missing, add_constant, robust_covar, remove_influence.
        :return error_str: If there is an error, return the error.
        """
        # find the location of the highest p-value
        max_loc_pval = np.argmax(indep_pvals)
        # record any columns we're dropping
        pval_data_col = self.indep.columns[
            max_loc_pval + first_non_const_indep]

        if self.pval_drop is None:
            self.pval_drop = [pval_data_col]
        else:
            self.pval_drop.append(pval_data_col)
        # remove the column with the highest p-value
        self.indep_in.drop(pval_data_col, axis=1, inplace=True)

        self._reset_vars()
        error_str = self.ols_regression(
            data_min=regr_inputs['data_min'],
            missing=regr_inputs['missing'],
            add_constant=True,
            robust_covar=regr_inputs['robust_covar'],
            remove_influence=regr_inputs['remove_influence'])

        return error_str

    def _multi_screen_pvals(self, regr_inputs: dict) -> str:
        """
        Deal with the insignificant factors as part of the 'postscreen'
            section of the multi_handler.
        :param regr_inputs: The set of inputs we need to re-run
            the ols_regression if necessary. These include: data_min,
            missing, robust_covar, remove_influence.
        :return error_str: If there is an error, return the error.
        """
        # determine if we have a constant
        if self.has_const is True:
            first_non_const_indep = 1
        else:
            first_non_const_indep = 0

        # look at the p-values
        indep_pvals = self.res.pvalues[first_non_const_indep:]
        # a rule of thumb is to remove factors with p-values < 0.05
        if max(indep_pvals) > 0.05:
            if len(indep_pvals) < first_non_const_indep + 1:
                error_str = self._error_too_few_factors()
            else:
                error_str = self._drop_insignificant_factor(
                    indep_pvals, first_non_const_indep, regr_inputs)
            return error_str
        else:
            self.indep_out = self.indep
            self.depen_out = self.depen
            # return nothing so we know we're done
            return None

    def ols_regression(self,
                       data_min: Union[int, float],
                       missing: str = 'drop',
                       add_constant: bool = True,
                       robust_covar: Union[str, None] = 'HC0',
                       remove_influence: bool = True,
                       indep_override: pd.DataFrame = None,
                       depen_override: pd.DataFrame = None) -> str:
        """
        Runs an ordinary least squares regression with a number of
            robustness steps. This includes removing independent data with
            too little data, removing influence points, removing collinear
            independent data, and removing independent data that's not
            significant.
        :param data_min: A multiple of the number of factors
            that we need as the minimum number of observations. This
            makes it so we don't have too small of a dataset, even
            though it could run technically. For example, if
            data_min=5 and we have 5 factors, we need more than 25
            observations. The comparison here is > so that if we
            have data_min=1, we don't allow the number of factors
            to exactly equal the number of observations.
        :param missing: Passed in due to it being an input we use
            in statsmodels OLS. The options are 'none', 'drop', and
            'raise'. Our default is 'drop' although statsmodels'
            default is 'none'.
        :param add_constant: Whether to add a constant to the independent
            data (it is not added by default in statsmodels).
        :param robust_covar: If we want the results to use a
            covariance matrix robust to any issues, which is a
            method the statsmodels OLS can use, use that here. See
            options here:
            https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html
        :param remove_influence: Whether to remove influential points
            using Cook's Distance.
        :param indep_override: Use this if you want to start
            with indep data other than the class's indep_in data.
        :param depen_override: Use this if you want to start
            with depen data other than the class's depen_in data.
        :return error_str: Error string if the regression can't finish.
        """

        # set up the indep and depen data that can be altered without
        # losing the original inputs
        if indep_override is None:
            self.indep = self.indep_in.copy()
        else:
            self.indep = indep_override
        if depen_override is None:
            self.depen = self.depen_in.copy()
        else:
            self.depen = depen_override

        # return if we have no data (check if indep or depen is empty)
        if self.indep.empty:
            error_str = ("There is no independent data to run a regression "
                         "on. Please ensure that you have enough data to run "
                         "a regression.")
            return error_str
        if self.depen.empty:
            error_str = ("There is no dependent data to run a regression "
                         "on. Please ensure that you have enough data to run "
                         "a regression.")
            return error_str

        # drop missing if desired
        if missing == 'drop':
            self._drop_na()
        if self.indep.empty:
            error_str = ("After dropping missing rows, there is no data "
                         "left to run a regression on. Please ensure that "
                         "you have enough data to run a regression.")
            return error_str

        # add an intercept
        if add_constant is True:
            self.indep = sm.add_constant(self.indep)
            self.has_const = True

        # set up the inputs into a dict to be used throughout
        regr_inputs = {'data_min': data_min,
                       'missing': missing,
                       'robust_covar': robust_covar,
                       'remove_influence': remove_influence}

        # error handle if we don't have enough data
        if (self.indep.shape[1] * data_min) > self.indep.shape[0]:
            error_str = self._small_data_orchestration(regr_inputs)

        # control for multifactor regressions
        # (all multifactor, single factor and single factor as
        # pre-screening for multifactor should run the above methods)
        # here we screen individual factors and then re-run the
        # regression with only the significant factors
        if self.multi_handler == 'multi':
            error_str = self._multi_screen_individual(regr_inputs)

        # fit the model
        mod = sm.OLS(endog=self.depen, exog=self.indep, missing=missing)
        self.res = mod.fit()

        # use the robust covariance if given
        if robust_covar is not None:
            self.res = self.res.get_robustcov_results(robust_covar)

        # remove influential points if desired
        if remove_influence:
            error_str = self._influence_orchestration(regr_inputs)
            if error_str == 'return_none':
                return None

        # deal with collinearity for a multifactor regression
        if self.multi_handler == 'postscreen':
            error_str = self._multi_screen_collinearity(regr_inputs)

        # deal with stepping down for a multifactor regression
        # we do this by removing the factor with the highest p-value
        # as long as it's above 0.05
        if self.multi_handler == 'postscreen':
            error_str = self._multi_screen_pvals(regr_inputs)
