import unittest
import pandas as pd
from pydts.examples_utils.generate_simulations_data import generate_quick_start_df
from pydts.fitters import TwoStagesFitter
import numpy as np


class TestTwoStagesFitter(unittest.TestCase):

    def setUp(self):
        self.real_coef_dict = {
            "alpha": {
                1: lambda t: -1 - 0.3 * np.log(t),
                2: lambda t: -1.75 - 0.15 * np.log(t)
            },
            "beta": {
                1: -np.log([0.8, 3, 3, 2.5, 2]),
                2: -np.log([1, 3, 4, 3, 2])
            }
        }
        self.df = generate_quick_start_df(n_patients=5000, n_cov=5, d_times=10, j_events=2, pid_col='pid', seed=0,
                                          real_coef_dict=real_coef_dict)
        self.m = TwoStagesFitter()
        self.fitted_model = TwoStagesFitter()
        self.fitted_model.fit(df=self.df.drop(['C', 'T'], axis=1))

    def test_fit_case_event_col_not_in_df(self):
        # Event column (here 'J') must be passed in df to .fit()
        with self.assertRaises(AssertionError):
            self.m.fit(df=self.df.drop(['C', 'J', 'T'], axis=1))

    def test_fit_case_duration_col_not_in_df(self):
        # Duration column (here 'X') must be passed in df to .fit()
        with self.assertRaises(AssertionError):
            self.m.fit(df=self.df.drop(['C', 'X', 'T'], axis=1))

    def test_fit_case_pid_col_not_in_df(self):
        # Duration column (here 'pid') must be passed in df to .fit()
        with self.assertRaises(AssertionError):
            self.m.fit(df=self.df.drop(['C', 'pid', 'T'], axis=1))

    def test_fit_case_cov_col_not_in_df(self):
        # Covariates columns (here ['Z1','Z2','Z3','Z4','Z5']) must be passed in df to .fit()
        with self.assertRaises(ValueError):
            self.m.fit(df=self.df.drop(['C', 'T'], axis=1), covariates=['Z6'])

    def test_fit_case_missing_jt(self):
        # df to .fit() should contain observations to all (event, times)

        # drop (events[1], times[2])
        tmp_df = self.df[
            ~((self.df[self.fitted_model.duration_col] == self.fitted_model.times[2]) &
              (self.df[self.fitted_model.event_type_col] == self.fitted_model.events[1]))
        ]

        with self.assertRaises(RuntimeError):
            self.m.fit(df=tmp_df.drop(['C', 'T'], axis=1))

    def test_fit_case_correct_fit(self):
        # Fit should be successful
        m = TwoStagesFitter()
        m.fit(df=self.df.drop(['C', 'T'], axis=1))

    def test_print_summary(self):
        self.fitted_model.print_summary()

    def test_plot_event_alpha_case_correct_event(self):
        self.fitted_model.plot_event_alpha(event=self.fitted_model.events[0], show=False)

    def test_plot_event_alpha_case_incorrect_event(self):
        with self.assertRaises(AssertionError):
            self.fitted_model.plot_event_alpha(event=100, show=False)

    def test_plot_all_events_alpha(self):
        self.fitted_model.plot_all_events_alpha(show=False)

    def test_get_beta_SE(self):
        self.fitted_model.get_beta_SE()

    def test_plot_all_events_beta(self):
        self.fitted_model.plot_all_events_beta(show=False)

    def test_predict_hazard_jt_case_covariate_not_in_df(self):
        # Covariates columns used in fit (here ['Z1','Z2','Z3','Z4','Z5']) must be passed in df to .predict()
        with self.assertRaises(AssertionError):
            self.fitted_model.predict_hazard_jt(
                df=self.df.drop(['C', 'T', 'Z1'], axis=1),
                event=self.fitted_model.events[0],
                t=self.fitted_model.times[0])

    def test_predict_hazard_jt_case_event_not_in_events(self):
        # Event passed to .predict() must be in fitted events
        with self.assertRaises(AssertionError):
            self.fitted_model.predict_hazard_jt(
                df=self.df.drop(['C', 'T'], axis=1), event=100, t=self.fitted_model.times[0])

    def test_predict_hazard_jt_case_time_not_in_times(self):
        # Event passed to .predict() must be in fitted events
        with self.assertRaises(AssertionError):
            self.fitted_model.predict_hazard_jt(
                df=self.df.drop(['C', 'T'], axis=1), event=self.fitted_model.events[0], t=1000)

    def test_predict_hazard_jt_case_successful_predict(self):
        self.fitted_model.predict_hazard_jt(
            df=self.df.drop(['C', 'T'], axis=1),
            event=self.fitted_model.events[0], t=self.fitted_model.times[0])

    def test_predict_hazard_t_case_successful_predict(self):
        self.fitted_model.predict_hazard_t(df=self.df.drop(['C', 'T'], axis=1), t=self.fitted_model.times[:3])

    def test_predict_hazard_all_case_successful_predict(self):
        self.fitted_model.predict_hazard_all(df=self.df.drop(['C', 'T'], axis=1))

    def test_predict_overall_survival_case_successful_predict(self):
        self.fitted_model.predict_overall_survival(
            df=self.df.drop(['C', 'T'], axis=1), t=self.fitted_model.times[5], return_hazards=True)

    def test_predict_prob_event_j_at_t_case_successful_predict(self):
        self.fitted_model.predict_prob_event_j_at_t(df=self.df.drop(['C', 'T'], axis=1),
                                                    event=self.fitted_model.events[0],
                                                    t=self.fitted_model.times[3])

    def test_predict_prob_event_j_all_case_successful_predict(self):
        self.fitted_model.predict_prob_event_j_all(df=self.df.drop(['C', 'T'], axis=1),
                                                   event=self.fitted_model.events[0])

    def test_predict_prob_events_case_successful_predict(self):
        self.fitted_model.predict_prob_events(df=self.df.drop(['C', 'T'], axis=1))

    def test_predict_event_cumulative_incident_function_case_successful_predict(self):
        self.fitted_model.predict_event_cumulative_incident_function(df=self.df.drop(['C', 'T'], axis=1),
                                                                     event=self.fitted_model.events[0])

    def test_predict_cumulative_incident_function_case_successful_predict(self):
        self.fitted_model.predict_cumulative_incident_function(df=self.df.drop(['C', 'T'], axis=1))


if __name__ == '__main__':
    unittest.main()
