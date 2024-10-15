###################################################
#  PyDTS: A Python Package for Discrete-Time Survival Analysis with Competing Risks (2022)
#  Meir, Tomer*, Gutman, Rom*, and Gorfine, Malka.
###################################################
# Replication file


from time import time
from pydts.data_generation import EventTimesSampler
from pydts.examples_utils.plots import add_panel_text, plot_events_occurrence, plot_example_pred_output, \
    plot_models_coefficients, plot_times, compare_beta_models_for_example, plot_jss_reps_coef_std, \
    plot_sampled_covariates_figure
from pydts.examples_utils.generate_simulations_data import generate_quick_start_df
from pydts.fitters import DataExpansionFitter, TwoStagesFitter, repetitive_fitters
from pydts.model_selection import PenaltyGridSearch
from pydts.cross_validation import PenaltyGridSearchCV, TwoStagesCV
from pydts.examples_utils.mimic_consts import *
from pydts.evaluation import *
from tableone import TableOne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

slicer = pd.IndexSlice

pd.set_option("display.max_rows", 500)
warnings.filterwarnings('ignore')
nb_start = time()


# The figures and tables of the paper were generated with lean_version = False.
# Estimated total running time (Apple MacBook Pro 32Gb RAM):
# lean_version = True: 3.5 hours
# lean_version = False: 11 hours without MIMIC, 17.5 hours with MIMIC
lean_version = False

# Optional settings:
# 1. Provide the MIMIC-IV v2.0 data directory (mimic-iv-2.0) to replicate the use-case section results.
# Leaving mimic_data_dir = None skips the replication of Section 5.
# For example:  mimic_data_dir = '/Users/tomer/data/mimic-iv-2.0'
# The MIMIC dataset is accessible at https://physionet.org/content/mimiciv/2.0/ and subjected to PhysioNet credentials.
mimic_data_dir = None

# 2. Figures\tables output directory
OUTPUT_DIR = ''

# 3. Show figures using plt.show():
PYPLOT_SHOW = True




# Section 3: Simulating discrete time survival data with competing events
############################################################################

ets = EventTimesSampler(d_times=7, j_event_types=2)
n_observations = 10000
np.random.seed(0)

observations_df = pd.DataFrame(columns=['Z1', 'Z2', 'Z3'])
observations_df['Z1'] = np.random.binomial(n=1, p=0.5, size=n_observations)
Z1_zero_index = observations_df.loc[observations_df['Z1'] == 0].index
observations_df.loc[Z1_zero_index, 'Z2'] = \
    np.random.normal(loc=72, scale=12, size=n_observations-observations_df['Z1'].sum())
Z1_one_index = observations_df.loc[observations_df['Z1'] == 1].index
observations_df.loc[Z1_one_index, 'Z2'] = \
    np.random.normal(loc=82, scale=12, size=observations_df['Z1'].sum())
observations_df['Z3'] = 1 + np.random.poisson(lam=4, size=n_observations)


# Figure 1
plot_sampled_covariates_figure(observations_df, fname=os.path.join(OUTPUT_DIR, 'figure_1.png'),
                               show=PYPLOT_SHOW)

prob_lof_at_t = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
observations_df = ets.sample_independent_lof_censoring(observations_df, prob_lof_at_t)

ets = EventTimesSampler(d_times=7, j_event_types=2)
censoring_coef_dict = {
    "alpha": {
        0: lambda t: -0.3 - 0.3 * np.log(t),
    },
    "beta": {
        0: -np.log([8, 0.95, 6]),
    }
}

observations_df = ets.sample_hazard_lof_censoring(observations_df, censoring_coef_dict)

coefficients_dict = {
    "alpha": {
        1: lambda t: -1 - 0.3 * np.log(t),
        2: lambda t: -1.75 - 0.15 * np.log(t)
    },
    "beta": {
        1: -np.log([0.8, 1.4, 3]),
        2: -np.log([1, 0.95, 2])
    }
}

observations_df = ets.sample_event_times(observations_df, coefficients_dict)
observations_df = ets.update_event_or_lof(observations_df)


# Table 2
tmp = observations_df.astype({'Z1': int, 'Z2': float, 'Z3': int, 'X': int,
                        'C': int, 'J': int, 'T': int}).round(2).head()[['Z1', 'Z2', 'Z3', 'T', 'C', 'X', 'J']]
print(tmp)
tmp.to_csv(os.path.join(OUTPUT_DIR, 'table_2.csv'))


# Section 4 - Examples
########################

real_coef_dict = {
    "alpha": {
        1: lambda t: -1 - 0.3 * np.log(t),
        2: lambda t: -1.75 - 0.15 * np.log(t)
    },
    "beta": {
        1: -np.log([0.8, 3, 3, 2.5, 2]),
        2: -np.log([1, 3, 4, 3, 2])
    }
}

patients_df = generate_quick_start_df(n_patients=50000, n_cov=5, d_times=30, j_events=2,
                                      pid_col='pid', seed=0, censoring_prob=0.8,
                                      real_coef_dict=real_coef_dict)

# Figure 2
plot_events_occurrence(patients_df, fname=os.path.join(OUTPUT_DIR, 'figure_2.png'), show=PYPLOT_SHOW)

print(patients_df.groupby(['J', 'X'])['pid'].count().unstack('J'))


# Estimation using Lee et al. 2018
print('Estimating Lee et al.')
fitter = DataExpansionFitter()
fitter.fit(df=patients_df.drop(['C', 'T'], axis=1))
fitter.print_summary()

summary = fitter.event_models[1].summary()
summary_df = pd.DataFrame([x.split(',') for x in summary.tables[1].as_csv().split('\n')])
summary_df.columns = summary_df.iloc[0]
summary_df = summary_df.iloc[1:].set_index(summary_df.columns[0])
lee_beta1_summary = summary_df.iloc[-5:]


summary = fitter.event_models[2].summary()
summary_df = pd.DataFrame([x.split(',') for x in summary.tables[1].as_csv().split('\n')])
summary_df.columns = summary_df.iloc[0]
summary_df = summary_df.iloc[1:].set_index(summary_df.columns[0])
lee_beta2_summary = summary_df.iloc[-5:]


# Prediction example
pred_df = fitter.predict_cumulative_incident_function(
    patients_df.drop(['J', 'T', 'C', 'X'], axis=1).head(3)).set_index('pid').T
pred_df.index.name = ''
pred_df.columns = ['ID=0', 'ID=1', 'ID=2']


print(pred_df)

# Two-step example
print('Estimating two-step')
new_fitter = TwoStagesFitter()
new_fitter.fit(df=patients_df.drop(['C', 'T'], axis=1))
new_fitter.print_summary()

print(new_fitter.get_beta_SE())

twostep_beta_summary = new_fitter.get_beta_SE()

twostep_beta1_summary = twostep_beta_summary.iloc[:, [0, 1]]
twostep_beta2_summary = twostep_beta_summary.iloc[:, [2, 3]]

lee_beta1_summary = lee_beta1_summary.iloc[:, [0, 1]].round(3)
lee_beta2_summary = lee_beta2_summary.iloc[:, [0, 1]].round(3)
lee_beta1_summary.columns = pd.MultiIndex.from_tuples([('Lee et al.', 'Estimate'), ('Lee et al.', 'SE')])
lee_beta2_summary.columns = pd.MultiIndex.from_tuples([('Lee et al.', 'Estimate'), ('Lee et al.', 'SE')])
beta_summary_comparison = pd.concat([lee_beta1_summary, lee_beta2_summary], axis=0)
beta_summary_comparison.index = [r'$\beta_{11}$', r'$\beta_{12}$', r'$\beta_{13}$', r'$\beta_{14}$', r'$\beta_{15}$',
                                 r'$\beta_{21}$', r'$\beta_{22}$', r'$\beta_{23}$', r'$\beta_{24}$', r'$\beta_{25}$']
twostep_beta1_summary.columns = pd.MultiIndex.from_tuples([('two-step', 'Estimate'), ('two-step', 'SE')])
twostep_beta2_summary.columns = pd.MultiIndex.from_tuples([('two-step', 'Estimate'), ('two-step', 'SE')])
tmp = pd.concat([twostep_beta1_summary.round(3), twostep_beta2_summary.round(3)], axis=0)
tmp.index = [r'$\beta_{11}$', r'$\beta_{12}$', r'$\beta_{13}$', r'$\beta_{14}$', r'$\beta_{15}$',
             r'$\beta_{21}$', r'$\beta_{22}$', r'$\beta_{23}$', r'$\beta_{24}$', r'$\beta_{25}$']
beta_summary_comparison = pd.concat([beta_summary_comparison, tmp], axis=1)
beta_summary_comparison.index.name =  r'$\beta_{jk}$'

true_col = -np.log([0.8, 3, 3, 2.5, 2, 1, 3, 4, 3, 2])
beta_summary_comparison.insert(loc=0, column='True', value=true_col)
beta_summary_comparison.astype(float).round(3).to_csv(os.path.join(OUTPUT_DIR, 'table_3.csv'))
#print(beta_comparison_table.to_latex(escape=False))

print(beta_summary_comparison)

pred_df = new_fitter.predict_cumulative_incident_function(
    patients_df.drop(['J', 'T', 'C', 'X'], axis=1).head(3)).set_index('pid').T
pred_df.index.name = ''
pred_df.columns = ['ID = 0', 'ID = 1', 'ID = 2']


# Figure 4
plot_example_pred_output(pred_df, fname=os.path.join(OUTPUT_DIR, 'figure_4.png'), show=plot_events_occurrence)


# Single build single run rep dict
rep_dict = {}
counts_df = patients_df[patients_df['X'] <= 30].groupby(['J', 'X']).size().to_frame(0)
rep_dict[0] = compare_beta_models_for_example(fitter.event_models,
                                              new_fitter.event_models, real_coef_dict=real_coef_dict)

# Results comparison of a single run
new_res_dict = plot_jss_reps_coef_std(rep_dict, True)
a = new_res_dict['alpha']
b = new_res_dict['beta']
times = [t+1 for t in list(a[1].reset_index().index)]

# Figure 3
plot_models_coefficients(a, b, times, counts_df, filename=os.path.join(OUTPUT_DIR, 'figure_3.png'), show=PYPLOT_SHOW)

# Results comparison of 100 repetitions (2 repetitions of lean_version = True)
print('Results comparison multiple runs')
rep = 2 if lean_version else 100
rep_dict, times_dict, counts_df = repetitive_fitters(rep=rep, n_patients=50000, n_cov=5,
                                                     d_times=30, j_events=2, pid_col='pid',
                                                     verbose=0, real_coef_dict=real_coef_dict, censoring_prob=0.8,
                                                     allow_fails=20)

# Figure 6
new_res_dict = plot_jss_reps_coef_std(rep_dict, True, filename=os.path.join(OUTPUT_DIR, 'figure_6.png'),
                                      show=PYPLOT_SHOW)
a = new_res_dict['alpha']
b = new_res_dict['beta']
times = [t+1 for t in list(a[1].reset_index().index)]

# Figure 5
plot_models_coefficients(a, b, times, counts_df, filename=os.path.join(OUTPUT_DIR, 'figure_5.png'),
                         show=PYPLOT_SHOW)

# Table 4
beta_comparison_table = pd.DataFrame(index=['True', 'Mean (Lee et al.)', 'SE (Lee et al.)',
                                            'Mean (two-step)', 'SE (two-step)'])
for j in [1, 2]:
    for i in range(1, 6):
        tmp_df = pd.DataFrame()
        for idx, k in enumerate(sorted(rep_dict.keys())):
            lee = rep_dict[k]['beta'][j].loc[f"Z{i}_{j}"]['Lee']
            ours = rep_dict[k]['beta'][j].loc[f"Z{i}_{j}"]['Ours']
            true = rep_dict[k]['beta'][j].loc[f"Z{i}_{j}"]['real']
            row = pd.Series({'True': true, 'Lee': lee, 'Ours': ours}, name=f"Z{i}_{j}_{k}")
            tmp_df = pd.concat([tmp_df, row], axis=1)
        beta_row = pd.Series({
            'True': tmp_df.iloc[0,0],
            'Mean (Lee et al.)': tmp_df.iloc[1].mean(),
            'SE (Lee et al.)': tmp_df.iloc[1].std(),
            'Mean (two-step)': tmp_df.iloc[2].mean(),
            'SE (two-step)': tmp_df.iloc[2].std()
        }, name=f'Z{i}_{j}')
        beta_comparison_table = pd.concat([beta_comparison_table, beta_row], axis=1)

beta_comparison_table = beta_comparison_table.round(3).T
beta_comparison_table.columns = pd.MultiIndex.from_tuples(
    [('True', ''), ('Lee et al.', 'Estimate'), ('Lee et al.', 'SE'), ('two-step', 'Estimate'), ('two-step', 'SE')])
beta_comparison_table.index = [r'$\beta_{11}$', r'$\beta_{12}$', r'$\beta_{13}$', r'$\beta_{14}$', r'$\beta_{15}$',
                               r'$\beta_{21}$', r'$\beta_{22}$', r'$\beta_{23}$', r'$\beta_{24}$', r'$\beta_{25}$']
beta_comparison_table.index.name = r'$\beta_{jk}$'

print(beta_comparison_table)
beta_comparison_table.astype(float).round(3).to_csv(os.path.join(OUTPUT_DIR, 'table_4.csv'))
#print(beta_comparison_table.to_latex(escape=False))


# Figure 7
colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:cyan', 'tab:purple']
_times_dict = {}
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
rep = 3 if lean_version else 10
for idc, d in enumerate([15, 30, 45, 60, 100]):
    print(f'Estimating timing: d={d}')
    rep_dict, times_dict, counts_df = repetitive_fitters(rep=rep, n_patients=50000, n_cov=5,
                                                         d_times=d, j_events=2, pid_col='pid',
                                                         verbose=0, real_coef_dict=real_coef_dict, censoring_prob=0.8,
                                                         allow_fails=20)
    _times_dict['Lee et al.'] = times_dict['Lee']
    _times_dict['two-step'] = times_dict['Ours']
    ax = plot_times(_times_dict, ax=ax, color=colors[idc], show=False)

ax.grid(which='minor', alpha=0.4)
ax.grid(which='major', alpha=0.7)
labels = ['d=15', 'd=30', 'd=45', 'd=60', 'd=100']
ax.legend(labels=labels)
leg = ax.get_legend()
for i in range(len(labels)):
    leg.legendHandles[i].set_color(colors[i])
fig.tight_layout()
if PYPLOT_SHOW:
    plt.show()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure_7.png'), dpi=300)


# Section 4.1: Regularization
L1_regularized_fitter = TwoStagesFitter()

fit_beta_kwargs = {
    'model_kwargs': {
        1: {'penalizer': 0.003, 'l1_ratio': 1},
        2: {'penalizer': 0.005, 'l1_ratio': 1}
}}

L1_regularized_fitter.fit(df=patients_df.drop(['C', 'T'], axis=1), fit_beta_kwargs=fit_beta_kwargs)


L1_regularized_fitter = TwoStagesFitter()
fit_beta_kwargs = {
    'model_kwargs': {
        1: {'penalizer': np.array([0.01, 0.01, 0.01, 0.01, 0]),
            'l1_ratio': 1},
        2: {'penalizer': np.array([0.05, 0.05, 0.05, 0.05, 0]),
            'l1_ratio': 1}
}}
L1_regularized_fitter.fit(df=patients_df.drop(['C', 'T'], axis=1),
                          fit_beta_kwargs=fit_beta_kwargs)


train_df, test_df = train_test_split(patients_df, train_size=0.8, random_state=1)
penalizers = np.exp([-2, -3, -4, -5, -6])
grid_search = PenaltyGridSearch()
optimal_set = grid_search.evaluate(train_df, test_df, l1_ratio=1,
                                   penalizers=penalizers,
                                   metrics = ['IBS', 'GBS', 'IAUC', 'GAUC'])

print(np.log(optimal_set))
print(np.log(grid_search.convert_results_dict_to_df(grid_search.global_bs).idxmin().values[0]))


grid_search_cv = PenaltyGridSearchCV()
results_df = grid_search_cv.cross_validate(patients_df, l1_ratio=1,
                                penalizers=penalizers, n_splits=5,
                                metrics=['IBS', 'GBS', 'IAUC', 'GAUC'])

print(results_df['Mean'].idxmax())

results_df_ = results_df.copy().reset_index()
results_df_.iloc[:, :2] = np.log(results_df_.iloc[:, :2])
results_df_ = results_df_.set_index(['level_0', 'level_1'])
results_df_.index.set_names(['log(eta_1)', 'log(eta_2)'], inplace=True)
print(results_df_)

results_df_.to_csv(os.path.join(OUTPUT_DIR, 'cv_results.csv'))


# Section 4.2: Performance measures

fitter = TwoStagesFitter()
fitter.fit(df=train_df.drop(['C', 'T'], axis=1))
pred_df = fitter.predict_prob_event_j_all(test_df.drop(['C', 'T'], axis=1), event=1)
auc_1 = event_specific_auc_at_t_all(pred_df, event=1)

pred_df = fitter.predict_prob_events(test_df)
ibs_1 = event_specific_integrated_brier_score(pred_df, event=1)
iauc_1 = event_specific_integrated_auc(pred_df, event=1)
bs = global_brier_score(pred_df)
auc = global_auc(pred_df)

cross_validator = TwoStagesCV()
cross_validator.cross_validate(full_df=patients_df.drop(['C', 'T'],
                               axis=1),
                               n_splits=5, seed=0,
                               metrics=['BS', 'IBS', 'GBS',
                                        'AUC', 'IAUC', 'GAUC'])


# Section 4.3: Data regrouping

df = generate_quick_start_df(n_patients=1000, n_cov=5, d_times=30, j_events=2, pid_col='pid', seed=0,
                             real_coef_dict=real_coef_dict)

regrouped_df = df.copy()
regrouped_df['X'].clip(upper=21, inplace=True)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
ax = axes[0]
ax = plot_events_occurrence(df, ax=ax)
add_panel_text(ax, 'a')
ax = axes[1]
ax = plot_events_occurrence(regrouped_df, ax=ax)
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[-1] = '21+'
ax.set_xticklabels(labels)
add_panel_text(ax, 'b')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure_8.png'), dpi=300)
if PYPLOT_SHOW:
    plt.show()



# Section 5 - Use case
# MIMIC-IV v2.0 data is required

if mimic_data_dir is not None:
    patients_file = os.path.join(mimic_data_dir, 'hosp', 'patients.csv.gz')
    admissions_file = os.path.join(mimic_data_dir, 'hosp', 'admissions.csv.gz')
    lab_file = os.path.join(mimic_data_dir, 'hosp', 'labevents.csv.gz')
    lab_meta_file = os.path.join(mimic_data_dir, 'hosp', 'd_labitems.csv.gz')

    patients_df = pd.read_csv(patients_file, compression='gzip')
    COLUMNS_TO_DROP = ['dod']
    patients_df.drop(COLUMNS_TO_DROP, axis=1, inplace=True)

    admissions_df = pd.read_csv(admissions_file, compression='gzip', parse_dates=[ADMISSION_TIME_COL,
                                                                                  DISCHARGE_TIME_COL, DEATH_TIME_COL,
                                                                                  ED_REG_TIME, ED_OUT_TIME])

    COLUMNS_TO_DROP = ['hospital_expire_flag', 'edouttime', 'edregtime', 'deathtime', 'language']
    admissions_df.drop(COLUMNS_TO_DROP, axis=1, inplace=True)

    admissions_df = admissions_df.merge(patients_df, on=[SUBJECT_ID_COL])

    # # Calculate Age at Admission and Group of Admission Year
    # Based on mimic IV example https://mimic.mit.edu/docs/iv/modules/hosp/patients/

    # Diff column first
    admissions_df[ADMISSION_YEAR_COL] = (admissions_df[ADMISSION_TIME_COL].dt.year - admissions_df['anchor_year'])

    # Age at admission calculation
    admissions_df[ADMISSION_AGE_COL] = (admissions_df[AGE_COL] + admissions_df[ADMISSION_YEAR_COL])

    # Admission year group lower bound calculation
    admissions_df[ADMISSION_YEAR_COL] = admissions_df[ADMISSION_YEAR_COL] + admissions_df[YEAR_GROUP_COL].apply(
        lambda x: int(x.split(' ')[0]))

    # # Calculating LOS (exact, days resolution) and night admission indicator
    admissions_df[LOS_EXACT_COL] = (admissions_df[DISCHARGE_TIME_COL] - admissions_df[ADMISSION_TIME_COL])
    admissions_df[NIGHT_ADMISSION_FLAG] = ((admissions_df[ADMISSION_TIME_COL].dt.hour >= 20) | \
                                           (admissions_df[ADMISSION_TIME_COL].dt.hour < 8)).values
    admissions_df[LOS_DAYS_COL] = admissions_df[LOS_EXACT_COL].dt.ceil('1d')

    max_clip_days = 28

    # # Taking only SPECIFIC_ADMISSION_TYPE admissions from now on
    SPECIFIC_ADMISSION_TYPE = ['DIRECT EMER.', 'EW EMER.']
    admissions_df = admissions_df[admissions_df[ADMISSION_TYPE_COL].isin(SPECIFIC_ADMISSION_TYPE)]

    # # Add direct emergency flag
    admissions_df[DIRECT_IND_COL] = (admissions_df[ADMISSION_TYPE_COL] == 'DIRECT EMER.').astype(int)

    # # Counting SPECIFIC_ADMISSION_TYPE admissions to each patient
    number_of_admissions = admissions_df.groupby(SUBJECT_ID_COL)[ADMISSION_ID_COL].nunique()
    number_of_admissions.name = ADMISSION_COUNT_COL

    admissions_df = admissions_df.merge(number_of_admissions, on=SUBJECT_ID_COL)

    # # Add recurrent admissions group per patient according to last admission
    ADMISSION_COUNT_BINS = [1, 1.5, 2.5, 5000]
    ADMISSION_COUNT_LABELS = ['1', '2', '3up']

    admissions_df[ADMISSION_COUNT_GROUP_COL] = pd.cut(admissions_df[ADMISSION_COUNT_COL],
                                                      bins=ADMISSION_COUNT_BINS,
                                                      labels=ADMISSION_COUNT_LABELS,
                                                      include_lowest=True)

    # # Adds last admission with previous admission in past month indicator
    indicator_diff = pd.to_timedelta('30d')

    tmp_admissions = admissions_df[admissions_df[ADMISSION_COUNT_COL] > 1]
    ind_ser = tmp_admissions.sort_values(by=[SUBJECT_ID_COL, ADMISSION_TIME_COL]).groupby(
        SUBJECT_ID_COL).apply(
        lambda tmp_df: (tmp_df[ADMISSION_TIME_COL] - tmp_df[DISCHARGE_TIME_COL].shift(1)) <= indicator_diff)

    ind_ser.index = ind_ser.index.droplevel(1)
    ind_ser.name = PREV_ADMISSION_IND_COL
    ind_ser = ind_ser.iloc[ind_ser.reset_index().drop_duplicates(subset=[SUBJECT_ID_COL], keep='last').index]

    admissions_df = admissions_df.merge(ind_ser.astype(int), left_on=SUBJECT_ID_COL, right_index=True, how='outer')
    admissions_df[PREV_ADMISSION_IND_COL].fillna(0, inplace=True)


    # # Keep only last admission per patient
    only_last_admission = admissions_df.sort_values(by=[ADMISSION_TIME_COL]).drop_duplicates(subset=[SUBJECT_ID_COL],
                                                                                             keep='last')

    # # Take only patients with last admission after MINIMUM YEAR
    MINIMUM_YEAR = 2014
    only_last_admission = only_last_admission[only_last_admission[ADMISSION_YEAR_COL] >= MINIMUM_YEAR]

    pids = only_last_admission[SUBJECT_ID_COL].drop_duplicates()
    adm_ids = only_last_admission[ADMISSION_ID_COL].drop_duplicates()

    # # Load relevant lab tests
    LOAD_SPECIFIC_COLUMNS = [SUBJECT_ID_COL, ADMISSION_ID_COL, ITEM_ID_COL, 'storetime', 'flag']
    chunksize = 10 ** 6
    full_df = pd.DataFrame()
    with pd.read_csv(lab_file, chunksize=chunksize, compression='gzip', parse_dates=[STORE_TIME_COL],
                     usecols=LOAD_SPECIFIC_COLUMNS) as reader:
        for chunk in reader:
            tmp_chunk = chunk[chunk[SUBJECT_ID_COL].isin(pids) & chunk[ADMISSION_ID_COL].isin(adm_ids)]
            tmp_adms = only_last_admission[
                only_last_admission[SUBJECT_ID_COL].isin(pids) & only_last_admission[ADMISSION_ID_COL].isin(adm_ids)]
            # tmp_patinets = patients_df[patients_df[SUBJECT_ID_COL].isin(pids)]
            tmp_chunk = tmp_chunk.merge(tmp_adms, on=[SUBJECT_ID_COL, ADMISSION_ID_COL])
            # tmp = tmp_chunk.merge(tmp_patinets, on=[SUBJECT_ID_COL])
            full_df = pd.concat([full_df, tmp_chunk])

    # # Continue only with included patients_df and admissions_df and full_df
    pids = full_df[SUBJECT_ID_COL].drop_duplicates().values
    adms_ids = full_df[ADMISSION_ID_COL].drop_duplicates().values
    patients_df = patients_df[patients_df[SUBJECT_ID_COL].isin(pids)]
    admissions_df = admissions_df[admissions_df[ADMISSION_ID_COL].isin(adms_ids)]

    # # Regrouping discharge location
    discharge_regrouping_df = pd.Series(DISCHARGE_REGROUPING_DICT).to_frame()
    discharge_regrouping_df.index.name = 'Original Group'
    discharge_regrouping_df.columns = ['Regrouped']
    admissions_df[DISCHARGE_LOCATION_COL].replace(DISCHARGE_REGROUPING_DICT, inplace=True)
    full_df[DISCHARGE_LOCATION_COL].replace(DISCHARGE_REGROUPING_DICT, inplace=True)


    # # Regroup Race
    race_regrouping_df = pd.Series(RACE_REGROUPING_DICT).to_frame()
    race_regrouping_df.index.name = 'Original Group'
    race_regrouping_df.columns = ['Regrouped']
    admissions_df[RACE_COL].replace(RACE_REGROUPING_DICT, inplace=True)
    full_df[RACE_COL].replace(RACE_REGROUPING_DICT, inplace=True)


    # # Taking only results 24 hours from admission
    full_df[ADMISSION_TO_RESULT_COL] = (full_df[STORE_TIME_COL] - full_df[ADMISSION_TIME_COL])
    full_df = full_df[full_df[ADMISSION_TO_RESULT_COL] <= pd.to_timedelta('1d')]

    full_df.sort_values(by=[ADMISSION_TIME_COL, STORE_TIME_COL]).drop_duplicates(
        subset=[SUBJECT_ID_COL, ADMISSION_ID_COL, ITEM_ID_COL],
        inplace=True, keep='last')

    # # Most common lab tests upon arrival
    lab_meta_df = pd.read_csv(lab_meta_file, compression='gzip')
    threshold = 25000

    common_tests = full_df.groupby(ITEM_ID_COL)[ADMISSION_ID_COL].nunique().sort_values(ascending=False)
    included_in_threshold = common_tests[common_tests > threshold].to_frame().merge(lab_meta_df, on=ITEM_ID_COL)
    full_df = full_df[full_df[ITEM_ID_COL].isin(included_in_threshold[ITEM_ID_COL].values)]
    minimal_item_id = included_in_threshold.iloc[-1][ITEM_ID_COL]

    pids = full_df[full_df[ITEM_ID_COL] == minimal_item_id][SUBJECT_ID_COL].drop_duplicates().values
    adms_ids = full_df[full_df[ITEM_ID_COL] == minimal_item_id][ADMISSION_ID_COL].drop_duplicates().values
    patients_df = patients_df[patients_df[SUBJECT_ID_COL].isin(pids)]
    admissions_df = admissions_df[admissions_df[ADMISSION_ID_COL].isin(adms_ids)]
    full_df = full_df[full_df[SUBJECT_ID_COL].isin(pids)]
    full_df = full_df[full_df[ADMISSION_ID_COL].isin(adms_ids)]

    full_df['flag'].fillna('normal', inplace=True)
    full_df['flag'].replace({'normal': 0, 'abnormal': 1}, inplace=True)
    full_df['flag'].value_counts()

    full_df = full_df.sort_values(by=[ADMISSION_TIME_COL, STORE_TIME_COL]).drop_duplicates(
        subset=[SUBJECT_ID_COL, ADMISSION_ID_COL, ITEM_ID_COL],
        keep='last')

    tmp = full_df[[SUBJECT_ID_COL, ADMISSION_ID_COL, ITEM_ID_COL, 'flag']]
    fitters_table = pd.pivot_table(tmp, values=['flag'], index=[SUBJECT_ID_COL, ADMISSION_ID_COL],
                                   columns=[ITEM_ID_COL], aggfunc=np.sum)

    fitters_table = fitters_table.droplevel(1, axis=0).droplevel(0, axis=1)
    dummies_df = full_df.drop_duplicates(subset=[SUBJECT_ID_COL]).set_index(SUBJECT_ID_COL)

    del full_df
    del admissions_df
    del patients_df


    # # Standardize age
    scaler = StandardScaler()
    dummies_df[STANDARDIZED_AGE_COL] = scaler.fit_transform(dummies_df[[AGE_COL]])

    J_DICT = {'HOME': 1, 'FURTHER TREATMENT': 2, 'DIED': 3, 'CENSORED': 0}
    GENDER_DICT = {'F': 1, 'M': 0}

    dummies_df[GENDER_COL] = dummies_df[GENDER_COL].replace(GENDER_DICT)


    included_in_threshold['label'] = included_in_threshold['label'].apply(lambda x: x.replace(' ', '')).apply(
        lambda x: x.replace(',', ''))
    RENAME_ITEMS_DICT = included_in_threshold[[ITEM_ID_COL, 'label']].set_index(ITEM_ID_COL).to_dict()['label']

    table1 = pd.concat([
        fitters_table.copy(),
        dummies_df[[NIGHT_ADMISSION_FLAG,
                    GENDER_COL,
                    DIRECT_IND_COL,
                    PREV_ADMISSION_IND_COL,
                    ADMISSION_AGE_COL]].astype(int),
        dummies_df[[INSURANCE_COL,
                    MARITAL_STATUS_COL,
                    RACE_COL,
                    ADMISSION_COUNT_GROUP_COL]],
        dummies_df[LOS_DAYS_COL].dt.days,
        dummies_df[DISCHARGE_LOCATION_COL].dropna().replace(J_DICT).astype(int)
    ], axis=1)

    table1.rename(RENAME_ITEMS_DICT, inplace=True, axis=1)
    table1.dropna(inplace=True)

    # %%
    ADMINISTRATIVE_CENSORING = 28

    censoring_index = table1[table1[LOS_DAYS_COL] > ADMINISTRATIVE_CENSORING].index
    table1.loc[censoring_index, DISCHARGE_LOCATION_COL] = 0
    table1.loc[censoring_index, LOS_DAYS_COL] = ADMINISTRATIVE_CENSORING + 1
    table1[GENDER_COL].replace(table1_rename_sex, inplace=True)
    table1[RACE_COL].replace(table1_rename_race, inplace=True)
    table1[MARITAL_STATUS_COL].replace(table1_rename_marital, inplace=True)
    table1[DIRECT_IND_COL].replace(table1_rename_yes_no, inplace=True)
    table1[NIGHT_ADMISSION_FLAG].replace(table1_rename_yes_no, inplace=True)
    table1[PREV_ADMISSION_IND_COL].replace(table1_rename_yes_no, inplace=True)
    table1[DISCHARGE_LOCATION_COL].replace(table1_rename_discharge, inplace=True)
    table1[ADMISSION_COUNT_GROUP_COL].replace({'3up': '3+'}, inplace=True)
    table1.rename(table1_rename_columns, inplace=True, axis=1)

    # %%
    columns = ['gender', 'admission_age', 'race', 'insurance', 'marital_status',
               'direct_emrgency_flag', 'night_admission', 'last_less_than_diff',
               'admissions_count_group', 'LOS days', 'discharge_location']
    columns = [table1_rename_columns[c] for c in columns]
    categorical = ['gender', 'race', 'insurance', 'marital_status',
                   'direct_emrgency_flag', 'night_admission', 'last_less_than_diff',
                   'admissions_count_group', 'discharge_location']
    categorical = [table1_rename_columns[c] for c in categorical]
    table1.dropna(inplace=True)
    groupby = [table1_rename_columns[DISCHARGE_LOCATION_COL]]
    mytable = TableOne(table1, columns, categorical, groupby, missing=False)

    # Patients' characteristics
    print(mytable.tableone.round(3).to_latex())

    columns = [DISCHARGE_LOCATION_COL, 'AnionGap', 'Bicarbonate', 'CalciumTotal', 'Chloride', 'Creatinine',
               'Glucose', 'Magnesium', 'Phosphate', 'Potassium', 'Sodium',
               'UreaNitrogen', 'Hematocrit', 'Hemoglobin', 'MCH', 'MCHC', 'MCV',
               'PlateletCount', 'RDW', 'RedBloodCells', 'WhiteBloodCells']
    categorical = [DISCHARGE_LOCATION_COL, 'AnionGap', 'Bicarbonate', 'CalciumTotal', 'Chloride', 'Creatinine',
                   'Glucose', 'Magnesium', 'Phosphate', 'Potassium', 'Sodium',
                   'UreaNitrogen', 'Hematocrit', 'Hemoglobin', 'MCH', 'MCHC', 'MCV',
                   'PlateletCount', 'RDW', 'RedBloodCells', 'WhiteBloodCells']
    columns = [table1_rename_columns[c] for c in columns]
    categorical = [table1_rename_columns[c] for c in categorical]
    groupby = [table1_rename_columns[DISCHARGE_LOCATION_COL]]
    mytable = TableOne(table1.dropna().replace(table1_rename_normal_abnormal), columns, categorical, groupby,
                       missing=False)

    # Patients' lab tests
    print(mytable.tableone.round(3).to_latex())

    # %%
    fitters_table = pd.concat([
        fitters_table.copy(),
        pd.get_dummies(dummies_df[INSURANCE_COL], prefix='Insurance', drop_first=True),
        pd.get_dummies(dummies_df[MARITAL_STATUS_COL], prefix='Marital', drop_first=True),
        pd.get_dummies(dummies_df[RACE_COL], prefix='Ethnicity', drop_first=True),
        pd.get_dummies(dummies_df[ADMISSION_COUNT_GROUP_COL], prefix='AdmsCount', drop_first=True),
        dummies_df[[NIGHT_ADMISSION_FLAG,
                    GENDER_COL,
                    DIRECT_IND_COL,
                    PREV_ADMISSION_IND_COL]].astype(int),
        dummies_df[STANDARDIZED_AGE_COL],
        dummies_df[LOS_DAYS_COL].dt.days,
        dummies_df[DISCHARGE_LOCATION_COL].dropna().replace(J_DICT).astype(int)
    ], axis=1)


    fitters_table.dropna(inplace=True)
    fitters_table = fitters_table[fitters_table.index.isin(table1.index)]

    fitters_table.reset_index(inplace=True)
    fitters_table.rename({DISCHARGE_LOCATION_COL: 'J', LOS_DAYS_COL: 'X', SUBJECT_ID_COL: 'pid'}, inplace=True, axis=1)
    fitters_table.rename(RENAME_ITEMS_DICT, inplace=True, axis=1)

    # %%
    fitters_table = fitters_table[fitters_table['X'] > 0]
    fitters_table.loc[fitters_table.X > ADMINISTRATIVE_CENSORING, 'J'] = 0
    fitters_table.loc[fitters_table.X > ADMINISTRATIVE_CENSORING, 'X'] = ADMINISTRATIVE_CENSORING + 1
    fitters_table['J'] = fitters_table['J'].astype(int)


    # Grid search for LASSO tuning parameters
    step = 3 if lean_version else 1
    penalizers = np.arange(-12, -0.9, step=step)
    n_splits = 4
    seed = 1

    penalty_cv_search = PenaltyGridSearchCV()
    gauc_cv_results = penalty_cv_search.cross_validate(full_df=fitters_table, l1_ratio=1, penalizers=np.exp(penalizers),
                                                       n_splits=n_splits, seed=seed)

    print(gauc_cv_results['Mean'].max())
    print(gauc_cv_results['Mean'].idxmax())
    print(np.log(gauc_cv_results['Mean'].idxmax()))
    chosen_eta = np.log(gauc_cv_results['Mean'].idxmax())

    res = [v for k, v in penalty_cv_search.integrated_auc.items()]
    pd.concat(res, axis=1, keys=list(range(1, 1 + len(res)))).mean(axis=1, level=1)

    # No regularization cross validation
    cross_validator_null = TwoStagesCV()
    cross_validator_null.cross_validate(full_df=fitters_table, n_splits=n_splits, seed=seed, nb_workers=1)

    np.mean(list(cross_validator_null.global_auc.values())).round(3)
    np.std(list(cross_validator_null.global_auc.values())).round(3)

    # Final regularized fitter
    reg_fitter = TwoStagesCV()
    fit_beta_kwargs = {
        'model_kwargs': {
            1: {'penalizer': np.exp(chosen_eta[0]), 'l1_ratio': 1},
            2: {'penalizer': np.exp(chosen_eta[1]), 'l1_ratio': 1},
            3: {'penalizer': np.exp(chosen_eta[2]), 'l1_ratio': 1},
        }
    }
    reg_fitter.cross_validate(fitters_table, n_splits=n_splits, seed=seed, nb_workers=1,
                              fit_beta_kwargs=fit_beta_kwargs)

    np.mean(list(reg_fitter.global_auc.values())).round(3), np.std(list(reg_fitter.global_auc.values())).round(3)

    pd.DataFrame.from_records(reg_fitter.integrated_auc).mean(axis=1).round(3), pd.DataFrame.from_records(
        reg_fitter.integrated_auc).std(axis=1).round(3)

    np.mean(list(reg_fitter.global_bs.values())).round(3), np.std(list(reg_fitter.global_bs.values())).round(3)

    pd.DataFrame.from_records(reg_fitter.integrated_bs).mean(axis=1).round(3), pd.DataFrame.from_records(
        reg_fitter.integrated_bs).std(axis=1).round(3)

    case = f'mimic_final_'
    two_step_timing = []
    lee_timing = []

    # Two step fitter
    new_fitter = TwoStagesFitter()
    print(f'Starting two-step')
    two_step_start = time()
    new_fitter.fit(df=fitters_table, nb_workers=1)
    two_step_end = time()
    print(f'Finished two-step: {two_step_end - two_step_start}sec')

    two_step_timing.append(two_step_end - two_step_start)

    # Lee et al fitter
    print(f'MIMIC analysis - Starting Lee et al.')
    lee_fitter = DataExpansionFitter()
    lee_start = time()
    lee_fitter.fit(df=fitters_table)
    lee_end = time()
    print(f'MIMIC analysis - Finished lee: {lee_end - lee_start}sec')

    lee_timing.append(lee_end - lee_start)

    # Regularized Two step fitter
    reg_fitter = TwoStagesFitter()
    print(f'MIMIC analysis - Starting regularized two-step')
    fit_beta_kwargs = {
        'model_kwargs': {
            1: {'penalizer': np.exp(chosen_eta[0]), 'l1_ratio': 1},
            2: {'penalizer': np.exp(chosen_eta[1]), 'l1_ratio': 1},
            3: {'penalizer': np.exp(chosen_eta[2]), 'l1_ratio': 1},
        }
    }
    reg_two_step_start = time()
    reg_fitter.fit(df=fitters_table, nb_workers=1, fit_beta_kwargs=fit_beta_kwargs)
    reg_two_step_end = time()
    print(f'MIMIC analysis - Finished two-step: {reg_two_step_end - reg_two_step_start}sec')

    lee_alpha_ser = lee_fitter.get_alpha_df().loc[:, slicer[:, [COEF_COL, STDERR_COL]]].unstack().sort_index()
    lee_beta_ser = lee_fitter.get_beta_SE().loc[:, slicer[:, [COEF_COL, STDERR_COL]]].unstack().sort_index()

    two_step_alpha_k_results = new_fitter.alpha_df[['J', 'X', 'alpha_jt']]
    two_step_beta_k_results = new_fitter.get_beta_SE().unstack().to_frame()

    reg_two_step_alpha_k_results = reg_fitter.alpha_df[['J', 'X', 'alpha_jt']]
    reg_two_step_beta_k_results = reg_fitter.get_beta_SE().unstack().to_frame()

    lee_alpha_k_results = lee_alpha_ser.to_frame()
    lee_beta_k_results = lee_beta_ser.to_frame()

    # Cache results
    two_step_alpha_k_results.to_csv(os.path.join(OUTPUT_DIR, f'{case}_two_step_alpha.csv'))
    two_step_beta_k_results.to_csv(os.path.join(OUTPUT_DIR, f'{case}_two_step_beta.csv'))
    reg_two_step_alpha_k_results.to_csv(os.path.join(OUTPUT_DIR, f'{case}_reg_two_step_alpha.csv'))
    reg_two_step_beta_k_results.to_csv(os.path.join(OUTPUT_DIR, f'{case}_reg_two_step_beta.csv'))
    lee_alpha_k_results.to_csv(os.path.join(OUTPUT_DIR, f'{case}_lee_alpha.csv'))
    lee_beta_k_results.to_csv(os.path.join(OUTPUT_DIR, f'{case}_lee_beta.csv'))


    covariates = [c for c in fitters_table.columns if c not in ['pid', 'J', 'X']]

    two_step_alpha_k_results = pd.read_csv(os.path.join(OUTPUT_DIR, f'{case}_two_step_alpha.csv'),
                                           index_col=['J', 'X'])
    two_step_beta_k_results = pd.read_csv(os.path.join(OUTPUT_DIR, f'{case}_two_step_beta.csv'),
                                          index_col=[0, 1])
    reg_two_step_alpha_k_results = pd.read_csv(os.path.join(OUTPUT_DIR, f'{case}_reg_two_step_alpha.csv'),
                                               index_col=['J', 'X'])
    reg_two_step_beta_k_results = pd.read_csv(os.path.join(OUTPUT_DIR, f'{case}_reg_two_step_beta.csv'),
                                              index_col=[0, 1])
    lee_alpha_k_results = pd.read_csv(os.path.join(OUTPUT_DIR, f'{case}_lee_alpha.csv'),
                                      index_col=[0, 1, 2])
    lee_beta_k_results = pd.read_csv(os.path.join(OUTPUT_DIR, f'{case}_lee_beta.csv'),
                                     index_col=[0, 1, 2])

    twostep_beta1_summary = two_step_beta_k_results.mean(axis=1).unstack([0]).round(3).iloc[:, [1, 0]]
    twostep_beta1_summary.index = [f'{iii.replace(" ", "")}_1' for iii in twostep_beta1_summary.index]
    twostep_beta2_summary = two_step_beta_k_results.mean(axis=1).unstack([0]).round(3).iloc[:, [3, 2]]
    twostep_beta2_summary.index = [f'{iii.replace(" ", "")}_2' for iii in twostep_beta2_summary.index]
    twostep_beta3_summary = two_step_beta_k_results.mean(axis=1).unstack([0]).round(3).iloc[:, [5, 4]]
    twostep_beta3_summary.index = [f'{iii.replace(" ", "")}_3' for iii in twostep_beta3_summary.index]

    reg_twostep_beta1_summary = reg_two_step_beta_k_results.mean(axis=1).unstack([0]).round(3).iloc[:, [1, 0]]
    reg_twostep_beta1_summary.index = [f'{iii.replace(" ", "")}_1' for iii in reg_twostep_beta1_summary.index]
    reg_twostep_beta2_summary = reg_two_step_beta_k_results.mean(axis=1).unstack([0]).round(3).iloc[:, [3, 2]]
    reg_twostep_beta2_summary.index = [f'{iii.replace(" ", "")}_2' for iii in reg_twostep_beta2_summary.index]
    reg_twostep_beta3_summary = reg_two_step_beta_k_results.mean(axis=1).unstack([0]).round(3).iloc[:, [5, 4]]
    reg_twostep_beta3_summary.index = [f'{iii.replace(" ", "")}_3' for iii in reg_twostep_beta3_summary.index]

    lee_beta1_summary = lee_beta_k_results.mean(axis=1).loc[slicer[1, :, :]].unstack([0]).round(3)
    lee_beta1_summary.index = [f'{iii.replace(" ", "")}_1' for iii in lee_beta1_summary.index]
    lee_beta2_summary = lee_beta_k_results.mean(axis=1).loc[slicer[2, :, :]].unstack([0]).round(3)
    lee_beta2_summary.index = [f'{iii.replace(" ", "")}_2' for iii in lee_beta2_summary.index]
    lee_beta3_summary = lee_beta_k_results.mean(axis=1).loc[slicer[3, :, :]].unstack([0]).round(3)
    lee_beta3_summary.index = [f'{iii.replace(" ", "")}_3' for iii in lee_beta3_summary.index]

    lee_beta1_summary.columns = pd.MultiIndex.from_tuples([('Lee et al.', 'Estimate'), ('Lee et al.', 'SE')])
    lee_beta2_summary.columns = pd.MultiIndex.from_tuples([('Lee et al.', 'Estimate'), ('Lee et al.', 'SE')])
    lee_beta3_summary.columns = pd.MultiIndex.from_tuples([('Lee et al.', 'Estimate'), ('Lee et al.', 'SE')])

    beta_summary_comparison = pd.concat([lee_beta1_summary, lee_beta2_summary, lee_beta3_summary], axis=0)

    twostep_beta1_summary.columns = pd.MultiIndex.from_tuples([('Two-Step', 'Estimate'), ('Two-Step', 'SE')])
    twostep_beta2_summary.columns = pd.MultiIndex.from_tuples([('Two-Step', 'Estimate'), ('Two-Step', 'SE')])
    twostep_beta3_summary.columns = pd.MultiIndex.from_tuples([('Two-Step', 'Estimate'), ('Two-Step', 'SE')])

    reg_twostep_beta1_summary.columns = pd.MultiIndex.from_tuples(
        [('Two-Step & LASSO', 'Estimate'), ('Two-Step & LASSO', 'SE')])
    reg_twostep_beta2_summary.columns = pd.MultiIndex.from_tuples(
        [('Two-Step & LASSO', 'Estimate'), ('Two-Step & LASSO', 'SE')])
    reg_twostep_beta3_summary.columns = pd.MultiIndex.from_tuples(
        [('Two-Step & LASSO', 'Estimate'), ('Two-Step & LASSO', 'SE')])

    tmp = pd.concat([twostep_beta1_summary.round(3), twostep_beta2_summary.round(3), twostep_beta3_summary.round(3)],
                    axis=0)
    tmp2 = pd.concat(
        [reg_twostep_beta1_summary.round(3), reg_twostep_beta2_summary.round(3), reg_twostep_beta3_summary.round(3)],
        axis=0)

    beta_summary_comparison = pd.concat([beta_summary_comparison, tmp, tmp2], axis=1)
    beta_summary_comparison.index.name = r'$\beta_{jk}$'
    beta_summary_comparison.index = [c.replace("_", " ") for c in beta_summary_comparison.index]

    beta_summary_comparison[('Lee et al.', 'Estimate (SE)')] = ['{:.3f} ({:.3f})'.format(x, y) for x, y in
                                                                beta_summary_comparison[[('Lee et al.', 'Estimate'),
                                                                                         ('Lee et al.', 'SE')]].values]
    beta_summary_comparison[('Two-Step', 'Estimate (SE)')] = ['{:.3f} ({:.3f})'.format(x, y) for x, y in
                                                              beta_summary_comparison[[('Two-Step', 'Estimate'),
                                                                                       ('Two-Step', 'SE')]].values]
    beta_summary_comparison[('Two-Step & LASSO', 'Estimate (SE)')] = ['{:.3f} ({:.3f})'.format(x, y) for x, y in
                                                                      beta_summary_comparison[
                                                                          [('Two-Step & LASSO', 'Estimate'),
                                                                           ('Two-Step & LASSO', 'SE')]].values]

    beta_summary_comparison = beta_summary_comparison[
        [('Lee et al.', 'Estimate (SE)'), ('Two-Step', 'Estimate (SE)'), ('Two-Step & LASSO', 'Estimate (SE)')]]

    risk1_rename_index_dict = {k + f' 1': v for k, v in rename_beta_index.items()}
    risk1 = beta_summary_comparison.iloc[:int(len(beta_summary_comparison) // 3)].rename(risk1_rename_index_dict,
                                                                                         axis=0).sort_index()
    risk1 = risk1.merge(pd.Series(beta_units, name=('', '')), left_index=True, right_index=True,
                        how='outer').sort_index(axis=1)
    print(risk1)
    risk1.to_csv(os.path.join(OUTPUT_DIR, 'table_5_MIMIC_beta_risk_1.csv'))
    #print(risk1.to_latex(escape=False))


    risk2_rename_index_dict = {k + f' 2': v for k, v in rename_beta_index.items()}
    risk2 = beta_summary_comparison.iloc[
            int(len(beta_summary_comparison) // 3):2 * (int(len(beta_summary_comparison) // 3))].rename(
        risk2_rename_index_dict, axis=0).sort_index()
    risk2 = risk2.merge(pd.Series(beta_units, name=('', '')), left_index=True, right_index=True,
                        how='outer').sort_index(axis=1)
    risk2.to_csv(os.path.join(OUTPUT_DIR, 'table_6_MIMIC_beta_risk_2.csv'))
    print(risk2)
    #print(risk2.to_latex(escape=False))

    risk3_rename_index_dict = {k + f' 3': v for k, v in rename_beta_index.items()}
    risk3 = beta_summary_comparison.iloc[2 * int(len(beta_summary_comparison) // 3):].rename(risk3_rename_index_dict,
                                                                                             axis=0).sort_index()
    risk3 = risk3.merge(pd.Series(beta_units, name=('', '')), left_index=True, right_index=True,
                        how='outer').sort_index(axis=1)

    risk3.to_csv(os.path.join(OUTPUT_DIR, 'table_7_MIMIC_beta_risk_3.csv'))
    print(risk3)
    #print(risk3.to_latex(escape=False))


    lof_censoring = (100 * len(
        fitters_table[(fitters_table['J'] == 0) & (fitters_table['X'] <= ADMINISTRATIVE_CENSORING)]) / len(
        fitters_table))
    adm_censoring = (100 * len(
        fitters_table[(fitters_table['J'] == 0) & (fitters_table['X'] > ADMINISTRATIVE_CENSORING)]) / len(
        fitters_table))
    risks = (100 * fitters_table.groupby(['J']).size() / fitters_table.groupby('J').size().sum()).round(1)
    print(f'Total number of patients: {len(fitters_table)}')
    print(
        f"LOF censoring: {lof_censoring:.1f}%, Administrative censoring: {adm_censoring:.1f}%, Home: {risks.loc[1]}%, Further treatment: {risks.loc[2]}%, Death: {risks.loc[3]}%")

    # %%
    chosen_auc_df = pd.DataFrame()
    for i_fold in range(n_splits):
        mixed_two_step = penalty_cv_search.folds_grids[i_fold].get_mixed_two_stages_fitter(np.exp(chosen_eta))
        test_df = fitters_table[fitters_table['pid'].isin(penalty_cv_search.test_pids[i_fold])]
        pred_df = mixed_two_step.predict_prob_events(test_df)
        auc_t = events_auc_at_t(pred_df)
        chosen_auc_df = pd.concat([chosen_auc_df, pd.concat([auc_t], keys=[i_fold])])

    counts = fitters_table.groupby(['J', 'X'])['pid'].count().unstack('J').fillna(0)


    ticksize = 15
    axes_title_fontsize = 17
    legend_size = 13

    risk_names = ['Home', 'Further Treatment', 'Death']
    risk_colors = ['tab:blue', 'tab:green', 'tab:red']
    abc_letters = ['a', 'b', 'c']
    def_letters = ['d', 'e', 'f']
    ghi_letters = ['g', 'h', 'i']

    fig, axes = plt.subplots(3, 3, figsize=(20, 17))

    for risk in [1, 2, 3]:
        nonzero_count = pd.DataFrame(index=list(range(n_splits)), columns=penalizers)
        for idp, penalizer in enumerate(penalizers):

            tmp_j1_params_df = pd.DataFrame()
            for i_fold in range(n_splits):
                params_ser = penalty_cv_search.folds_grids[i_fold].meta_models[np.exp(penalizer)].beta_models[
                    risk].params_
                nonzero_count.loc[i_fold, penalizer] = (params_ser.round(3).abs() > 0).sum()
                tmp_j1_params_df = pd.concat([tmp_j1_params_df, params_ser], axis=1)

            ser_1 = tmp_j1_params_df.mean(axis=1)
            ser_1.name = penalizer

            if idp == 0:
                j1_params_df = ser_1.to_frame()
            else:
                j1_params_df = pd.concat([j1_params_df, ser_1], axis=1)

        ax = axes[0, risk - 1]
        add_panel_text(ax, abc_letters[risk - 1])
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        ax.tick_params(axis='both', which='minor', labelsize=ticksize)
        ax.set_xlabel(fr'Log ($\eta_{risk}$)', fontsize=axes_title_fontsize)
        ax.set_ylabel(f'Number of Non-Zero Coefficients', fontsize=axes_title_fontsize)
        ax.set_title(rf'$\beta_{risk}$ - {risk_names[risk - 1]}', fontsize=axes_title_fontsize)
        ax.axvline(chosen_eta[risk - 1], color=risk_colors[risk - 1], alpha=1, ls='--', lw=1,
                   label=rf'Chosen $Log (\eta_{risk})$')
        ax.set_ylim([0, 40])

        for idp, penalizer in enumerate(penalizers):

            count = nonzero_count[penalizer].mean()
            if idp == 0:
                ax.scatter(penalizer, count, color=risk_colors[risk - 1], alpha=0.8, marker='P', label=f'4-Fold mean')
            else:
                ax.scatter(penalizer, count, color=risk_colors[risk - 1], alpha=0.8, marker='P')
            if penalizer == chosen_eta[risk - 1]:
                print(f"Risk {risk}: {count} non-zero coefficients at chosen eta {chosen_eta[risk - 1]}")

        ax.legend(fontsize=legend_size)

        ax = axes[1, risk - 1]
        add_panel_text(ax, def_letters[risk - 1])
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        ax.tick_params(axis='both', which='minor', labelsize=ticksize)
        for i in range(len(j1_params_df)):
            ax.plot(penalizers, j1_params_df.iloc[i].values, lw=1)

            if i == 0:
                ax.set_ylabel(f'{n_splits}-Fold Mean Coefficient Value', fontsize=axes_title_fontsize)
                ax.set_xlabel(fr'Log ($\eta_{risk}$)', fontsize=axes_title_fontsize)
                ax.set_title(rf'$\beta_{risk}$ - {risk_names[risk - 1]}', fontsize=axes_title_fontsize)
                ax.axvline(chosen_eta[risk - 1], color=risk_colors[risk - 1], alpha=1, ls='--', lw=1)

        ax = axes[2, risk - 1]
        add_panel_text(ax, ghi_letters[risk - 1])
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        ax.tick_params(axis='both', which='minor', labelsize=ticksize)
        mean_auc = chosen_auc_df.loc[slicer[:, risk], :].mean(axis=0)
        std_auc = chosen_auc_df.loc[slicer[:, risk], :].std(axis=0)
        ax.errorbar(mean_auc.index, mean_auc.values, yerr=std_auc.values, fmt="o", color=risk_colors[risk - 1],
                    alpha=0.8)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_yticklabels([c.round(1) for c in np.arange(0, 1.1, 0.1)])
        ax.set_xlabel(r'Time', fontsize=axes_title_fontsize)
        ax.set_ylabel(f'AUC (t)', fontsize=axes_title_fontsize)
        ax.set_title(fr'{risk_names[risk - 1]}, Log ($\eta_{risk}$) = {chosen_eta[risk - 1]}',
                     fontsize=axes_title_fontsize)
        ax.set_ylim([0, 1])
        ax.axhline(0.5, ls='--', color='k', alpha=0.3)
        ax2 = ax.twinx()
        ax2.bar(counts.index, counts[risk].values.squeeze(), color=risk_colors[risk - 1], alpha=0.8, width=0.4)
        ax2.set_ylabel('Number of observed events', fontsize=axes_title_fontsize, color=risk_colors[risk - 1])
        ax2.tick_params(axis='y', colors=risk_colors[risk - 1])
        ax2.set_ylim([0, 5100])
        ax2.tick_params(axis='both', which='major', labelsize=ticksize)
        ax2.tick_params(axis='both', which='minor', labelsize=ticksize)

    fig.tight_layout()

    fig.savefig(os.path.join(OUTPUT_DIR, 'figure_10.png'), dpi=300)
    if PYPLOT_SHOW:
        plt.show()

    first_model_name = 'Lee et al.'
    second_model_name = 'two-step'
    third_model_name = 'Regularized two-step'
    times = range(1, ADMINISTRATIVE_CENSORING + 1)

    lee_colors = ['tab:blue', 'tab:green', 'tab:red']
    two_step_colors = ['navy', 'darkgreen', 'tab:brown']
    reg_two_step_colors = ['darkviolet', 'olive', 'maroon']
    true_colors = ['tab:blue', 'tab:green', 'tab:red']

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    two_step_alpha_k_results = pd.read_csv(os.path.join(OUTPUT_DIR, f'{case}_two_step_alpha.csv'),
                                           index_col=['J', 'X'])
    lee_alpha_k_results = pd.read_csv(os.path.join(OUTPUT_DIR, f'{case}_lee_alpha.csv'),
                                      index_col=[0, 1, 2])

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    tmp_j1_params_df = pd.DataFrame()
    for i_fold in range(n_splits):
        mixed_two_step = penalty_cv_search.folds_grids[i_fold].get_mixed_two_stages_fitter(np.exp(chosen_eta))
        tmp_j1_params_df = pd.concat([tmp_j1_params_df, mixed_two_step.alpha_df.set_index(['J', 'X'])['alpha_jt']],
                                     axis=1)

    ser_1 = tmp_j1_params_df.mean(axis=1)
    ser_1.name = penalizer

    for j in [1, 2, 3]:
        tmp_alpha = lee_alpha_k_results.loc[slicer[j, COEF_COL, :]].mean(axis=1)
        tmp_alpha.index = [int(idx.split(')[')[1].split(']')[0]) for idx in tmp_alpha.index]
        tmp_alpha = pd.Series(tmp_alpha.values.squeeze().astype(float), index=tmp_alpha.index)

        ax.scatter(tmp_alpha.index, tmp_alpha.values,
                   label=f'J={j} ({first_model_name})', color=lee_colors[j - 1], marker='o', alpha=0.3, s=40)

        tmp_alpha = two_step_alpha_k_results.loc[slicer[j, 'alpha_jt']]
        ax.scatter(tmp_alpha.index, tmp_alpha.values,
                   label=f'J={j} ({second_model_name})', color=two_step_colors[j - 1], marker='*', alpha=0.7, s=20)

        ax.scatter(range(1, ADMINISTRATIVE_CENSORING + 1), ser_1.loc[slicer[j, :]].values,
                   label=f'J={j} ({third_model_name})', color=reg_two_step_colors[j - 1], marker='>', alpha=0.7, s=20)

        ax.set_xlabel(r'Time', fontsize=18)
        ax.set_ylabel(r'$\alpha_{jt}$', fontsize=18)
        ax.legend(loc='upper right', fontsize=12)

    ax.set_ylim([-13, 4.5])

    ax2 = ax.twinx()
    ax2.bar(counts.index, counts[1].values.squeeze(), label='J=1', color='navy', alpha=0.4, width=0.4)
    ax2.bar(counts.index, counts[2].values.squeeze(), label='J=2', color='darkgreen', alpha=0.4, align='edge',
            width=0.4)
    ax2.bar(counts.index, counts[3].values.squeeze(), label='J=3', color='tab:red', alpha=0.6, align='edge',
            width=-0.4)
    ax2.legend(loc='upper center', fontsize=12)
    ax2.set_ylabel('Number of observed events', fontsize=16, color='red')
    ax2.tick_params(axis='y', colors='red')
    ax2.set_ylim([0, 8500])
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='minor', labelsize=15)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figure_9.png'), dpi=300)
    if PYPLOT_SHOW:
        plt.show()
nb_end = time()

print(f'Total running time: {int(nb_end-nb_start)} seconds.')
