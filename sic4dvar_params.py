"""
SIC4DVAR-LC
Copyright (C) 2025 INRAE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

@authors:   Callum TYLER, callum.tyler@inrae.fr
            Dylan QUITTARD, dylan.quittard@inrae.fr
                All functions not mentioned above.
            Cécile Cazals, cecile.cazals@cs-soprasteria.com
                logger

Description:
    Parameters file for sic4dvar (algo315).
"""
from pathlib import Path
import numpy as np
import multiprocessing
from datetime import datetime
reach_type_to_process = {'seq': [1, 3, 4, 5], 'set': [1]}
algo_bounds = [[5.0], [-10.05, -0.05, -0.25], [10.0, 60.0, 5.0]]
algo_bounds[2][2] = (algo_bounds[2][1] - algo_bounds[2][0]) / 20.0
slope_smoothing_num_passes = 0
approx_section_params = [8, 0.15, 2]
kDim = 101
shape03 = 10.0
val1 = -1.0
shape13 = 2.1
val2 = 28.6
shape23 = 2.1
known_friction_shape = 100.0
local_QM1 = 1.0
cython_version = False
min_num_nodes = 1
reachNB = 0
extrapolation = True
Option = 2
LSMX = 5
LSMT = 1
corx = 0.2 * 1000
cort = 1 * 6 * 3600.0
DX_Length = 20 * 1000.0
DX_max_in = np.inf
DX_max_out = 500.0
valid_min_z = -1000.0
valid_min_dA = -10000000.0
figures = False
node_run = False
DT_obs = 1200.0
node_length = True
create_tables = False
opt_sword_boost = False
large_deviations = True
old_extrapolation = False
a31_early_stop = True
force_specific_dates = False
start_date = datetime(2023, 3, 31)
end_date = datetime(2025, 5, 5)
replace_config = True
config_file_path = '/app/sic4dvar_param_confluence.ini'
densification = False
num_cores = multiprocessing.cpu_count()
max_cores = int(num_cores / 2)
no_print = False
q_mean_computed = False
override_cort = False
smooth_estimate = True
uniform_friction = False
qsdev_activate = False
qsdev_option = 1
indicators_experimental_computation = True
use_fallback_sos = False
use_dynamic_spread = False
use_mean_for_bounds = True
relative_variance = 1.0
auto_spread_for_runs = False
smoothing_keep_drops = False
corx_option = 0
drop_threshold = 0.3 / 200
ML_time_series_path = '/mnt/DATA/worksync/files/data/ML_dataframes/ML_time_series_new.nc'
q_prior_from_ML = False
ML_priors_force_quit = True
run_preprocessing = True
run_extrapolation = True
day_length = 1000000.0
experimental_stop_at_bathy = False
experimental_discharge_ML = False
ml_stations_sync = False
experimental_run_MMI = False
smoothing_prior = False
reference_force_quit = False
use_SVS = False
SVS_file = ''
pooling = False
static_slope = True
slope_smooth_wse_ranking = False
use_zb_early_break = False
slope_smooth_max_iter = 1
svs_reach_id = 'v17b'
start_from_downstream = False
use_gauge_external_file = False
gauge_external_file = ''
use_friction_external_file = False
friction_external_file = ''
tmp_write_to_A0_and_n = False
optional_outputs = False
create_series_from_monthly_q = True
algo3_displace_series = True
remake_zb_table = False
use_flow_duration_q = True
mapping_to_wse_obs = False
use_series_cv = True
monthly_mask = 'days'
correction_type = 'mult'
discharge_correction = True
fill_nan_with_neighbours = False
final_smoothing_discharge = False
exclude_invalid_overpasses = False
regularization_profile = True
compute_mean_elevation = True
save_quantiles_densification = True
use_quantiles_for_densification = False
dA_from_mike_method = False
force_compute_dA = True
use_aggregated_node_data = False
use_projected_width = True
algo3_monthly_series_pdf = False
algo3_yearly_mean_adjustement = False
use_expected_coeff = False
quantile_bathy_experiment = False
test = 'normal'
if test == 'stations':
    shape03 = 100.0
    auto_spread_for_runs = False
    station_period_only = True
    activate_facc = False
    use_fallback_sos = False
    fallback_sos_dir = ''
    use_dynamic_spread = False
    use_mean_for_bounds = True
    q_prior_from_ML = False
    reference_force_quit = True
    SVS_file = '/mnt/DATA/worksync/files/data/gages/SVS_v17/SVS_v1_0.5_trans_v17.nc'
    use_SVS = True
    svs_reach_id_var = 'reach_id_v17b'
    use_gauge_external_file = False
    gauge_external_file = '/mnt/DATA/CONFLUENCE/sic_update_2025_10/Bathymetry_validation_2026/ADCP_compiled.csv'
    use_friction_external_file = False
    friction_external_file = '/mnt/DATA/CONFLUENCE/sic_update_2025_10/Bathymetry_validation_2026/ADCP_compiled.csv'
elif test == 'ML':
    shape03 = 30.0
    auto_spread_for_runs = False
    station_period_only = True
    activate_facc = False
    use_fallback_sos = False
    fallback_sos_dir = ''
    use_dynamic_spread = False
    use_mean_for_bounds = True
    q_prior_from_ML = True
else:
    shape03 = 30.0
    shape13 = 2.1
    val1 = -0.05
    auto_spread_for_runs = False
    station_period_only = False
    activate_facc = True
    use_fallback_sos = False
    fallback_sos_dir = ''
    use_dynamic_spread = False
    use_mean_for_bounds = True
    q_prior_from_ML = False
eps1 = 0.01
eps2 = 0.0001
force_create_reach_t = True
run_algo31_v3 = False
def_float_atol = 0.01
pankaj_test = False
V32 = False
sigmaZ = 1.0
thres = 1.0
DH_Iter = 10
DH_Esp = 0.0001
useEXT = True
