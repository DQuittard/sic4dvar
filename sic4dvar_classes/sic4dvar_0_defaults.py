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

Created on September 15th 2023 at 16:00
by @Isadora Silva

Last modified on February 7th 2025 at 22:00
by @Isadora Silva

@authors: Isadora Silva
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
import numpy as np

@dataclass
class SIC4DVarLowCostDefaults:
    def_algo_version: str = '31_v4'
    def_handle_reaches: str = 'separately'
    def_ref_datetime: datetime = datetime(2000, 1, 1, 0, 0, 0, 0)
    def_freq_datetime: str = '6h'
    def_dist_dif_obs: int = 10
    def_outl_z_median_n_iters: int = 0
    def_outl_z_space_n_iters: int = 0
    def_outl_z_time_n_iters: int = 0
    def_outl_z_median_thr: float = 0.1
    def_outl_z_space_thr: float = 0.1
    def_outl_z_time_thr: float = 0.1
    def_outl_z_space_w_size: int = 5
    def_outl_w_median_n_iters: int = 0
    def_outl_w_space_n_iters: int = 0
    def_outl_w_time_n_iters: int = 0
    def_outl_w_median_thr: float = 10.0
    def_outl_w_space_thr: float = 10.0
    def_outl_w_time_thr: float = 10.0
    def_outl_w_space_w_size: int = 5
    def_lsm_z_x: int = 10
    def_lsm_z_t: int = 1
    def_lsm_cor_x: float | None = None
    def_lsm_z_x_behavior: str | Literal['force', 'sort', 'print', 'none', ''] = ''
    def_lsm_cor_t: float = 0.25 * 24 * 3600
    def_lsm_z_min_dz: float = 0.01
    def_lsm_z_inter: bool = False
    def_lsm_z_inter_min_dz: float = def_lsm_z_min_dz
    def_lsm_z_inter_max_dz: float = 0.5
    def_lsm_z_first_sweep: str = 'backward'
    def_lsm_z_rem_bias_in_loop: bool = True
    def_lsm_w_x: int = 3
    def_lsm_w_t: int = 1
    def_lsm_w_z: int = 10
    def_lsm_cor_z: float | None = None
    def_lsm_w_min_dw: float | None = 0.005
    def_lsm_w_weight_exp_beta: float = 0.5
    def_lsm_w_first_sweep: str = 'forward'
    def_lsm_w_rem_bias_in_loop: bool = False
    def_fill_missing_z_bool: bool = True
    def_fill_unobserved_nodes_z_bool: bool = False
    def_dx_max_z_in: float = np.inf
    def_dx_max_z_out: float = 1000.0
    def_fill_missing_w_bool: bool = True
    def_fill_missing_w_weight: str = 'mean_cs'
    def_fill_missing_w_extrap_min: bool = True
    def_fill_missing_w_extrap_max: bool = True
    def_dx_max_w_in: float = np.inf
    def_dx_max_w_out: float = 300.0
    def_cs_max_n_points: int = 8
    def_cs_max_dist: float = 0.15
    def_cs_depth_max: float = 250.0
    def_cs_depth_min: float = 0.25
    def_cs_float_atol: float = 0.01
    def_cs_width_top_check: bool = True
    def_cs_width_top_modify: bool = True
    def_cs_zb_shrink: bool = True
    def_cs_width_bottom_check: bool = True
    def_cs_build_method: str = 'relax_sweeps'
    def_cs_zb_method: str = 'burn'
    def_cs_width_top_max_angle: float = 30.0
    def_cs_width_top_max_dx: float = 3.0
    def_cs_wb_n_iters: float = 5
    def_cs_wb_float_atol: float = 0.5
    def_cs_wb_depth_min: float = 0.1
    def_cs_wb_depth_rel_min: float = 0.1
    def_cs_wb_q_quant: float = 0.5
    def_cs_wb_rel_tol_deb_solver: float = 0.5
    def_cs_wb_rel_diff_debitance_tol: float = 0.1
    def_cs_wb_rel_diff_q_tol: float = 0.25
    def_cs_wb_rel_diff_q_t_thr: float = 0.05
    def_cs_from_q_out_method: str = 'numerical'
    def_q_min_n_nodes: int = 1
    def_q_min_n_times: int = 1
    def_q_min_per_nodes: float = 5
    def_q_min_per_times: float = 5
    def_slope_max_dx: float = 2.0 * 10000.0
    def_slope_min_dx: float = 1.2 * 1000.0
    def_slope_max: float = 0.005
    def_slope_min: float = 1e-07
    def_slope_lsm: int = 0
    def_slope_from_nodes_method: str = 'lin'
    def_q_algo31_pdf: str = 'beta'
    def_q_bound_coef: float = 5.0
    def_q_pdf_std: float = 1.0
    def_q_prior_from_monthly = False
    def_q_dry_out_of_bounds: str = 'NaN'
    def_km_low: float = 10.0
    def_km_up: float = 60.0
    def_km_n_iters: int = 20
    def_zb_p_low_beta: float = 0.0
    def_zb_p_up_beta: float = 100.0
    def_zb_n_iters: int = 20
    def_k_dim: float = 101
    def_shape03: float = 5.0
    def_local_qm1: float = 5.0
    def_float_atol: float = 1e-08
    def_thres: float = 1.0
    def_use_ext_bool = True
    def_val_swot_q_flag = (0, 1)
    def_swot_q_min_n_nodes: int = 3
    def_swot_q_min_n_times: int = 10
    def_manning_n_min: float = 0.01
    def_manning_n_max: float = 0.25
sic_def = SIC4DVarLowCostDefaults()
_attr_dict = {'Simulated reach(es)': ('reach_id',), 'General parameters': ('algo_version', 'start_datetime_param', 'end_datetime_param', 'handle_reaches_param', 'ref_datetime_param', 'freq_datetime_param', 'float_atol_param'), 'Data sources': ('observations_source', 'cross_sections_source', 'q_prior_source', 'slope_source'), 'Outlier z params': ('node_z_outlier_removed_bool', 'outlier_z_median_n_iters_param', 'outlier_z_space_n_iters_param', 'outlier_z_time_n_iters_param', 'outlier_z_median_thr_param', 'outlier_z_space_thr_param', 'outlier_z_time_thr_param', 'outlier_z_space_w_size_param'), 'Outlier w params': ('node_w_outlier_removed_bool', 'outlier_w_median_n_iters_param', 'outlier_w_space_n_iters_param', 'outlier_w_time_n_iters_param', 'outlier_w_median_thr_param', 'outlier_w_space_thr_param', 'outlier_w_time_thr_param', 'outlier_w_space_w_size_param'), 'Smoothing z parameters': ('node_z_smooth_bool', 'smooth_z_in_space_bool', 'lsm_z_x_param', 'lsm_cor_x_param', 'smooth_z_in_time_bool', 'lsm_z_t_param', 'lsm_cor_t_param', 'lsm_z_min_dz_param', 'lsm_z_inter_bool', 'lsm_z_inter_max_dz_param', 'lsm_z_inter_min_dz_param', 'lsm_z_first_sweep_param', 'lsm_z_remove_bias_in_loop_bool'), 'Smoothing w parameters': ('node_w_obs_bool', 'node_w_smooth_bool', 'smooth_w_in_space_bool', 'lsm_w_x_param', 'smooth_w_in_time_bool', 'lsm_w_t_param', 'lsm_w_min_dw_param', 'lsm_w_first_sweep_param', 'lsm_w_remove_bias_in_loop_bool'), 'Filling missing z parameters': ('node_z_fill_bool', 'fill_missing_z_bool', 'fill_missing_z_unobserved_nodes_bool', 'dx_max_z_in_param', 'dx_max_z_out_param'), 'Filling missing w parameters': ('node_w_fill_bool', 'fill_missing_w_bool', 'fill_missing_w_weight_param', 'fill_missing_w_extrap_min_bool', 'fill_missing_w_extrap_max_bool', 'dx_max_w_in_param', 'dx_max_w_out_param'), 'Slope parameters': ('reach_s_obs_bool', 'slope_node_bool', 'reach_s_fixed_bool', 'slope_load_bool', 'slope_bool', 'slope_from_nodes_method_param', 'slope_lsm_param', 'slope_max_dx_param', 'slope_min_dx_param', 'slope_max_param', 'slope_min_param'), 'Cross-Section building parameters': ('cross_section_obs_bool', 'cross_section_ref_bool', 'cross_section_comb_bool', 'cross_section_reg_bool', 'cross_section_modif_zb_bool', 'cross_section_shrink_zb_bool', 'cross_section_use_obs_bool', 'cross_section_build_method_param', 'cross_section_bottom_method_param', 'cross_section_bottom_shrink_bool', 'cross_section_depth_min_param', 'cross_section_depth_max_param', 'cross_section_width_bottom_check_bool', 'cross_section_width_top_check_bool', 'cross_section_width_top_modify_bool', 'cross_section_width_top_max_angle_param', 'cross_section_width_top_max_dx_param', 'cross_section_float_atol_param', 'cross_section_simp_max_dist_param', 'cross_section_simp_max_n_points_param', 'lsm_w_z_param', 'lsm_cor_z_param', 'lsm_w_weight_exp_beta_param', 'cross_section_width_bottom_n_iters_param', 'cross_section_width_bottom_float_atol_param', 'cross_section_width_bottom_depth_min_param', 'cross_section_width_bottom_depth_rel_min_param', 'cross_section_width_bottom_q_quant_param', 'cross_section_width_bottom_rel_tol_deb_solver_param', 'cross_section_width_bottom_rel_diff_debitance_tol_param', 'cross_section_width_bottom_rel_diff_q_tol_param', 'cross_section_width_bottom_rel_diff_q_t_thr_param'), 'Filtering data parameters': ('q_min_n_nodes_param', 'q_min_n_times_param', 'q_min_per_nodes_param', 'q_min_per_times_param'), 'Algo 31 parameters': ('q_algo31_pdf_param', 'q_bound_coef_param', 'q_pdf_std_param', 'q_prior_from_monthly_bool', 'q_dry_out_of_bounds_param', 'km_low_param', 'km_up_param', 'km_n_iters_param', 'zb_p_low_beta_param', 'zb_p_up_beta_param', 'zb_n_iters_param', 'k_dim_param', 'shape03_param', 'local_qm1_param'), 'Number of nodes, reaches and time instances': ('total_n_nodes', 'total_n_reaches', 'total_n_times'), 'Arrays loaded inputs': ('dist_description', 'time_description', 'q_prior_description', 'node_z_init_description', 'node_w_init_description', 'reach_da_description', 'reach_s_description', 'reach_w_description'), 'Arrays modified inputs': ('node_z_fill_description', 'node_z_smooth_description', 'node_w_fill_description', 'node_w_smooth_description'), 'Arrays uncertainties': ('node_z_init_low_bound_description', 'node_z_init_up_bound_description', 'node_z_fill_low_bound_description', 'node_z_fill_up_bound_description', 'node_z_smooth_low_bound_description', 'node_z_smooth_up_bound_description', 'node_w_init_low_bound_description', 'node_w_init_up_bound_description', 'node_w_fill_low_bound_description', 'node_w_fill_up_bound_description', 'node_w_smooth_low_bound_description', 'node_w_smooth_up_bound_description'), 'Arrays cross-Section': ('cross_section_w_obs_description', 'cross_section_w_ref_description', 'cross_section_w_comb_description', 'cross_section_w_reg_description', 'cross_section_w_description', 'cross_section_w_out_description', 'cross_section_w_min_description', 'cross_section_z_obs_description', 'cross_section_z_ref_description', 'cross_section_z_comb_description', 'cross_section_z_reg_description', 'cross_section_z_description', 'cross_section_z_out_description', 'cross_section_z_min_description', 'cross_section_z_out_min_description'), 'Array slope': ('slope_description',), 'Area': ('wet_area_added_zb_loop_description', 'wet_area_max_dry_description'), 'Roughness': ('roughness_out_description',), 'Array discharge output': ('reach_q_out_description',)}
_attr_dict_summary = {'Simulated reach(es)': ('reach_id',), 'General parameters': ('algo_version', 'start_datetime_param', 'end_datetime_param', 'handle_reaches_param', 'freq_datetime_param'), 'Data sources': ('observations_source', 'cross_sections_source', 'q_prior_source', 'slope_source'), 'Outlier z params': ('node_z_outlier_removed_bool',), 'Outlier w params': ('node_w_outlier_removed_bool',), 'Smoothing z parameters': ('node_z_smooth_bool', 'lsm_z_x_param', 'lsm_z_t_param'), 'Smoothing w parameters': ('node_w_obs_bool', 'lsm_w_x_param', 'lsm_w_t_param'), 'Filling missing z parameters': ('node_z_fill_bool', 'fill_missing_z_bool', 'fill_missing_z_unobserved_nodes_bool'), 'Filling missing w parameters': ('node_w_fill_bool', 'fill_missing_w_bool'), 'Slope parameters': ('reach_s_obs_bool', 'slope_node_bool', 'reach_s_fixed_bool', 'slope_load_bool', 'slope_bool'), 'Cross-Section building parameters': ('cross_section_obs_bool', 'cross_section_ref_bool', 'cross_section_comb_bool', 'cross_section_reg_bool', 'cross_section_modif_zb_bool', 'cross_section_shrink_zb_bool', 'cross_section_use_obs_bool', 'cross_section_build_method_param', 'cross_section_bottom_method_param', 'cross_section_bottom_shrink_bool', 'cross_section_simp_max_n_points_param'), 'Algo 31 parameters': ('q_algo31_pdf_param', 'q_bound_coef_param', 'q_pdf_std_param', 'q_prior_from_monthly_bool', 'km_low_param', 'km_up_param', 'km_n_iters_param', 'zb_p_low_beta_param', 'zb_p_up_beta_param', 'zb_n_iters_param', 'local_qm1_param'), 'Number of nodes, reaches and time instances': ('total_n_nodes', 'total_n_reaches', 'total_n_times')}