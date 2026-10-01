_G = 'node_z'
_F = 'filtered_data'
_E = False
_D = True
_C = None
_B = 1.0
_A = 0.0
import numpy as np
from sic4dvar_functions import sic4dvar_calculations as calc
from pathlib import Path
from scipy.optimize import minimize_scalar
from sic4dvar_functions.sic4dvar_calculations import verify_name_length, fnc_APR
from sic4dvar_functions.sic4dvar_helper_functions import *
from sic4dvar_functions.sic4dvar_gnuplot_save import gnuplot_save_tables, gnuplot_save_slope, gnuplot_save_zm
from lib.lib_indicators import integrated_mean, integrated_variance
from lib.lib_verif import check_na
from sic4dvar_modules.sic4dvar_compute_slope_and_bathymetry import call_func_APR
from sic4dvar_functions.Y786 import K
import math

def Z0(node_xr, node_yr, node_a, node_p, Wmean, last_node_for_integral, Zb2, ZB_update, Zb, t, node_x, SLOPEM1, ZM=_A, KMI=_B, option_recompute_area=_E, sic4dvar_dict=[], alpha4=_B):
    SS1 = _A
    tmp_R_array = []
    for n in range(last_node_for_integral):
        Zmin, Wmin = (np.nanmin(node_yr[n]), np.nanmin(node_xr[n]))
        if Wmin == 0:
            Wmin = _B
        if t == 0:
            Zb[n] = node_yr[n][0] + Zb2 + ZB_update * (Wmean / Wmin)
        Z1 = node_yr[n][0]
        W1 = node_xr[n][0]
        if Zmin != Z1:
            0
        if Wmin != W1:
            0
        if Wmin == 0 or check_na(Wmin):
            0
        if W1 == 0:
            0
        if option_recompute_area:
            ZM_est = (1 + ZM * alpha4) * sic4dvar_dict[_F][_G][n, t] - ZM * alpha4 * (node_yr[n][0] + Zb2)
            node_a[n], node_p[n], _, _ = fnc_APR([ZM_est], node_xr[n], node_yr[n])
        A1 = node_a[n][t]
        P1 = node_p[n][t]
        A0 = W1 * (Z1 - Zb[n])
        if check_na(A0):
            0
        P0 = 2 * (Z1 - Zb[n])
        AA = A0 + A1
        PP = P0 + P1
        if AA < 0:
            0
        if PP < 0:
            0
        R = AA / PP
        tmp_R_array.append(R)
        QMi2i3 = AA * math.pow(R, 2.0 / 3.0)
        if n == 0:
            QM0 = QMi2i3
        else:
            if QM0 <= 0 or QMi2i3 <= 0:
                0
            if QM0 > 0 and QMi2i3 > 0:
                SS1 += abs(node_x[n] - node_x[n - 1]) * (math.pow(QM0, -2) + math.pow(QMi2i3, -2)) / 2.0
            QM0 = QMi2i3
    tmp_R_array = np.array(tmp_R_array)
    tmp_R_array = np.mean(tmp_R_array)
    Sl = (1 + ZM * alpha4) * SLOPEM1[t] - ZM * alpha4 * (node_yr[n][0] - node_yr[n][-1])
    if Sl / abs(node_x[-1] - node_x[0]) < 1e-06:
        Sl = abs(node_x[-1] - node_x[0]) * 1e-06
    QMi2i3 = math.sqrt(Sl / SS1) * KMI
    return (QMi2i3, SS1, Zb, tmp_R_array)

def Z1(node_xr, node_yr, node_a, node_p, Wmean, last_node_for_integral, Zb2, ZB_update, Zb, t, node_x, SLOPEM1, ZM=_B, KMI=_B, option_recompute_area=_E, sic4dvar_dict=[]):
    SS1 = _A
    tmp_R_array = []
    for n in range(last_node_for_integral):
        Zmin, Wmin = (np.nanmin(node_yr[n]), np.nanmin(node_xr[n]))
        if Wmin == 0:
            Wmin = _B
        if t == 0:
            Zb[n] = node_yr[n][0] + Zb2 + ZB_update * (Wmean / Wmin)
        Z1 = node_yr[n][0]
        W1 = node_xr[n][0]
        if Zmin != Z1:
            0
        if Wmin != W1:
            0
        if Wmin == 0 or check_na(Wmin):
            0
        if W1 == 0:
            0
        if option_recompute_area:
            node_a[n], node_p[n], _, _ = fnc_APR([sic4dvar_dict[_F][_G][n, t] * ZM], node_xr[n], node_yr[n])
        A1 = node_a[n][t]
        P1 = node_p[n][t]
        A0 = W1 * (Z1 - Zb[n])
        if check_na(A0):
            0
        P0 = 2 * (Z1 - Zb[n])
        AA = A0 + A1
        PP = P0 + P1
        if AA < 0:
            0
        if PP < 0:
            0
        R = AA / PP
        tmp_R_array.append(R)
        QMi2i3 = AA * math.pow(R, 2.0 / 3.0)
        if n == 0:
            QM0 = QMi2i3
        else:
            if QM0 <= 0 or QMi2i3 <= 0:
                0
            if QM0 > 0 and QMi2i3 > 0:
                SS1 += abs(node_x[n] - node_x[n - 1]) * (math.pow(QM0, -2) + math.pow(QMi2i3, -2)) / 2.0
            QM0 = QMi2i3
    tmp_R_array = np.array(tmp_R_array)
    tmp_R_array = np.mean(tmp_R_array)
    Sl = SLOPEM1[t] * ZM
    if Sl / abs(node_x[-1] - node_x[0]) < 1e-06:
        Sl = abs(node_x[-1] - node_x[0]) * 1e-06
    QMi2i3 = math.sqrt(Sl / SS1) * KMI
    return (QMi2i3, SS1, Zb, tmp_R_array)

def Z2(ZM, node_xr, node_yr, node_a, node_p, Wmean, last_node_for_integral, Zb2, ZB_update, Zb, t, node_x, SLOPEM1, Kmi_acc, sic4dvar_dict, Q_REF, alpha4):
    Q_EST, _, _, _ = Z0(node_xr, node_yr, node_a, node_p, Wmean, last_node_for_integral, Zb2, ZB_update, Zb, t, node_x, SLOPEM1, ZM=ZM, KMI=Kmi_acc, option_recompute_area=_D, sic4dvar_dict=sic4dvar_dict)
    if check_na(Q_EST) or Q_EST <= 0:
        0
    return (Q_REF - Q_EST) ** 2

def Z3(apr_array, Qwbm, params, SLOPEM1, valid_output, node_z, node_z_input, node_z_ini, node_x, last_node_for_integral, bathymetry_array, param_dict, reach_id='', reach_t=[], bb=9999.0, reliability='', Qsdev=_A, last_time_instant=-9999.0, input_data=[], time_indexes_to_keep=[], ZM_array_input=[], friction_value=_C, series_cv=_C, dynamic_slope=_C, monthly_time_series=_C, monthly_weights=_C, series_mean=_C, correlation_criteria=_C, sic4dvar_dict=[]):
    F = 'gnuplot_data'
    E = 'add'
    D = 'mult'
    C = 'output_dir'
    B = 'q_parametrization'
    A = 'gnuplot_saving'
    node_w_simp = apr_array['node_w_simp']
    node_a = apr_array['node_a']
    node_p = apr_array['node_p']
    node_r = apr_array['node_r']
    node_xr = bathymetry_array['node_xr']
    node_yr = bathymetry_array['node_yr']
    if params.V32:
        nsobs = 0
        for ij in range(node_z.shape[1]):
            if params.useEXT:
                INdata = node_z[:, ij]
                index_temp = np.where(INdata.mask == _E)
                index_temp2 = np.argwhere(np.isnan(INdata[index_temp]))
                if len(index_temp) > 0:
                    nsobs += len(index_temp[0])
                if len(index_temp2) > 0:
                    nsobs -= len(index_temp2[0])
            else:
                INdata = node_z_ini[:, ij]
                index_temp = np.where(INdata.mask == _E)
                index_temp2 = np.argwhere(np.isnan(INdata[index_temp]))
                if len(index_temp) > 0:
                    nsobs += len(index_temp[0])
                if len(index_temp2) > 0:
                    nsobs -= len(index_temp2[0])
    Wmean, Qp, QM1, QM2 = ([], [], [], [])
    Wmin = 10000
    Wmean = []
    for i in range(len(node_w_simp)):
        Wmean.append(min(node_w_simp[i]))
        if np.min(node_w_simp[i]) < Wmin:
            Wmin = np.min(node_w_simp[i])
    Wmean = np.average(Wmean)
    Qp = params.algo_bounds[0][0]
    QM1 = Qwbm / Qp
    QM2 = Qwbm * Qp
    if params.qsdev_activate:
        QM1s = Qsdev / Qp
        QM2s = Qsdev * Qp
    Zb1 = params.algo_bounds[1][0]
    Zb2 = params.algo_bounds[1][1]
    dZb = (Zb1 - Zb2) / 40
    if -bb > Zb1:
        Zb1 = -bb
        dZb = (Zb1 - Zb2) / 40
    if abs(Zb1) > Wmean:
        Zb1 = -Wmean + dZb
        dZb = (Zb1 - Zb2) / 40
    if params.pankaj_test:
        Zb1 = _A
        Zb2 = _A
    Km1 = params.algo_bounds[2][0]
    Km2 = params.algo_bounds[2][1]
    dKm = params.algo_bounds[2][2]
    I2m = 1 + int((Zb1 - Zb2) / dZb)
    I3m = 1 + int((Km2 - Km1) / dKm)
    trialps0_max = _A
    p_YMS_a = _A
    iexit = 0
    temp_i3 = []
    if params.use_dynamic_spread:
        beta_value, QM1, QM2 = define_spread(input_data['quant_mean'], input_data['quant_var'], _D, Qp, input_data['quantiles'], params.relative_variance)
        if not check_na(beta_value):
            params.shape03 = beta_value
            gamma = gamma_table(beta_value)
        else:
            QM1 = Qwbm / Qp
            QM2 = Qwbm * Qp
            gamma = _B
    else:
        QM1 = Qwbm / Qp
        QM2 = Qwbm * Qp
        gamma = _B
    if params.qsdev_activate:
        if not check_na(params.shape03):
            gamma = gamma_table(params.shape03)
        else:
            gamma = _B
    shapeP = []
    shapeP += [[(Qwbm * gamma - QM1) / (QM2 - QM1), params.shape03]]
    if params.qsdev_activate:
        shapePs = []
        shapePs += [[(Qsdev - QM1s) / (QM2s - QM1s), params.shape03]]
    delayZB = (params.val1 - Zb2) / (Zb1 - Zb2)
    shapeP += [[delayZB, params.shape13]]
    if friction_value is not _C:
        mean_K = friction_value
        shape_K = params.known_friction_shape
    else:
        mean_K = params.val2
        shape_K = params.shape23
    delayK = (mean_K - Km1) / (Km2 - Km1)
    shapeP += [[delayK, shape_K]]
    kDim = params.kDim
    q_pdf_table = calc.betad(int(param_dict[B]), kDim, shapeP[0][0], shapeP[0][1])
    if params.qsdev_activate:
        qs_pdf_table = calc.betad(int(param_dict[B]), kDim, shapePs[0][0], shapePs[0][1])
    zb_pdf_table = calc.betad(0, kDim, shapeP[1][0], shapeP[1][1])
    km_pdf_table = calc.betad(0, kDim, shapeP[2][0], shapeP[2][1])
    step = 1 / (kDim - 1)
    value = 0
    q_pdf_table = np.array(q_pdf_table[1])
    if params.qsdev_activate:
        qs_pdf_table = np.array(qs_pdf_table[1])
    zb_pdf_table = np.array(zb_pdf_table[1])
    km_pdf_table = np.array(km_pdf_table[1])
    q_pdf_table = q_pdf_table + value
    if params.qsdev_activate:
        qs_pdf_table = qs_pdf_table + value
    q_positional_arg = np.arange(0, len(q_pdf_table))
    if params.qsdev_activate:
        qs_positional_arg = np.arange(0, len(qs_pdf_table))
    zb_pdf_table[0] = zb_pdf_table[1]
    zb_pdf_table[-1] = zb_pdf_table[len(km_pdf_table) - 2]
    km_pdf_table[0] = km_pdf_table[1]
    km_pdf_table[-1] = km_pdf_table[len(km_pdf_table) - 2]
    if param_dict[A]:
        reach_id = str(reach_id)
        reach_id = verify_name_length(reach_id)
        reach_id = verify_name_length(reach_id)
        output_dir = param_dict[C].joinpath('_data', reach_id)
        if not output_dir.is_dir():
            output_dir.mkdir(parents=_D, exist_ok=_D)
        q_pdf_table_saved = q_pdf_table / np.max(q_pdf_table)
        zb_pdf_table_saved = zb_pdf_table / np.max(zb_pdf_table)
        km_pdf_table_saved = km_pdf_table / np.max(km_pdf_table)
        gnuplot_save_tables(q_pdf_table_saved, zb_pdf_table_saved, km_pdf_table_saved, output_dir.joinpath('tables'), 1)
    temp = []
    SS1_array = []
    if not valid_output:
        return (np.ones(node_z[0].shape[0]) * np.nan, 0)
    LSM = params.slope_smoothing_num_passes
    for k in range(LSM):
        SLOPEM2 = [0.75 * SLOPEM1[0] + 0.25 * SLOPEM1[1]]
        for t in range(1, node_z[0].shape[0] - 1):
            SLOPEM2 += [0.25 * SLOPEM1[t - 1] + 0.5 * SLOPEM1[t] + 0.25 * SLOPEM1[t + 1]]
        SLOPEM2 += [0.25 * SLOPEM1[-2] + 0.75 * SLOPEM1[-1]]
        SLOPEM1 = np.array(SLOPEM2)
    QM = np.zeros(node_z[0].shape[0])
    Nbeta = _A
    QM0 = _A
    QMT_array = np.zeros((I2m, node_z[0].shape[0]), dtype=np.float64)
    QMEAN_array = np.zeros(I2m, dtype=np.float64)
    QMEAN_array2 = np.zeros(I2m, dtype=np.float64)
    QMEAN_array_old = np.zeros(I2m, dtype=np.float64)
    time_scaling = np.zeros(I2m, dtype=np.float64)
    time_scaling2 = np.zeros(I2m, dtype=np.float64)
    temp_Q = np.zeros((len(node_z), node_z[0].shape[0]))
    temp_Q = []
    temp_Q_t = []
    if params.V32:
        Q_sample = np.zeros((I2m * I3m, node_z[0].shape[0]))
        Weight = np.zeros(I2m * I3m)
        iis = 0
        cost_all = []
    test_tmp = []
    Q_pdf_save = np.zeros((I2m, I3m))
    QMEAN_list = []
    reach_t_days = deepcopy(reach_t) / 86400
    ref_date = reach_t_days[0]
    QMEAN_years_indexes = []
    QMEAN_current_indexes = []
    for index_day, day in enumerate(reach_t_days):
        if day - ref_date <= params.day_length:
            QMEAN_current_indexes.append(np.where(reach_t_days == day)[0][0])
        else:
            ref_date = deepcopy(day)
            QMEAN_years_indexes.append(QMEAN_current_indexes)
            QMEAN_current_indexes = []
            QMEAN_current_indexes.append(np.where(reach_t_days == day)[0][0])
        if index_day == len(reach_t_days) - 1:
            QMEAN_years_indexes.append(QMEAN_current_indexes)
    first_time_instant_array = []
    last_time_instant_array = []
    for year_index in QMEAN_years_indexes:
        first_time_instant_array.append(year_index[0])
        last_time_instant_array.append(year_index[-1])
    tmp_QMT_array_list = []
    Zb_acc = _A
    Kmi_acc = _A
    R = _A
    R_max = _A
    mean_pdf_product = []
    expected_coeff = _C
    ZM_array_save = []
    for i2 in range(1, I2m + 1):
        QMEAN_list = []
        year_index = 0
        time_scaling_array = []
        ZBA = Zb2 + (i2 - 1) * dZb
        Zb = np.zeros(len(node_z))
        temp_ss1_array = []
        ZB_update = (i2 - 1) * dZb
        if params.use_zb_early_break:
            if np.abs(ZB_update) > 2 * R_max:
                break
        tmp_R_array = []
        for t in range(node_z[0].shape[0]):
            SS1 = _A
            temp_Q_t = []
            if np.array(ZM_array_input).size > 0:
                ZM = ZM_array_input[t]
            else:
                ZM = _B
            QMi2i3, SS1, Zb, R_mean = Z1(node_xr, node_yr, node_a, node_p, Wmean, last_node_for_integral, Zb2, ZB_update, Zb, t, node_x, SLOPEM1, ZM=ZM, KMI=_B, option_recompute_area=_E)
            tmp_R_array.append(R_mean)
            QMEAN_array_old[i2 - 1] += QMi2i3
            if t > first_time_instant_array[year_index] and t < last_time_instant_array[year_index]:
                QMEAN_array2[i2 - 1] += (QMi2i3 + QMi2i3_0) / 2 * (reach_t[t] - reach_t[t - 1])
                time_scaling2[i2 - 1] += reach_t[t] - reach_t[t - 1]
            if t == last_time_instant_array[year_index]:
                QMEAN_array2[i2 - 1] += (QMi2i3 + QMi2i3_0) / 2 * (reach_t[t] - reach_t[t - 1])
                time_scaling2[i2 - 1] += reach_t[t] - reach_t[t - 1]
                year_index += 1
                QMEAN_list.append(QMEAN_array2[i2 - 1])
                QMEAN_array2[i2 - 1] = _A
                time_scaling_array.append(time_scaling2[i2 - 1])
                time_scaling2[i2 - 1] = _A
            QMT_array[i2 - 1, t] = QMi2i3
            SS1_array.append(1 / SS1)
            QMi2i3_0 = QMi2i3
        if i2 == 1:
            tmp_R_array = np.array(tmp_R_array)
            R = np.mean(tmp_R_array)
            R_max = np.max(tmp_R_array)
        if param_dict[A]:
            if not output_dir.is_dir():
                output_dir.mkdir(parents=_D, exist_ok=_D)
            times2 = np.around(reach_t / 3600 / 24)
            times2 = times2 - min(times2)
            gnuplot_save_slope(temp_ss1_array, times2, output_dir.joinpath('integral'))
        SS1_array = []
        QMEAN_array_old /= node_z[0].shape[0]
        for m in range(0, len(QMEAN_list)):
            QMEAN_list[m] /= time_scaling_array[m]
        if params.qsdev_activate:
            Qsdev = _A
            time_scaling_sdev = _A
            for ist in range(1, node_z[0].shape[0]):
                s0 = (QMT_array[i2 - 1, ist - 1] - QMEAN_list[m]) ** 2
                s1 = (QMT_array[i2 - 1, ist] - QMEAN_list[m]) ** 2
                time_scaling_sdev += reach_t[ist] - reach_t[ist - 1]
                Qsdev = Qsdev + (s0 + s1) / 2 * (reach_t[ist] - reach_t[ist - 1])
            Qsdev = np.sqrt(Qsdev / time_scaling_sdev)
        QMT_const_array = np.copy(QMT_array)
        count = _A
        for t in range(0, len(QMT_array[i2 - 1, :])):
            if check_na(QMT_array[i2 - 1, t]) or QMT_array[i2 - 1, t] < 1e-09:
                QMT_array[i2 - 1, t] = 1e-09
                count += _B
        if count > 0:
            0
        if params.algo3_displace_series and series_cv is not _C and (dynamic_slope is not _C):
            QMT_2D = []
            for n in range(0, len(node_a)):
                QMT_2D.append(QMT_array[i2 - 1, :])
            QMT_2D = np.array(QMT_2D)
            QMT_2D = K(dim=0, value0_array=QMT_2D, base0_array=reach_t, max_iter=4, cor=604800, always_run_first_iter=_E, behavior='', inter_behavior=_E, inter_behavior_min_thr=params.def_float_atol, inter_behavior_max_thr=params.DX_max_in, check_behavior='', min_change_v_thr=0.0001, plot=_E, plot_title='Relaxation sweep in time QMT without Interchange', clean_run=_D, debug_mode=_E)
            QMT_for_cv = QMT_2D[0]
            mean_QMT = integrated_mean(QMT_for_cv, reach_t)
            std_QMT = integrated_variance(QMT_for_cv, reach_t, mean_QMT)
            cv_QMT = std_QMT / mean_QMT
            alpha = np.exp(-0.5 * (cv_QMT / series_cv))
            QMT_array[i2 - 1, :] = QMT_array[i2 - 1, :] * (1 - alpha + alpha * dynamic_slope / SLOPEM1) ** 0.5
            mean_QMT2 = integrated_mean(QMT_array[i2 - 1, :], reach_t)
            std_QMT2 = integrated_variance(QMT_array[i2 - 1, :], reach_t, mean_QMT2)
            cv_QMT2 = std_QMT2 / mean_QMT2
        if params.use_flow_duration_q and monthly_weights is not _C:
            QMT_mean_per_month = np.zeros(len(monthly_weights))
            Q_series_mean_per_month = np.zeros(len(monthly_weights))
            QMT_array_1 = deepcopy(QMT_array[i2 - 1, :])
            if correlation_criteria is not _C:
                if correlation_criteria <= _A:
                    correlation_criteria = _A
                    beta2 = _A
                else:
                    beta2 = 0.5
            else:
                beta2 = _A
            if i2 == 1:
                0
            power = _B
            Q_prime = deepcopy(QMT_array_1)
            QMT_array_1_mean = integrated_mean(QMT_array_1, reach_t)
            for i in range(0, len(monthly_weights)):
                compute_alpha2 = _D
                Q_mean_denom = integrated_mean(QMT_array_1 * monthly_weights[i], reach_t)
                Phi_mean = integrated_mean(monthly_weights[i], reach_t)
                if check_na(Phi_mean) or Phi_mean < 1e-09:
                    Phi_mean = _B
                QMT_mean_per_month[i] = Q_mean_denom / Phi_mean
                Q_mean_denom = integrated_mean(monthly_time_series * monthly_weights[i], reach_t)
                Q_series_mean_per_month[i] = Q_mean_denom / Phi_mean
                if check_na(QMT_mean_per_month[i]) or QMT_mean_per_month[i] < 1e-09:
                    alpha2 = _B
                    compute_alpha2 = _E
                if check_na(Q_series_mean_per_month[i]) or Q_series_mean_per_month[i] < 1e-09:
                    alpha2 = _B
                    compute_alpha2 = _E
                if check_na(series_mean) or series_mean < 1e-09:
                    alpha2 = _B
                    compute_alpha2 = _E
                if compute_alpha2:
                    if params.correction_type == D:
                        alpha2 = QMT_array_1_mean / QMT_mean_per_month[i] * (Q_series_mean_per_month[i] / series_mean)
                    elif params.correction_type == E:
                        alpha2 = QMT_array_1_mean * (Q_series_mean_per_month[i] / series_mean) - QMT_mean_per_month[i]
                if params.correction_type == D:
                    Q_prime = Q_prime * (1 + beta2 * (alpha2 ** power - 1) * monthly_weights[i])
                elif params.correction_type == E:
                    Q_prime = Q_prime + beta2 * alpha2 * monthly_weights[i]
            Q_prime_mean = integrated_mean(Q_prime, reach_t)
            QMT_array_1_old = deepcopy(QMT_array_1)
            QMT_array_1 = Q_prime / Q_prime_mean * QMT_array_1_mean if Q_prime_mean > 0 else QMT_array_1
            QMT_array[i2 - 1, :] = deepcopy(QMT_array_1)
        if params.mapping_to_wse_obs:
            Kmi_acc_mapping = _B
            ZM_opt_array = []
            node_yr_0_vector = []
            for n in range(0, len(node_yr)):
                node_yr_0_vector.append(node_yr[n][0])
            node_yr_0_vector = np.array(node_yr_0_vector)
            denom_alpha3 = _A
            alpha4 = _B
            for t in range(0, node_z.shape[1]):
                alpha4 = _B
                sum_wse_sq = _A
                Q_REF = QMT_array[i2 - 1, t]
                args_brent = (deepcopy(node_xr), deepcopy(node_yr), deepcopy(node_a), deepcopy(node_p), deepcopy(Wmean), deepcopy(last_node_for_integral), deepcopy(Zb2), deepcopy(ZB_update), deepcopy(Zb), t, deepcopy(node_x), deepcopy(SLOPEM1), deepcopy(Kmi_acc_mapping), deepcopy(sic4dvar_dict), deepcopy(Q_REF), deepcopy(alpha4))
                opt_res = minimize_scalar(Z2, bounds=(-0.9, 2.0), args=args_brent, method='bounded', options={'maxiter': 20})
                ZM_opt = opt_res.x
                if not opt_res.success:
                    ZM_opt = _A
                ZM_opt_array.append(ZM_opt)
                sum_wse_sq = np.mean((node_z[:, t] - node_yr_0_vector) ** 2)
                gamma_squared = ZM_opt ** 2 if not check_na(ZM_opt) else _A
                denom_alpha3 += gamma_squared * sum_wse_sq
            denom_alpha3 /= node_z.shape[1]
            sigma_wse_obs = 0.2
            alpha3 = sigma_wse_obs ** 2 / denom_alpha3 if denom_alpha3 > 0 else _C
            alpha3 = np.sqrt(alpha3) if alpha3 is not _C and alpha3 > 0 else _C
            if alpha3 > _B:
                alpha3 = _B
            scale = 1 + alpha3 * (np.array(ZM_opt_array) - 1)
            ZM_array_save.append(ZM_opt_array)
            Q_EST2_array = []
            alpha4 = alpha3 if alpha3 is not _C else _B
            for t in range(0, node_z.shape[1]):
                Q_EST2, _, _, _ = Z0(node_xr, node_yr, deepcopy(node_a), deepcopy(node_p), Wmean, last_node_for_integral, Zb2, ZB_update, deepcopy(Zb), t, node_x, deepcopy(SLOPEM1), ZM=ZM_opt_array[t], KMI=Kmi_acc_mapping, option_recompute_area=_D, sic4dvar_dict=sic4dvar_dict, alpha4=alpha4)
                Q_EST2_array.append(Q_EST2)
            QMT_array[i2 - 1, :] = deepcopy(Q_EST2_array)
        QMT_series_constant = deepcopy(QMT_array[i2 - 1, :])
        if i2 == 1 and params.remake_zb_table:
            mean_QMT3 = integrated_mean(QMT_array[i2 - 1, :], reach_t)
            std_QMT3 = integrated_variance(QMT_array[i2 - 1, :], reach_t, mean_QMT3)
            cv_QMT3 = std_QMT3 / mean_QMT3
            alpha1 = np.exp(-0.5 * (cv_QMT3 / series_cv))
            shape13 = params.shape13 / (1 - alpha1) if 1 - alpha1 > 0 else np.nan
            delayZB = (params.val1 - Zb2) / (Zb1 - Zb2)
            zb_pdf_table = calc.betad(0, kDim, delayZB, shape13)
            zb_pdf_table = np.array(zb_pdf_table[1])
            zb_pdf_table[0] = zb_pdf_table[1]
            zb_pdf_table[-1] = zb_pdf_table[len(zb_pdf_table) - 2]
            product_table = q_pdf_table * zb_pdf_table * km_pdf_table
            expected_value_product = _A
            expected_value_q = _A
            product_pdf_acc = _A
            q_pdf_acc = _A
            for i in range(1, kDim):
                expected_value_product += product_table[i] * ((i - 1) / (kDim - 1))
                product_pdf_acc += product_table[i]
                q_pdf_acc += q_pdf_table[i]
                expected_value_q += q_pdf_table[i] * ((i - 1) / (kDim - 1))
            expected_value_q /= q_pdf_acc
            expected_value_product /= product_pdf_acc
            expected_coeff = expected_value_product / expected_value_q if expected_value_q > 0 else _C
        trialps1_max = _A
        k_step = np.exp(_B / (I3m - 1) * np.log(Km2 / Km1))
        p_YMS_a1 = _A
        yarg_a1 = np.zeros(node_z[0].shape[0], dtype=np.float64)
        Kmi_acc2 = _A
        Zb_acc2 = _A
        for i3 in range(1, I3m + 1):
            SS1_list = []
            Q_pdf_list = []
            theta_list = []
            theta = (Zb2 + (i2 - 1) * dZb - Zb2) / (Zb1 - Zb2)
            if theta < _A:
                theta = _A
            if theta > _B:
                theta = _B
            if not params.pankaj_test:
                Zb_pdf = zb_pdf_table[int(theta * (kDim - 1))]
            if params.pankaj_test:
                Zb_pdf = _B
            theta = (Km1 + (i3 - 1) * dKm - Km1) / (Km2 - Km1)
            if theta < _A:
                theta = _A
            if theta > _B:
                theta = _B
            Km_pdf = km_pdf_table[int(theta * (kDim - 1))]
            if params.uniform_friction:
                Kmi3 = Km1 + (i3 - 1) * dKm
            elif not params.uniform_friction:
                if i3 == 1:
                    Kmi3 = Km1
                else:
                    Kmi3 = Km1 * k_step ** (i3 - 1)
            SS1 = QMEAN_array[i2 - 1] * Kmi3
            for m in range(0, len(QMEAN_list)):
                SS1_list.append(QMEAN_list[m] * Kmi3)
                if params.qsdev_activate:
                    if params.qsdev_option == 0:
                        SS1_qs = Qsdev * Kmi3
                    elif params.qsdev_option == 1:
                        SS1_qs = Qsdev / QMEAN_list[m]
                theta_list.append((SS1_list[m] - QM1) / (QM2 - QM1))
                if params.qsdev_activate:
                    theta_qs = (SS1_qs - QM1s) / (QM2s - QM1s)
                if theta_list[m] <= 0 or check_na(theta_list[m]):
                    theta_list[m] = 1e-06
                if theta_list[m] >= 1:
                    theta_list[m] = 1
                if params.qsdev_activate:
                    if theta_qs <= 0 or check_na(theta_qs):
                        theta_qs = 1e-06
                    if theta_qs >= 1:
                        theta_qs = 1
                if params.use_expected_coeff and expected_coeff is not _C:
                    Q_pdf_list.append(interp_pdf_tables(kDim - 1, theta_list[m] * kDim * expected_coeff, q_positional_arg, q_pdf_table))
                else:
                    Q_pdf_list.append(interp_pdf_tables(kDim - 1, theta_list[m] * kDim, q_positional_arg, q_pdf_table))
                if params.qsdev_activate:
                    Qs_pdf = interp_pdf_tables(kDim - 1, theta_qs * kDim, qs_positional_arg, qs_pdf_table)
                Q_pdf_save[i2 - 1, i3 - 1] = Q_pdf_list[m]
            Q_pdf = _B
            for m in range(0, len(QMEAN_list)):
                Q_pdf = Q_pdf * Q_pdf_list[m]
            if params.V32:
                Qtrial = theta
                if theta > _B:
                    Qtrial = _B
                if theta < _A:
                    Qtrial = _A
                QMT_array[i2 - 1, :] = QMT_const_array[i2 - 1, :] / QMEAN_array[i2 - 1] * (Qtrial * QM2 + (_B - Qtrial) * QM1)
            elif params.algo3_displace_series and series_cv is not _C and (dynamic_slope is not _C):
                QMT_array[i2 - 1, :] = QMT_series_constant * Kmi3
            else:
                QMT_array[i2 - 1, :] = QMT_const_array[i2 - 1, :] * Kmi3
            if not params.qsdev_activate:
                Nbeta += Zb_pdf * Km_pdf * Q_pdf
            if params.qsdev_activate:
                Nbeta += Zb_pdf * Km_pdf * (Q_pdf * Qs_pdf)
            if params.qsdev_activate:
                trial_ps0 = Zb_pdf * Km_pdf * (Q_pdf * Qs_pdf)
            else:
                trial_ps0 = Zb_pdf * Km_pdf * Q_pdf
            if params.qsdev_activate:
                ss_a = Km_pdf * (Q_pdf * Qs_pdf)
            else:
                ss_a = Km_pdf * Q_pdf
            ss_b = Zb_pdf
            yarg1 = deepcopy(QMT_array[i2 - 1, :])
            yarg1_a = deepcopy(yarg1)
            if i3 > 1:
                p_YMS_a1 += (ss_a + ss_a1) / 2.0 * (Kmi3 - Kmi3_0)
                for t in range(0, len(node_z[0])):
                    yarg_a1[t] += (yarg1_a[t] * ss_a + yarg_a2[t] * ss_a1) / 2.0 * (Kmi3 - Kmi3_0)
                Kmi_acc2 += (Kmi3 * ss_a + Kmi3_0 * ss_a1) / 2.0 * (Kmi3 - Kmi3_0)
                Zb_acc2 += (ZBA * ss_a + ZBA2 * ss_a1) / 2.0 * (Kmi3 - Kmi3_0)
            ss_a1 = deepcopy(ss_a)
            yarg_a2 = deepcopy(yarg1_a)
            Kmi3_0 = deepcopy(Kmi3)
            ZBA2 = deepcopy(ZBA)
            if params.V32:
                NBS = len(node_z[:, :]) - 1
                icost = _A
                cost = _A
                Q_sample[iis] = QMT_array[i2 - 1, :]
                Weight[iis] = Zb_pdf * Km_pdf * Q_pdf
                TDIM = len(Q_sample[iis, :])
                for ij in range(0, TDIM):
                    if Zb_pdf > _A and Km_pdf > _A and (Q_pdf > _A) and (Weight[iis] > _A):
                        NBT = len([Q_sample[iis, ij]]) - 1
                        if not node_z_ini[NBS, ij]:
                            icost += _A
                        else:
                            if params.useEXT:
                                Z_SS, A_SS, P_SS, R_SS, cost = calc.compute_steady_state(NBS, NBT, node_xr, node_yr, node_z_ini[:, ij], node_x, Zb, Kmi3, [Q_sample[iis, ij]])
                            else:
                                Z_SS, A_SS, P_SS, R_SS, cost = calc.compute_steady_state(NBS, NBT, node_xr, node_yr, node_z_ini[:, ij], node_x, Zb, Kmi3, [Q_sample[iis, ij]])
                            icost += cost
                if Zb_pdf > _A and Km_pdf > _A and (Q_pdf > _A) and (icost > _A):
                    cost_all += icost.tolist()
                else:
                    cost_all += [np.nan]
                iis += 1
            if params.a31_early_stop:
                if (trial_ps0 == _A and i3 == 1 and (i2 == 1)) and theta == _B:
                    iexit = 1
                    break
                if trial_ps0 > trialps1_max:
                    trialps1_max = deepcopy(trial_ps0)
                if trialps1_max > _A:
                    if trial_ps0 / trialps1_max < 0.01:
                        break
        if i2 > 1:
            p_YMS_a += (p_YMS_a1 * ss_b + ss_a2 * ss_b2) / 2.0 * -(ZBA - ZBA0)
            for t in range(0, len(node_z[0])):
                QM[t] += (yarg_a1[t] * ss_b + yarg_a3[t] * ss_b2) / 2.0 * -(ZBA - ZBA0)
            Zb_acc += (Zb_acc2 * ss_b + Zb_acc3 * ss_b2) / 2.0 * -(ZBA - ZBA0)
            Kmi_acc += (Kmi_acc2 * ss_b + Kmi_acc3 * ss_b2) / 2.0 * -(ZBA - ZBA0)
        ss_a2 = deepcopy(p_YMS_a1)
        ss_b2 = deepcopy(ss_b)
        ZBA0 = deepcopy(ZBA)
        yarg_a3 = deepcopy(yarg_a1)
        Zb_acc3 = deepcopy(Zb_acc2)
        Kmi_acc3 = deepcopy(Kmi_acc2)
        temp_i3.append(i3)
        if params.a31_early_stop:
            if trialps1_max > trialps0_max:
                trialps0_max = deepcopy(trialps1_max)
            if trialps0_max > _A:
                if trialps1_max / trialps0_max < 0.01:
                    iexit = 1
            if i2 > 1 and iexit == 1:
                break
    if p_YMS_a > _A:
        Kmi_acc = Kmi_acc / p_YMS_a
        Zb_acc = Zb_acc / p_YMS_a
    else:
        Kmi_acc = np.nan
        Zb_acc = np.nan
    Q_min = QMT_array[0, :] * Km1
    Q_max = QMT_array[-1, :] * Km2
    Nbeta = p_YMS_a
    if Nbeta == 0:
        if param_dict['q_min_modif']:
            QM = Q_min
            reliability = 'unreliable'
        else:
            valid_output = 0
            reliability = 'invalid'
            QM = np.ones(node_z[0].shape[0]) * np.nan
    if Nbeta != 0:
        QM /= Nbeta
    if params.V32:
        if nsobs != 0 and (not check_na(nsobs)):
            lcmax = int(math.log(nsobs) / math.log(2.0)) + 1
        else:
            lcmax = np.nan
        distp0 = []
        lc0 = []
        lc0 += [lcmax]
        id = 0
        distp, LH_pdf, QPost = calc.likelihood_2(TDIM, nsobs, lc0[id], Q_sample, QM, cost_all, Weight)
        distp0 += [distp]
        while distp < params.thres and distp >= distp0[id] and (not check_na(distp)):
            id += 1
            lc0 += [lc0[id - 1] + 1]
            distp, LH_pdf, QPost = calc.likelihood_2(TDIM, nsobs, lc0[id], Q_sample, QM, cost_all, Weight)
            distp0 += [distp]
        while distp >= params.thres and distp <= distp0[id] and (not check_na(distp)):
            id += 1
            lc0 += [lc0[id - 1] - 1]
            distp, LH_pdf, QPost = calc.likelihood_2(TDIM, nsobs, lc0[id], Q_sample, QM, cost_all, Weight)
            distp0 += [distp]
        ilc = min(range(len(distp0)), key=lambda k: abs(distp0[k] - _B))
        lc = lc0[ilc]
        distp, LH_pdf, QPost = calc.likelihood_2(TDIM, nsobs, lc, Q_sample, QM, cost_all, Weight)
        QM = QPost
    for t in range(node_z[0].shape[0]):
        if QM[t] < params.local_QM1:
            QM[t] = params.local_QM1
    mask_qm = np.ones(len(QM), dtype=bool)
    for i in range(len(QM)):
        if not check_na(QM[i]):
            mask_qm[i] = _E
    QM_masked = np.ma.array(QM, mask=mask_qm)
    if param_dict[A]:
        reach_id = str(reach_id)
        reach_id = verify_name_length(reach_id)
        reach_id = verify_name_length(reach_id)
        node_x = node_x
        test_t_array = reach_t
        nodes2 = (node_x - node_x[0]) / 1000
        times2 = test_t_array / 3600 / 24
        if not output_dir.is_dir():
            output_dir.mkdir(parents=_D, exist_ok=_D)
        gnuplot_save_zm(ZM_array_save, times2, output_dir.joinpath('ZM_array'), spaces=1)
    if param_dict[A]:
        reach_id = str(reach_id)
        reach_id = verify_name_length(reach_id)
        reach_id = verify_name_length(reach_id)
        node_x = node_x
        test_t_array = reach_t
        nodes2 = (node_x - node_x[0]) / 1000
        times2 = test_t_array / 3600 / 24
        if not output_dir.is_dir():
            output_dir.mkdir(parents=_D, exist_ok=_D)
        gnuplot_save_q(QM, times2, output_dir.joinpath('qalgo31'))
    if param_dict[A]:
        reach_id = str(reach_id)
        reach_id = verify_name_length(reach_id)
        reach_id = verify_name_length(reach_id)
        output_dir = output_dir.joinpath(F, reach_id)
        node_x = node_x
        test_t_array = reach_t
        nodes2 = (node_x - node_x[0]) / 1000
        times2 = test_t_array / 3600 / 24
        if not output_dir.is_dir():
            output_dir.mkdir(parents=_D, exist_ok=_D)
    if param_dict[A]:
        reach_id = str(reach_id)
        reach_id = verify_name_length(reach_id)
        reach_id = verify_name_length(reach_id)
        output_dir = param_dict[C].joinpath(F, reach_id)
        if not output_dir.is_dir():
            output_dir.mkdir(parents=_D, exist_ok=_D)
        q_pdf_table_saved = q_pdf_table / np.max(q_pdf_table)
        zb_pdf_table_saved = zb_pdf_table / np.max(zb_pdf_table)
        km_pdf_table_saved = km_pdf_table / np.max(km_pdf_table)
        gnuplot_save_tables(q_pdf_table_saved, zb_pdf_table_saved, km_pdf_table_saved, output_dir.joinpath('tables_modified'), 1)
    if _E:
        plt.plot(QM_masked)
        plt.show()
        plt.clf()
    return (QM_masked, valid_output, reliability, Kmi_acc, Zb_acc)