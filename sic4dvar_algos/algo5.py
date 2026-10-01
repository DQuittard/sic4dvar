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

"""
import inspect
import scipy.optimize as sciop
from lib.lib_variables import equations_dict
from sic4dvar_functions.sic4dvar_helper_functions import *

def bounds_preparation(_dA, _w, equations_dict, equation='ManningLW'):
    epsilon = 0.01
    equations_dict[equation]['bounds'] = {}
    if 'a0' in equations_dict[equation]['parameters']:
        '\n        if _dA.max() < 1.0:\n            a_min, a_max = abs(_dA.min())*100, abs(_dA.min())*100 + abs(_dA.max())*100\n        else:\n            a_min, a_max = abs(_dA.min())*1.5, math.pow(_dA.max(),3) ## Bounds of dA\n        #TODO abs puis min et max (if abs(dA))\n        '
        dA_error = np.where(_dA < params.valid_min_dA)
        _dA[dA_error] = np.nan
        bnds_min = -np.nanmin(_dA) + epsilon
        bnds_max = max(np.nanmax(_w) * 100, 10 * abs(bnds_min))
        equations_dict[equation]['bounds']['a0'] = tuple((bnds_min, bnds_max))
    if 'n' in equations_dict[equation]['parameters']:
        bnds_min, bnds_max = (0.001, 1)
        equations_dict[equation]['bounds']['n'] = tuple((bnds_min, bnds_max))
    if 'cf' in equations_dict[equation]['parameters']:
        bnds_min, bnds_max = (0.001, 1)
        equations_dict[equation]['bounds']['cf'] = tuple((bnds_min, bnds_max))
    if 'alpha' in equations_dict[equation]['parameters']:
        bnds_min, bnds_max = (0.001, 1)
        equations_dict[equation]['bounds']['alpha'] = tuple((bnds_min, bnds_max))
    if 'beta' in equations_dict[equation]['parameters']:
        bnds_min, bnds_max = (-1, -0.0001)
        equations_dict[equation]['bounds']['beta'] = tuple((bnds_min, bnds_max))
    return equations_dict

def algo5(_q_ref, _dA, _w, _s, reach_id, equations_dict=equations_dict, equation='ManningLW', _z=[], param_dict=[]):
    equations_dict = bounds_preparation(_dA, _w, equations_dict, equation)
    data_to_use_dict = {}
    data_to_use_dict['q'] = np.copy(_q_ref)
    data_to_use_dict['abar'] = np.copy(_dA.data)
    data_to_use_dict['w'] = np.copy(_w.data)
    data_to_use_dict['s'] = np.copy(_s.data)
    '\n        if np.array(_z).size > 0:\n            data_to_use = np.zeros((5, _q_ref.shape[0]))\n        else:\n            data_to_use = np.zeros((4, _q_ref.shape[0]))\n\n        #data_to_use = np.zeros((4, len(_dA)))\n        data_to_use[0] = np.copy(_q_ref)\n        \n        data_to_use[1] = np.copy(_dA.data)\n        \n\n        data_to_use[2] = np.copy(_w.data)\n        \n        data_to_use[3] = np.copy(_s.data)\n        '
    if np.array(_z).size > 0:
        data_to_use_dict['wse'] = np.copy(_z.data)
    bnds = tuple((equations_dict[equation]['bounds'][name] for name in equations_dict[equation]['parameters']))
    if params.cython_version:
        results = sciop.differential_evolution(calc.objective_internal_data_cy, bounds=bnds, args=data_grouped_tuple, polish=True)
        a0, n = (results.x[0], results.x[1])
        q_est = np.copy(_q_ref)
        for i in range(q_est.shape[0]):
            if (data_to_use[i, 3] <= 0.0 or check_na(data_to_use[i, 3])) or (data_to_use[i, 2] <= 0.0 or check_na(data_to_use[i, 2])) or (data_to_use[i, 1] <= params.valid_min_dA or check_na(data_to_use[i, 1])):
                q_est[i] = np.nan
            else:
                q_est[i] = cy.calc_q_manning_strickler_with_n(a0, data_to_use[i, 1], data_to_use[i, 3], n, data_to_use[i, 2])
    else:
        data_to_use_tuple = tuple(data_to_use_dict.values())
        args = (equation, data_to_use_tuple, tuple(equations_dict[equation]['bounds'].keys()))
        if False:
            results = sciop.differential_evolution(calc.objective_internal_data_Q_any, bounds=bnds, args=args, polish=True, strategy='best1bin')
        else:
            initial_guesses = []
            for key in list(equations_dict[equation]['bounds'].keys()):
                initial_guesses.append((equations_dict[equation]['bounds'][key][1] - equations_dict[equation]['bounds'][key][0]) / 2)
            logging.debug(f'BOUNDS: {bnds[0]}')
            new_lb = [bnds[0][0], bnds[1][0]]
            new_ub = [bnds[0][1], bnds[1][1]]
            residuals = calc.objective_internal_data_Q_any(initial_guesses, *args)
            print(_q_ref, _dA, _w, _s)
            if not np.all(np.isfinite(residuals)):
                logging.error(f'Non-finite residuals: {residuals}')
                logging.error(f'Initial guesses: {initial_guesses}')
                raise ValueError('Initial residuals are not finite')
            results = sciop.least_squares(calc.objective_internal_data_Q_any, x0=initial_guesses, bounds=(new_lb, new_ub), args=args)
        '\n            if equation == "ManningLW":\n                #Q_Manning_LW (originally used in algo5)\n                results = sciop.differential_evolution(calc.objective_internal_data_Q_Manning_LW, bounds=bnds,                 args=args, polish=True, strategy="best1bin")\n\n            if equation == "DarcyW":\n                #Q_DarcyW\n                results = sciop.differential_evolution(calc.objective_internal_data_Q_DarcyW, bounds=bnds,                 args=data_grouped_tuple, polish=True, strategy="best1bin")\n\n            if equation == "ManningVK":\n                #Q_ManningVK\n                results = sciop.differential_evolution(calc.objective_internal_data_Q_Manning_VK, bounds=bnds,                 args=data_grouped_tuple, polish=True, strategy="best1bin")\n            \n            if equation == "ManningLW" or equation == "DarcyW":\n                a0, n = results.x[0], results.x[1]\n\n            elif equation == "ManningVK":\n                a0, a, b = results.x[0], results.x[1], results.x[2]\n                logging.debug(f"A: {a}, B: {b}")\n            '
        results_dict = {}
        for j in range(0, len(results.x)):
            results_dict[list(equations_dict[equation]['bounds'].keys())[j]] = results.x[j]
        logging.info(f'Convergence success with strat best1bin: {results.success}')
        '\n            old_cost = results.fun\n            \n            temp_index = []\n            temp_cost = []\n            \n            other_results = []\n            costs = []\n            success = [] \n            \n            \n            for j in range(0,1):\n\n                other_results = []\n                costs = []\n                success = []    \n            \n                if not results.success or param_dict[\'algo5_strats\']:\n                    #print(bug)\n                    \n                    if not param_dict[\'algo5_strats\']:\n                        other_strats = ["best1exp","rand1exp","randtobest1exp","currenttobest1exp",                                    "best2exp","rand2exp","randtobest1bin","currenttobest1bin",                                        "best2bin","rand2bin","rand1bin"]\n                    \n                    if param_dict[\'algo5_strats\']:\n                        other_strats = ["best1bin", "best1exp","rand1exp","randtobest1exp","currenttobest1exp",                                "best2exp","rand2exp","randtobest1bin","currenttobest1bin",                                    "best2bin","rand2bin","rand1bin"]\n                        \n    \n                    for strat in other_strats:\n                        results = sciop.differential_evolution(calc.objective_internal_data, bounds=bnds,                         args=data_grouped_tuple, polish=True, strategy=strat)\n                        other_results.append([results.x[0],results.x[1]])\n                        costs.append(results.fun)\n                        success.append(results.success)\n                        logging.info(f"Convergence success with strat {strat}: {results.success}")\n                    #print(bug)\n                    \n    \n                try: \n                    index_best = np.nanargmin(costs)\n                    temp_index.append(index_best)\n                    temp_cost.append(costs[index_best])\n                except ValueError:\n                    index_best = []\n                #temp_cost.append(costs[index_best])\n            \n            try: \n                index_best = np.nanargmin(temp_cost)\n                index_best = temp_index[index_best]\n                #print(bug)\n\n            except ValueError:\n                index_best = []\n            #print(buggy)\n            \n            if np.array(index_best).size > 0:\n                \n                if not param_dict[\'algo5_strats\']:\n                    if results.fun > costs[index_best] or check_na(results.fun):\n                        a0 = other_results[index_best][0]\n                        n = other_results[index_best][1]\n                        logging.info(f"Best strat: {other_strats[index_best]}, cost = {costs[index_best]}")\n                    \n                    else:\n                        logging.info(f"Best strat: best1bin, cost = {results.fun}")\n                        \n                if param_dict[\'algo5_strats\']:\n                    if results.fun > costs[index_best] or check_na(results.fun):\n                        a0 = other_results[index_best][0]\n                        n = other_results[index_best][1]\n                        logging.info(f"Best strat: {other_strats[index_best]}, cost = {costs[index_best]}")\n                \n            #print(f"Best strat: {other_strats[index_best]}, cost = {costs[index_best]}")\n                        \n                #f = open(str(params.output_dir) + "/" + str(reach_id) + ".txt", "a")\n                #f.write(f"{other_strats[index_best]}")\n                #f.close()\n            \n            '
        q_est = np.copy(_q_ref)
        '\n            if equation == "ManningVK":\n                mean_n = []\n                a_tot = []\n                b_tot = []\n\n            '
        for i in range(q_est.shape[0]):
            if (data_to_use_dict['s'][i] <= 0.0 or check_na(data_to_use_dict['s'][i])) or (data_to_use_dict['w'][i] <= 0.0 or check_na(data_to_use_dict['w'][i])) or (data_to_use_dict['abar'][i] <= params.valid_min_dA or check_na(data_to_use_dict['abar'][i])):
                q_est[i] = np.nan
            else:
                print(f'Compute {equation} equation with parameters !')
                if calc.check_suitability_any(results_dict, data_to_use_dict, equation, i):
                    print('Suitable for computation.')
                    params_func = inspect.signature(getattr(calc, equation)).parameters
                    param_names = list(params_func.keys())
                    function_params = np.zeros(len(param_names))
                    for j, parameter in enumerate(param_names):
                        index_param = np.where(np.array(param_names) == parameter)[0]
                        if np.array(index_param).size > 0:
                            if parameter in data_to_use_dict:
                                function_params[j] = data_to_use_dict[parameter][i]
                            elif parameter in results_dict:
                                function_params[j] = results_dict[parameter]
                    func_params_dict = dict(zip(param_names, function_params))
                    q_est[i] = getattr(calc, equation)(**func_params_dict)
                else:
                    q_est[i] = np.nan
                '\n                    if equation == "ManningLW":\n                        q_est[i] = calc.calc_q_manning_strickler_with_n(a0, data_to_use[i, 1],                         data_to_use[i, 3], n, data_to_use[i, 2]) ## Estimate discharge\n                    if equation == "DarcyW":\n                        q_est[i] = calc.calc_q_darcyw(a0, data_to_use[i, 1],                         data_to_use[i, 3], n, data_to_use[i, 2]) ## Estimate discharge\n                    if equation == "ManningVK":\n                        #h=_z.data-np.nanmin(_z.data)+a0/_w.data\n                        max_w = np.maximum(10, _w[i].data)\n                        h=(a0+_dA[i].data)/max_w\n                        #print("h:", h)\n                        #print(bug)\n                        #n = a * (h[i] ** b)\n                        n = a * (h ** b)\n                        #print("N:", n, "H:", h)\n                        a_tot.append(a)\n                        b_tot.append(b)\n                        q_est[i] = calc.calc_q_manning_strickler_with_n(a0, data_to_use[i, 1],                         data_to_use[i, 3], n, data_to_use[i, 2]) ## Estimate discharge\n                        mean_n.append(n)\n                    '
    '\n        ## make masked array\n        mask_q_est = np.ones(len(q_est), dtype=bool)\n        for i in range(len(q_est)):\n            if not check_na(q_est[i]): #modif DQ #np.isnan(q_est[i])\n                mask_q_est[i] = False\n        q_est_masked = np.ma.array(q_est, mask=mask_q_est)\n        \n        '
    q_est_masked = q_est
    for bound in equations_dict[equation]['bounds'].keys():
        print(equations_dict[equation]['bounds'][bound][1])
        if results_dict[bound] < equations_dict[equation]['bounds'][bound][0]:
            results_dict[bound] = equations_dict[equation]['bounds'][bound][0]
            logging.info(f'{bound} outside of bounds, set to lower bound: {results_dict[bound]}')
        elif results_dict[bound] > equations_dict[equation]['bounds'][bound][1]:
            results_dict[bound] = equations_dict[equation]['bounds'][bound][1]
            logging.info(f'{bound} outside of bounds, set to upper bound: {results_dict[bound]}')
    return (q_est_masked, results_dict)