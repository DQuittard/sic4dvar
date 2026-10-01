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
code_to_continent = {1: 'AF', 2: 'EU', 3: 'AS', 4: 'AS', 5: 'OC', 6: 'SA', 7: 'NA', 8: 'NA', 9: 'NA'}
continent_to_station = {'AF': ['DWA'], 'AS': ['MLIT'], 'EU': ['EAU', 'DEFRA'], 'NA': ['WSC', 'USGS', 'MEFCCWP'], 'OC': ['ABOM'], 'SA': ['DGA']}
station_to_continent = {'DWA': 'AF', 'EAU': 'EU', 'WSC': 'NA', 'ABOM': 'OC', 'USGS': 'NA', 'DEFRA': 'EU', 'MEFCCWP': 'NA', 'MLIT': 'AS'}
constraint_to_model_name = {'constrained': 'GRADES', 'unconstrained': 'WBM'}
equations_dict = {}
equations_dict['ManningLW'] = {'parameters': ['a0', 'n']}
equations_dict['DarcyW'] = {'parameters': ['a0', 'cf']}
equations_dict['ManningVK'] = {'parameters': ['a0', 'alpha', 'beta']}