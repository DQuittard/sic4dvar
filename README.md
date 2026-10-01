Authors: Callum DA Tyler, Pierre-Olivier Malaterre, Hind Oubanas, Igor Gejadze, Isadora Rezende De Oliveira, Dylan Quittard, Cécile Cazals
Documentation last updated: 2025 April 1st
Person to contact for issues: Dylan Quittard, dylan.quittard@inrae.fr

# SWOT INRAe Algorithm 315 (SIC4DVar)
## Purpose
Estimates the discharge of a river using a chain of two algorithm, algo31-lc and algo5. Algo31-lc estimates a rivers discharge through processing the rivers node level data of width and elevation. Algo5 uses the rivers width, elevation, cross-sectional area, slope and estimated discharge from algo31-lc to estimate the minimum wetter cross-sectional area and the roughness (mannings coefficient) of the river bed. Then finally re-estimates the rivers discharge using the latest data. This version of the algo315 is designed to run on the Confluence platform.

## Overview
Parameters will be loaded from sic4dvar_params.py. A reach ID will be passed to the algorithm which will be used to identify which data to load. The input data will be loaded from a netCDF file. The netCDF conforms to the SWORD fromat. The algorithms require the following parameters: time series of at node level: river width & river elevation; time series at reach level: river width, slope, elevation, cross-sectional area. Results will be output to another netCDF file. The results contain estimated river discharge from algo31-lc, estimated river discharge from algo5 and the estimated wetted cross-sectional area and the estimated river bed roughness (manning's coefficient) of the river. In the event that the algorithms cannot run, the netCDF will contain Numpy NaNs for each of these four variables.  
Algo315 is designed to run on one reach of a river at a time. There is no internal parallelisation, all processes are sequential.

## Input
### Data
The data used for sic4dvar comes from the SWORD database and SWOT observations. They contain, at a reach and node level, the data variables that we require for estimating the discharge of rivers. These include: width, elevation, slope, cross-sectional area. Further information can be found [here](https://podaac-tools.jpl.nasa.gov/drive/files/misc/web/misc/swot_mission_docs/pdd/D-56413_SWOT_Product_Description_L2_HR_RiverSP_20200825a.pdf). This data is contained within separate NetCDF files for each continent (SWORD). The input data must adhere to the format mentioned in the linked documentation.
The input path can be specified in the configuration file that you will use.

### Parameter file
To allow the user to configure the processing of river data, a parameter file, 'sic4dvar_params.py' has been created and is loaded automatically when the sic4dvar.py is ran. 
Additionnally, sic4dvar now uses configuration files (.ini files) to load most parameters.

## Output
### Data
This program will output the results of the algorithms into another netCDF file. Appropriate names: '[reach_id]_sic4dvar.nc', i.e., '77449100131_sic4dvar.nc'. A netCDF will contain :  
- Qalgo31, estimated discharge of reach computed by algo31-lc  
- Qalgo5, estimated discharge of reach compute by algo5
- A0, estimated wetted area computed by algo5  
- n, estimated mannings coefficient computed by algo5  

In the event that the algorithms could not estimate these variables then NaNs will be found in the netCDF.

### Log
For every time the sic4dvar program is run a log file will be generated per reach. These will contain information about processing the reaches data. The location it will be saved can be specified in the input parameter file. 

## Usage
To use, configure the variables in sic4dvar_params.py and the configuration file, the run the following command:  
`python3 sic4dvar.py`  

