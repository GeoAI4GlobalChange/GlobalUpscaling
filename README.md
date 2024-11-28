# Global wetland CH4 upscaling
The repository contains codes of upscaling and analyzing wetland methane emissions at the global scale. Specifically, we used causality guided machine learning models, eddy covariance and chamber measurements of CH4 fluxes, and environmental predictors (including soil temperature, air temperature, gross primary productivity, air pressure, precipitation, wind speed, snow cover, soil water content, soil PH, and soil texture) to  upscale global wetland CH4 emissions from 2003 to 2023. We then identified dominant controls on wetland CH4 emission trends regionally and globally using statistical attribution models.  
## Code introduction
1)	Causality-guided machine learning models  
In the ”Causal_ML_Demo.py” file, it contains code to train causality-guided machine learning models.    
2) Quantifying dominant controls on wetland CH4 emission trends  
In the”methane_analysis_demo.py” file, run the “grid_trend_analysis” function.  

## Data availability
The original eddy covariance methane flux datasets are from FLUXNET-CH4 (https://fluxnet.org/data/fluxnet-ch4-community-product/) and AmeriFlux (https://ameriflux.lbl.gov/sites/site-search/). The chamber datasets we compiled will be publicly available along with our paper.  
For the input drivers, GPP was obtained from the GOSIF dataset, which is available at https://globalecology.unh.edu/data/GOSIF-GPP.html. Other variables (soil temperature, air temperature, air pressure, precipitation, wind speed, snow cover, and soil water content) were obtained from ERA5-land, available at https://cds.climate.copernicus.eu/datasets/derived-era5-land-daily-statistics. The monthly global dataset of water storage/height anomalies is from GRACE-FO (https://doi.org/10.5067/TEMSC-3JC63).  
For more datasets used for uncertainty analysis, please see the correponding paper.

## Computing environment  
1) Packages required: netCDF4, pandas, numpy, scipy, pytorch, scikit-learn, pickle, statsmodels  
2) Anaconda is recommended to install all the packages required by runing: Conda install list_of_packages  
3) The code has been tested on: Windows 11 version 23H2 with Python 3.8.19 and Intel(R)Core(TM)i9-9900K CPU    

## References
Details of the research will be seen in our paper (under review now). If you have any questions about the code or dataset, please contact kunxiaojiayuan@lbl.gov or fali2@stanford.edu.  
