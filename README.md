Generic MAP raster processing code
----------------------------------

Repository contains several ipython notebooks developed and used by Harry Gibson to process raster datasets in line with MAP requirements.

* Aggregate_Rasters.ipynb: Spatially aggregate (resample down) continuous or categorical rasters (to generate 5km cubes, landcover summaries, etc)
* CalcMeanAndSD.ipynb: Calculate synoptic data from MODIS cubes, including from-daily and from-monthly outputs
* DiurnalDifference.ipynb: Calculate difference between LST Day and LST Night images
* Extract_And_Align_Rasters.ipynb: Extracts sub-images from global 1km MODIS imagery, optionally realigning geotransform to match the slightly inaccurate version used in other MAP datasets
* Mean_Topographic_Correction.ipynb: Code for reprocessing mean / standard deviation dataset for BRDF imagery prior to gapfilling, to reduce the anomalies caused by severe topographic occlusion in mountainous areas
* PopulationReallocator.ipynb: Provides land-sea clipping functionality for population data or any other data where totals must be maintained 
