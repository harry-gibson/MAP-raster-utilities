Generic MAP raster processing code
----------------------------------

Repository contains core raster-processing algorithms developed and used by Harry Gibson to process raster datasets in line with MAP requirements. These include:
* Aggregation of continuous-type rasters (decreasing resolution, taking mean / min / max etc of input cells)
* Aggregation of categorical-type rasters (decreasing resolution, taking majority / class fraction etc of input cells)
* Extracting subsets of existing rasters
* Matching categorical-type rasters (such as rasterised admin units) to a template (such as the MAP coastline template)
* Matching continuous-type rasters (such as population grids) to a template (such as the MAP coastline template), whilst maintaining totals
* Matching rasters (the outputs of these algorithms or anything else) to the MAP mastergrid lat/lon (EPSG:4326) templates (i.e. ensuring that cell size and alignment match)

Core processing algorithms are written in Cython, so this package needs to be built and installed. 

Jupyter notebooks are provided in the root folder of the repository to demonstrate the use of all these core processing algorithms, it is envisaged that most use will be by taking a copy of one of these notebooks and just modifying file paths to suit.
