# This file supports the Spatial_Aggregator_Continuous_Usage notebook. A peculiarity of how Jupyter works requires 
# a helper function to be imported from an external (to the notebook) module. 

# The helper class
from  raster_utilities.aggregation.spatial.SpatialAggregator import *

def callAgg(inArgs):
    f, outDir, ndvOut, stats, aggArgs = inArgs
    try:
        agg = SpatialAggregator([f], outDir, ndvOut, stats, aggArgs)
        agg.RunAggregation()
    except KeyboardInterrupt, e:
        pass
