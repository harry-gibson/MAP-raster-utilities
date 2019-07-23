# The helper class
from  raster_utilities.aggregation.spatial.SpatialAggregator import *

def callAgg(inArgs):
    f, outDir, ndvOut, stats, aggArgs = inArgs
    try:
        agg = SpatialAggregator([f], outDir, ndvOut, stats, aggArgs)
        agg.RunAggregation()
    except KeyboardInterrupt, e:
        pass
