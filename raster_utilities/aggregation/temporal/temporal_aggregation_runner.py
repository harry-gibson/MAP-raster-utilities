from temporal_aggregation import TemporalAggregator_Dynamic
from General_Raster_Funcs.GeotransformCalcs import *
def temporalAggregationRunner(filesDict, top, bottom, width, outputNDV, stats, doSynoptic):
    '''Runs temporal aggregation across a set of files.

    Files should be provided in filesDict; each item should be a key representing an
    output point (timespan) and a value representing the file paths for that point in time.
    The keys could be months (e.g. "2001-02"), years (e.g. "2002"), synoptic months
    (e.g. "February"), or anything else. If there is only one item then only synoptic outputs
    can be generated.

    All of the files (across all output time points) must have the same geotransform and extent.

    top and bottom allow specification of a subset (horizontal slice) of the files to run the
    aggregation for, in case there isn't enough memory to do the whole extent in one go. In this
    case output files will have a suffix indicating the top pixel coordinate relative to the incoming
    files.

    width must match the overall width of the incoming files (vertical slicing isn't supported)

    outputNDV doesn't have to match the incoming NDV

    stats is a list containing some or all of
    ["min", "max", "mean", "sd", "sum", "count"]
    The more statistics are specified, the more memory is required.

    doSynoptic specifies whether "overall" statistics should be calculated in addition to one
    per timestep - this doubles memory use. This has no effect if filesDict only has one item.
    '''
    if not (isinstance(bottom, int) and isinstance(top, int) and isinstance(width, int)):
        raise TypeError("top, bottom and width must be integer values")
    if not ((bottom > top ) and top >= 0):
        raise ValueError("bottom must be greater than top and top must be GTE zero")
    sliceHeight = bottom - top
    nPix = sliceHeight * width
    bpp = {"count": 2, "mean": 8, "sd": 8, "min": 4, "max": 4, "sum":4}
    try:
        bppTot = sum([bpp[s] for s in stats])
    except KeyError:
        raise KeyError("Invalid statistic specified!")
    if (("sd" in stats) and ("mean" not in stats)):
        bppTot += bpp["mean"]
    bppTot *= nPix
    runSynoptic = (len(filesDict.keys()) > 1) and doSynoptic
    if runSynoptic:
        bppTot *= 2
    gb = bppTot / 2e30
    if gb > 30:
        print("Requires more than 30GB, are you sure this is wise....")

    statsCalculator = TemporalAggregator_Dynamic(sliceHeight, width, outputNDV, stats, runSynoptic)
    sliceGT = None
    sliceProj = None
    isFullFile = False
    for timeKey, timeFiles in filesDict.iteritems():
        logmsg(timeKey)
        for timeFile in timeFiles:
            data, thisGT, thisProj, thisNdv = ReadAOI_PixelLims(timeFile, None, (top, bottom))
            if sliceGT is None:
                # first file
                sliceGT = thisGT
                sliceProj = thisProj
                props = GetRasterProperties(timeFile)
                if sliceHeight == props["height"]:
                    isFullFile = True
            else:
                if sliceGT != thisGT or sliceProj != thisProj:
                    raise




