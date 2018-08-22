import os
import subprocess
import glob
from TiffFile import SingleBandTiffFile, RasterProps

class tileProcessor:
    """Handles splitting an output extent into tiles and then merging these tiles into a global output

    Writing a subset of a global tiff tend to corrupt for some reason in LZW files. So instead we write individual
    whole-file tiles and then merge these afterwards."""
    def __init__(self, outputPath, overallProps):
        self._tileFiles = []
        self._outputFileNoExt = ""
        assert isinstance(overallProps, RasterProps)
        self._overallProps = overallProps


    def AddTile(self, tileFile):
        if os.path.exists(tileFile):
            self._tileFiles.append(tileFile)

    def FindTiles(self, tilePattern):
        tileFiles = glob.glob(tilePattern)
        self._tileFiles.extend(tileFiles)

    def buildOutput(self, cleanupIntermediates=True):
        outFile = self._getTiffOutput()
        if os.path.exists(outFile) and cleanupIntermediates:
            for f in self._tileFiles:
                os.remove(f)
            vrtFile = self._outputFileNoExt + ".vrt"
            os.remove(vrtFile)
        return SingleBandTiffFile(outFile)

    def _getTiffOutput(self):
        transCommandTemplate =  "gdal_translate -of GTiff -co COMPRESS=LZW " + \
                       "-co PREDICTOR=2 -co TILED=YES -co BIGTIFF=YES " + \
                       "-co NUM_THREADS=ALL_CPUS --config GDAL_CACHEMAX 8000 {0} {1}"

        tiffName = self._outputFileNoExt + ".tif"
        if os.path.exists(tiffName):
            return tiffName
        vrtName = self.getVrtOutput()
        transCommand = transCommandTemplate.format(tiffName, vrtName)
        txt = subprocess.check_output(transCommand)
        return tiffName

    def _getVrtOutput(self):
        """Builds a vrt file from the input tile, if not already done, and returns its path
        TODO - check inputs are appropriately aligned and only 1 band etc"""
        vrtCommandTemplate = "gdalbuildvrt {0} {1}"
        vrtName = self._outputFileNoExt + ".vrt"
        if os.path.exists(vrtName):
            return vrtName
        else:
            vrtCommand = vrtCommandTemplate.format(vrtName, self._tileFiles)
            # raises an exception if the command fails (nonzero exit code)
            txt = subprocess.check_output(vrtCommand)
        return vrtName


