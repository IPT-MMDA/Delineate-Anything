from osgeo import gdal, ogr, osr

gpkg_ds = ogr.Open("PATH_TO_GEOPACKAGE")

# pixel size metadata to write
pixel_size = [5, 5]

gpkg_ds.SetMetadata({
        "VERTEX_STEP_X": pixel_size[0],
        "VERTEX_STEP_Y": pixel_size[1]
    })