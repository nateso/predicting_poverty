import ee
from datetime import datetime, timedelta

# define the Landsat class
class LandsatSR:
    def __init__(self, point: ee.Geometry, start_ts: str) -> None:
        '''
        Args
        - point: ee.Geometry
        - start_date: str, string representation of start date
        '''
        self.point = point

        self.start_date = datetime.strptime(start_ts, "%Y-%m-%d").date()
        self.end_date = self.start_date + timedelta(days=365)

        # for now just use LS7 images (they are available for the full time period)
        # however, there is an issue in the sensor resulting in weird stripes on the images
        # The stripe is in different locations though every time an image is taken, thus when taking
        # the median this should not matter too much (let's hope - hehe)
        self.l5 = self.init_coll('LANDSAT/LT05/C02/T1_L2').map(self.select_rename_l57).map(self.rescale)
        self.l7 = self.init_coll('LANDSAT/LE07/C02/T1_L2').map(self.select_rename_l57).map(self.rescale)
        self.l8 = self.init_coll('LANDSAT/LC08/C02/T1_L2').map(self.select_rename_l8).map(self.rescale)

        self.merged = self.l5.merge(self.l7).merge(self.l8).sort('system:time_start')
        self.merged_cr = self.merged.map(self.remove_clouds)

        self.scale = self.merged.first().select('BLUE').projection().nominalScale().getInfo()
        self.visParams = {'bands': ['RED', 'GREEN', 'BLUE'], 'min': 0.0, 'max': 0.3}
        self.nImg = self.merged.size().getInfo()

    def init_coll(self, name: str) -> ee.ImageCollection:
        '''
        Creates a ee.ImageCollection containing images of desired points
        between the desired start and end dates.

        Args
        - name: str, name of collection

        Returns: ee.ImageCollection
        '''
        img_col = ee.ImageCollection(name) \
            .filterBounds(self.point) \
            .filterDate(str(self.start_date), str(self.end_date))  # filter the correct channels of the image

        return img_col

    @staticmethod
    def select_rename_l8(img: ee.Image) -> ee.Image:
        '''
        Args
        - img: ee.Image, Landsat 8 image

        Returns
        - img: ee.Image, with selected channels and renamed
        '''

        img = img.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                          'ST_B10', 'QA_PIXEL', 'QA_RADSAT'])

        newnames = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2',
                    'TEMP1', 'pixel_qa', 'radsat_qa']

        return img.rename(newnames)

    @staticmethod
    def select_rename_l57(img: ee.Image) -> ee.Image:
        '''
        Args
        - img: ee.Image, Landsat 5/7 image

        Returns
        - img: ee.Image, with selected channels and renamed
        '''
        img = img.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'ST_B6',
                          'SR_B7', 'QA_PIXEL', 'QA_RADSAT'])
        newnames = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2',
                    'TEMP1', 'pixel_qa', 'radsat_qa']
        return img.rename(newnames)

    @staticmethod
    def rescale(img: ee.Image) -> ee.Image:
        '''
        Args
        - img: ee.Image, Landsat 5/7/8

        Returns
        - img: ee.Image, with rescaled

        See: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C01_T1_SR

        Name                Scale Factor  Offset  Description
        BLUE                0.0000275     -0.2    Band 1 (blue) surface reflectance, 0.45-0.52 um
        GREEN               0.0000275     -0.2    Band 2 (green) surface reflectance, 0.52-0.60 um
        RED                 0.0000275     -0.2    Band 3 (red) surface reflectance, 0.63-0.69 um
        NIR                 0.0000275     -0.2    Band 4 (near infrared) surface reflectance, 0.77-0.90 um
        SWIR1               0.0000275     -0.2    Band 5 (shortwave infrared 1) surface reflectance, 1.55-1.75 um
        SWIR2               0.0000275     -0.2    Band 6 (shortwave infrared 2) surface reflectance, 2.08-2.35 um
        TEMP1               0.00341802     149    Band 7 brightness temperature (Kelvin), 10.40-12.50 um

        pixel_qa                                  Pixel quality attributes generated from the CFMASK algorithm,
                                                  see Pixel QA table
        radsat_qa                                 Radiometric saturation QA, see Radiometric Saturation QA table
        '''
        opt = img.select(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'])
        therm = img.select(['TEMP1'])
        masks = img.select(['pixel_qa', 'radsat_qa'])

        opt = opt.multiply(0.0000275).add(-0.2)
        therm = therm.multiply(0.00341802).add(149)

        scaled = ee.Image.cat([opt, therm, masks]).copyProperties(img)
        # system properties are not copied
        scaled = scaled.set('system:time_start', img.get('system:time_start'))
        return scaled

    # genereal functions

    def remove_clouds(self, img: ee.Image) -> ee.Image:
        '''
        Args
        - img: ee.Image

        Returns
        - img: ee.Image, input image with cloud-shadow, cloud and unclear pixels over land removed
        '''
        mask_img = self.decode_qamask(img)
        cloudshadow_mask = mask_img.select('pxqa_cloudshadow')
        cloud_mask = mask_img.select('pxqa_cloud')
        # clear_mask = mask_img.select('pxqa_clear')
        # no_water_mask = mask_img.select('no_water')
        # clear_no_water_mask = clear_mask.multiply(no_water_mask)

        return img.updateMask(cloudshadow_mask).updateMask(cloud_mask)  # .updateMask(cear_no_water_mask)

    @staticmethod
    def decode_qamask(img: ee.Image) -> ee.Image:
        '''
        Args
        - img: ee.Image, LS 5/7/8 image containing pixel_qa band

        Returns
        - masks: ee.Image, where each band is one mask clear, cloud, cloud_shadow

        Pixel QA Bit Flags
        Bit  Attribute
        0    Fill
        1    Dilated Cloud
        2    Unused
        3    Cloud --> 1 << 3  = 8
        4    Cloud Shadow --> 1 << 4 = 16
        5    Snow
        6    Clear
        7    Water --> 1 << 7 =

        '''
        qa = img.select('pixel_qa')

        cloud = qa.bitwiseAnd(1 << 3).eq(0)  # 0 = cloud, 1 = not cloud
        cloud = cloud.updateMask(cloud).rename(['pxqa_cloud'])

        cloud_shadow = qa.bitwiseAnd(1 << 4).eq(0)  # 0 = shadow, 1 = not shadow
        cloud_shadow = cloud_shadow.updateMask(cloud_shadow).rename(['pxqa_cloudshadow'])

        clear = qa.bitwiseAnd(1 << 6).neq(0)  # 0 = not clear, 1 = clear
        clear = clear.updateMask(clear).rename(['pxqa_clear'])

        no_water = qa.bitwiseAnd(1 << 7).neq(0)  # 0 = water, 1 no_water
        no_water = no_water.updateMask(no_water).rename(['no_water'])

        mask_img = ee.Image.cat([clear, cloud, cloud_shadow, no_water])
        return mask_img

# Create a Modis class
class ModisSR:
    def __init__(self, point: ee.Geometry, start_ts: str):
        self.start_date = datetime.strptime(start_ts, "%Y-%m-%d").date()
        self.end_date = self.start_date + timedelta(days = 365)
        self.point = point

    #***************************************************************************
    # functions to initiatlise the different inner classes --> avoids loading all at once (slow)

    def init_SR_500(self):
        return self.SR_500(self)

    def init_SR_250(self):
        return self.SR_250(self)

    #***************************************************************************
    # Parent functions to load the data

    def init_coll(self, name: str):
        '''
        initiates the image collection and restricts it to the correct time
        '''
        img_col = ee.ImageCollection(name)\
                    .filterBounds(self.point)\
                    .filterDate(str(self.start_date), str(self.end_date))

        return img_col

    def retain_highest_qual(self, img):
        qa = img.select('qa_mask')
        # the way this works: the first two bits indicate the overall quality of the image.
        # 00 indicates the highest quality, 01, 10, 11 some minor qualities.
        # the qa mask contains 32 bit integers, but we are only interested in the first two
        # bits. The bitwise AND operation between two binary numbers, ouputs a 1 if both
        # bits are 1 and 0 otherwise: e.g 100101 and 000011 = 000001. Thus,
        # perform a bitwise AND operation between the qamask and the binary representation of 3
        # this retains only the information in the first 2 bits and sets all other bits to 0
        best_qual = qa.bitwiseAnd(3)
        # in decimals best_qual = 0 indicates the highest quality (00 = 0, 01 = 1, 10, = 2, 11 = 3)
        # thus ensure it is equal to 0
        best_qual_mask = best_qual.eq(0)
        return img.updateMask(best_qual_mask)

    #***************************************************************************
    # Defining the classes

    class SR_500:
        def __init__(self, parent):
            self.img_col = parent.init_coll('MODIS/061/MOD09GA')\
                                 .select(['sur_refl_b01','sur_refl_b02','sur_refl_b03',
                                          'sur_refl_b04','sur_refl_b05','sur_refl_b06',
                                          'sur_refl_b07','QC_500m'])\
                                 .map(self.rename).map(self.rescale).map(parent.retain_highest_qual)

        def rename(self, img):
            newnames = ['RED','NIR1','BLUE','GREEN','NIR2','SWIR1','SWIR2','qa_mask']
            return img.rename(newnames)

        def rescale(self, img):
            opt = img.select(['RED','NIR1','BLUE','GREEN','NIR2','SWIR1','SWIR2'])
            qual = img.select('qa_mask')

            # rescale the opt bands
            opt = opt.multiply(0.0001)
            scaled = ee.Image.cat([opt, qual]).copyProperties(img)

            # system properties are not copied
            scaled = scaled.set('system:time_start', img.get('system:time_start'))
            return scaled

        def compute_NDWI_Gao(self):
            '''
            The NDWI follwing Gao 1996, is defined as (NIR - SWIR)/(NIR + SWIR)
            I.e. the normalised difference between the NIR and SWIR bands
            It it thought to capture the water content in vegetation
            '''
            ndwi_gao = self.img_col.map(lambda img: img.normalizedDifference(['NIR1', 'SWIR1']).rename("NDWI_Gao"))
            return ndwi_gao

        def compute_NDWI_McFeeters(self):
            '''
            The NDWI following McFeeters, id defined as (GREEN - NIR)/(GREEN + NIR)
            It captures changes in water bassins.
            '''
            ndwi_McF = self.img_col.map(lambda img: img.normalizedDifference(['GREEN', 'NIR1']).rename("NDWI_McF"))
            return ndwi_McF

        def compute_NDVI(self):
            ndvi = self.img_col.map(lambda img: img.normalizedDifference(['NIR1', 'RED']).rename("NDVI"))
            return ndvi

    class SR_250:
        def __init__(self, parent):
            self.img_col = parent.init_coll('MODIS/061/MOD09GQ')\
                                 .select(['sur_refl_b01','sur_refl_b02','QC_250m'])\
                                 .map(self.rename).map(self.rescale).map(parent.retain_highest_qual)

        def rename(self, img):
            new_names = ['RED', 'NIR', 'qa_mask']
            return img.rename(new_names)

        def rescale(self, img):
            opt = img.select(['RED','NIR'])
            qual = img.select('qa_mask')

            # rescale the opt bands
            opt = opt.multiply(0.0001)
            scaled = ee.Image.cat([opt, qual]).copyProperties(img)

            # system properties are not copied
            scaled = scaled.set('system:time_start', img.get('system:time_start'))
            return scaled

        def compute_NDVI(self):
            ndvi = self.img_col.map(lambda img: img.normalizedDifference(['NIR', 'RED']).rename("NDVI"))
            return ndvi

# define the RS Feature class
class RS_Feats:
    def __init__(self, point: ee.Geometry, start_ts: str):
        self.start_date = datetime.strptime(start_ts, "%Y-%m-%d").date()
        self.end_date = self.start_date + timedelta(days = 365)
        self.point = point

    #***************************************************************************
    # functions to initiatlise the different inner classes --> avoids loading all at once (slow)
    def init_wsf(self):
        return self.WSF(self)

    def init_nl_harm(self):
        return self.NL_Harm(self)

    def init_ecmwf(self):
        return self.ECMWF(self)

    def init_modis_lc(self):
        return(self.Modis_LC(self))

    def init_esa_lc(self):
        return self.ESA_LC(self)

    def init_precip(self):
        return self.Precip(self)

    #***************************************************************************
    # Parent functions to load the data

    def init_coll(self, name: str):
        '''
        initiates the image collection and restricts it to the correct time
        '''
        img_col = ee.ImageCollection(name)\
                    .filterBounds(self.point)\
                    .filterDate(str(self.start_date), str(self.end_date))

        return img_col

    def init_yearly_coll(self, name: str):
        img_col = ee.ImageCollection(name)\
                    .filterBounds(self.point)\
                    .filterDate(str(self.start_date.year), str(self.end_date.year))
        return img_col

    def get_scale(self, img_col):
        scale = img_col.first().projection().nominalScale().getInfo()
        return scale

    def make_roi(self):
        return self.point.buffer(128 * 30).bounds()

    #***************************************************************************
    # Defining the classes

    class WSF:
        def __init__(self, parent):
            self.img = self.load_wsf_img(parent).rename('WSF')

        def load_wsf_img(self, parent):
            img = ee.ImageCollection('projects/sat-io/open-datasets/WSF/WSF_2019')\
                          .filterBounds(parent.point)\
                          .select('b1')\
                          .mosaic()
            #img = img.where(img.eq(255), 1).unmask(0)
            return img

    class NL_Harm:
        def __init__(self, parent):
            self.img = parent.init_yearly_coll('projects/sat-io/open-datasets/npp-viirs-ntl')\
                              .first()\
                              .select('b1')\
                              .rename('NIGHTLIGHTS')
            self.img = self.img.unmask(0) # starting in 2019, water bodies are masked in the output - set them to 0 as in previous years.

    class ECMWF:
        def __init__(self, parent):
            self.point = parent.point
            bands = ['temperature_2m', 'total_precipitation_sum']
            self.img_col = parent.init_coll("ECMWF/ERA5_LAND/DAILY_AGGR").select(bands).map(self.rename)
            self.scale = parent.get_scale(self.img_col)
            #self.weather_vars = self.aggregate_img_col().getInfo()['features'][0]['properties']

        def rename(self, img):
            img = img.rename(['temp', 'precip'])
            return img

        def aggregate_img_col(self):
            temp_mean = self.img_col.select('temp').mean().rename('mean_temp')
            temp_min = self.img_col.select('temp').min().rename("min_temp")
            temp_max = self.img_col.select('temp').max().rename("max_temp")
            precip_sum = self.img_col.select('precip').sum().rename('sum_precip')
            agg_img = temp_mean.addBands(temp_min).addBands(temp_max).addBands(precip_sum)
            weather_vars = agg_img.sample(self.point, scale = self.scale)
            return weather_vars

    class Precip:
        def __init__(self, parent):
            self.point = parent.point
            self.img = parent.init_coll("UCSB-CHG/CHIRPS/PENTAD").sum()
            self.scale = self.img.projection().nominalScale().getInfo()
            self.data = self.sample_data()

        def sample_data(self):
            precip = self.img.sample(self.point, self.scale)
            return precip.getInfo()['features'][0]['properties']['precipitation']


    class Modis_LC:
        def __init__(self, parent):
            self.img = parent.init_yearly_coll("MODIS/061/MCD12Q1")\
                                 .first()\
                                 .select('LC_Type1')\
                                 .rename('LandCoverModis')

    class ESA_LC:
        def __init__(self, parent):
            self.img = ee.ImageCollection('ESA/WorldCover/v100')\
                             .filterBounds(parent.point).first().rename('LandCoverESA')

            self.cropland = self.mask_cropland()
            self.water = self.mask_water()

        def mask_cropland(self):
            cropland_mask = self.img.eq([20,30,40,90]).reduce(ee.Reducer.max()) # mask shrublands, grassland, cropland, herbaceous wetland (rice cultivation for instance)
            return cropland_mask.rename('cropland')

        def mask_water(self):
            water_mask = self.img.eq([80]).reduce(ee.Reducer.max())
            return water_mask.rename('water')

# define the World Pop class
class WorldPop:
    def __init__(self, point: ee.Geometry, start_ts: str, country: str):
        self.start_date = datetime.strptime(start_ts, "%Y-%m-%d").date()
        self.end_date = self.start_date + timedelta(days = 365)
        self.point = point
        self.country = country.upper()

        self.img = self.init_yearly_coll('WorldPop/GP/100m/pop').first()

    def init_yearly_coll(self, name: str):
        img_col = ee.ImageCollection(name)\
                    .filterBounds(self.point)\
                    .filterDate(str(self.start_date.year), str(self.end_date.year))\
                    .filterMetadata('country','equals',self.country)
        return img_col



