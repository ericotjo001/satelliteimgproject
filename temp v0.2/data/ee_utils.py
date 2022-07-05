import os
import ee
import pandas as pd
from tqdm import tqdm
import numpy as np

def get_start_end_date(year):
    start_date = '%s-1-1'%(year)
    end_date = '%s-12-31'%(year)
    return start_date, end_date

def decode_qamask(img: ee.Image) -> ee.Image:
    '''
    Args
    - img: ee.Image, Landsat 5/7/8 image containing 'pixel_qa' band
    Returns
    - masks: ee.Image, contains 5 bands of masks
    Pixel QA Bit Flags (universal across Landsat 5/7/8)
    Bit  Attribute
    0    Fill
    1    Clear
    2    Water
    3    Cloud Shadow
    4    Snow
    5    Cloud
    '''
    qa = img.select('QA_PIXEL')
    clear = qa.bitwiseAnd(2).neq(0)  # 0 = not clear, 1 = clear
    clear = clear.updateMask(clear).rename(['pxqa_clear'])

    water = qa.bitwiseAnd(4).neq(0)  # 0 = not water, 1 = water
    water = water.updateMask(water).rename(['pxqa_water'])

    cloud_shadow = qa.bitwiseAnd(8).eq(0)  # 0 = shadow, 1 = not shadow
    cloud_shadow = cloud_shadow.updateMask(cloud_shadow).rename(['pxqa_cloudshadow'])

    snow = qa.bitwiseAnd(16).eq(0)  # 0 = snow, 1 = not snow
    snow = snow.updateMask(snow).rename(['pxqa_snow'])

    cloud = qa.bitwiseAnd(32).eq(0)  # 0 = cloud, 1 = not cloud
    cloud = cloud.updateMask(cloud).rename(['pxqa_cloud'])

    masks = ee.Image.cat([clear, water, cloud_shadow, snow, cloud])
    return masks

def applyScaleFactor(image):
    image.select('SR_B.').multiply(0.0000275).add(-0.2);
    return image

def mask_qaclear(img: ee.Image) -> ee.Image:
    '''
    Args
    - img: ee.Image
    Returns
    - img: ee.Image, input image with cloud-shadow, snow, cloud, and unclear
        pixels masked out
    '''
    qam = decode_qamask(img)
    cloudshadow_mask = qam.select('pxqa_cloudshadow')
    snow_mask = qam.select('pxqa_snow')
    cloud_mask = qam.select('pxqa_cloud')
    return img.updateMask(cloudshadow_mask).updateMask(snow_mask).updateMask(cloud_mask)


import requests        

def save_an_image(datacode, bands, coords, fname, start_date, end_date, save_dir, dim=256,
    radius=3000, qa_clear=True, img_reduction='mean'):
    """
    args e.g.
        datacode get from Landsat website, e.g. 'LANDSAT/LC08/C02/T1_L2'
        bands e.g. ['SR_B4','SR_B3','SR_B2'], in Landsat 8 this is RGB
        coords: (longitudinal, latitude)
        fname: any filename you like. Better coded like country_year
        start_date, end_data = '2015-1-1', '2017-12-31'
        dim: size of downloaded image
        radius: in metres

        19637 is approx 0.3*65455, the recommended colour for visualization
          where 65455 is the colour max for SR_B.
    """
    f_name = fname +'.png'
    img_save_dir = os.path.join(save_dir,f_name)
    if os.path.exists(img_save_dir):
        return

    LON, LAT= coords
    roi = ee.Geometry.Point(LON,LAT).buffer(radius) # in meters. Singapore width is 26km
    imgcol = ee.ImageCollection(datacode)
    imgcol = imgcol.filterDate(start_date,end_date).filterBounds(roi)
    # print('imgcol.size().getInfo():',imgcol.size().getInfo())
    imgcol.map(applyScaleFactor)
    if qa_clear:
        imgcol = imgcol.map(mask_qaclear)
    
    imgcol = imgcol.select(bands)
    if img_reduction =='mean': # Landsat8: use this with mask_qaclear, good result! 
        img = imgcol.mean() 
    elif img_reduction=='median':
        img = imgcol.median()
    else:
        raise NotImplementedError()
        
    img.visualize(bands=bands,gamma=1)
    
    url = img.getThumbUrl({'min':0,'max':19637,'dimensions': dim, 'region': roi,}) # max 65455 is from the website's specification

    page = requests.get(url)
    
    with open(img_save_dir, 'wb') as f:
        f.write(page.content)


def download_dhs_img(DHS_DATA_DIR, data_source = 'landsat8_color', split='train', 
                     dim=256, radius=4000, qa_clear=True, img_reduction='mean', STOP_AT=None, randomizequeue=False):
    df = pd.read_csv(DHS_DATA_DIR['%s_csv'%(str(split))])
    
    if data_source == 'landsat8_color': 
        datacode = 'LANDSAT/LC08/C02/T1_L2'  
        bands = ['SR_B4','SR_B3','SR_B2']
        fname_appendix = ''
        save_dir = DHS_DATA_DIR['%s_img'%(str(split))]
    elif data_source == 'landsat8_infrared':
        datacode = 'LANDSAT/LC08/C02/T1_L2'  
        bands = ['SR_B5','SR_B6','SR_B7']
        fname_appendix = '_infra'
        save_dir = DHS_DATA_DIR['%s_img'%(str(split))] +'infra'
    else:
        raise NotImplementedError()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    toqueue = list(df.index[:])
    if randomizequeue:
        # just for observation during debugging. Disable it during actual download 
        # (otherwise the progressbar is messed up)
        np.random.shuffle(toqueue) 
    indices = tqdm(toqueue,0)
    n = len(df.index[:])
    counter=0
    for i in indices:
        thisrow = df.loc[i]
        coords = (thisrow['LONGNUM'],thisrow['LATNUM'])
        fname = thisrow['id'] + fname_appendix
        start_date, end_date = get_start_end_date(thisrow['year'])
        
        save_an_image(datacode, bands, coords, fname, start_date, end_date, save_dir, 
                      dim=dim, radius=radius, qa_clear=qa_clear, img_reduction=img_reduction)
        indices.set_description('processing:%s/%s'%(str(i),str(n)))
        counter+=1
        if STOP_AT is not None:
            if counter>=STOP_AT:
                break

# similar as DHS version, just minor change
def download_lsms_img(LSMS_DATA_DIR, data_source = 'landsat8_color', split='train', 
                     dim=256, radius=4000, qa_clear=True, img_reduction='mean', STOP_AT=None, randomizequeue=False):
    df = pd.read_csv(LSMS_DATA_DIR['%s_csv'%(str(split))])
    
    if data_source == 'landsat8_color': 
        datacode = 'LANDSAT/LC08/C02/T1_L2'  
        bands = ['SR_B4','SR_B3','SR_B2']
        fname_appendix = ''
        save_dir = LSMS_DATA_DIR['%s_img'%(str(split))]
    elif data_source == 'landsat8_infrared':
        datacode = 'LANDSAT/LC08/C02/T1_L2'  
        bands = ['SR_B5','SR_B6','SR_B7']
        fname_appendix = '_infra'
        save_dir = LSMS_DATA_DIR['%s_img'%(str(split))] +'infra'
    else:
        raise NotImplementedError()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    toqueue = list(df.index[:])
    indices = tqdm(toqueue,0)
    n = len(df.index[:])
    counter=0
    for i in indices:
        thisrow = df.loc[i]
        coords = (thisrow['lon'],thisrow['lat'])
        fname = thisrow['id'] + fname_appendix
        start_date, end_date = get_start_end_date(thisrow['year'])
        
        save_an_image(datacode, bands, coords, fname, start_date, end_date, save_dir, 
                      dim=dim, radius=radius, qa_clear=qa_clear, img_reduction=img_reduction)
        indices.set_description('processing:%s/%s'%(str(i),str(n)))
        counter+=1
        if STOP_AT is not None:
            if counter>=STOP_AT:
                break
