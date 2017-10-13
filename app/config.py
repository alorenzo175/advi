# coding: utf-8


from collections import namedtuple
import os


ALPHA = 0.7
RED = '#AB0520'
BLUE = '#0C234B'
TITLE = 'UA HAS ADVI'
DATA_DIRECTORY = '~/.wrf'
WS_ORIGIN = os.getenv('WS_ORIGIN', 'localhost:5006')
GA_TRACKING_ID = ''
PREFIX = ''
SIZING_MODE = 'scale_width'
POSSIBLE_MODELS = ('WRFGFS_00Z', 'WRFGFS_06Z', 'WRFGFS_12Z',
                   'WRFNAM_00Z', 'WRFNAM_06Z', 'WRFNAM_12Z',
                   'WRFRUC_12Z')
CUSTOM_BOKEH_MODELS = (('app.models.binned_color_mapper', 'BinnedColorMapper'),
                       ('app.models.disabled_select', 'DisabledSelect'))

MENU_VARS = (('2m Temperature', 'temp'),
             ('1 hr Temperature Change', 'dt'),
             ('10m Wind Speed', 'wspd'),
             ('1 hr Precip', 'rain'),
             ('Accumulated Precip', 'rainac'),
             ('GHI', 'ghi'),
             ('DNI', 'dni'),
             ('Composite Radar', 'radar'))


variable_options = namedtuple(
    'variable_options',
    ['MIN_VAL', 'MAX_VAL', 'VAR', 'CMAP', 'XLABEL', 'NBINS'])

VAR_OPTS = {
    'radar': variable_options(
        MIN_VAL=0,
        MAX_VAL=80,
        VAR='MDBZ',
        CMAP='nws_radar',
        XLABEL='Composite Reflectivity (dBZ)',
        NBINS=17),
    'rain': variable_options(
        MIN_VAL=0,
        MAX_VAL=2,
        VAR='RAIN1H',
        CMAP='magma',
        XLABEL='One-hour Precip (in)',
        NBINS=21),
    'rainac': variable_options(
        MIN_VAL=0,
        MAX_VAL=2,
        VAR='RAINNC',
        CMAP='magma',
        XLABEL='Precip Accumulation (in)',
        NBINS=21),
    'dt': variable_options(
        MIN_VAL=-20,
        MAX_VAL=20,
        VAR='DT',
        CMAP='coolwarm',
        XLABEL='One-Hour Temperature Change (°F)',
        NBINS=40),
    'temp': variable_options(
        MIN_VAL=0,
        MAX_VAL=120,
        VAR='T2',
        CMAP='plasma',
        XLABEL='2m Temperature (°F)',
        NBINS=61),
    'wspd': variable_options(
        MIN_VAL=0,
        MAX_VAL=44,
        VAR='WSPD',
        CMAP='viridis',
        XLABEL='10m Wind Speed (knots)',
        NBINS=25),
    'ghi': variable_options(
        MIN_VAL=0,
        MAX_VAL=1200,
        VAR='SWDNB',
        CMAP='viridis',
        XLABEL='GHI (W/m^2)',
        NBINS=25),
    'dni': variable_options(
        MIN_VAL=0,
        MAX_VAL=1100,
        VAR='SWDDNI',
        CMAP='viridis',
        XLABEL='DNI (W/m^2)',
        NBINS=25)
    }
