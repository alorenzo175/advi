# coding: utf-8

from collections import OrderedDict
import datetime as dt
from functools import partial
import importlib
import logging
from pathlib import Path
import os
import sys
import warnings


from bokeh import events
from bokeh.colors import RGB
from bokeh.layouts import gridplot, column, row
from bokeh.models import (
    Range1d, LinearColorMapper, ColorBar, FixedTicker,
    ColumnDataSource, WMTSTileSource, Slider)
from bokeh.models.widgets import Select, Div
from bokeh.plotting import figure, curdoc
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable, get_cmap, register_cmap
import numpy as np
import pandas as pd
import tables
from tornado import gen


from models.disabled_select import DisabledSelect
from models.binned_color_mapper import BinnedColorMapper
import config
import data

nws_radar_cmap = ListedColormap(
    name='nws_radar', colors=(
        "#646464",
        "#04e9e7",
        "#019ff4",
        "#0300f4",
        "#02fd02",
        "#01c501",
        "#008e00",
        "#fdf802",
        "#e5bc00",
        "#fd9500",
        "#fd0000",
        "#d40000",
        "#bc0000",
        "#f800fd",
        "#9854c6",
        "#fdfdfd",
        ))
register_cmap(cmap=nws_radar_cmap)


class ADVIApp(object):
    models = {}
    sources = {}

    def __init__(self, variable):
        self.variable = variable
        self.variable_options = config.VAR_OPTS[variable]
        self.width = 768
        self.height = int(self.width / 1.6)
        self._first = True

    def setup_coloring(self):
        levels = MaxNLocator(
            nbins=self.variable_options.NBINS).tick_values(
                self.variable_options.MIN_VAL, self.variable_options.MAX_VAL)
        self.levels = levels
        cmap = get_cmap(self.variable_options.CMAP)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        color_pal = [RGB(*val).to_hex() for val in
                     sm.to_rgba(levels, bytes=True, norm=True)[:-1]]
        self.continuous_mapper = LinearColorMapper(
            color_pal, low=sm.get_clim()[0], high=sm.get_clim()[1])
        bin_pal = color_pal.copy()
        bin_pal.append('#ffffff')
        self.bin_mapper = BinnedColorMapper(bin_pal, alpha=config.ALPHA)

    def register_models_and_sources(self):
        map_fig, map_sources = make_map_figure(
            self.width, self.height, self.bin_mapper, self.continuous_mapper,
            self.levels)

        self.sources.update(map_sources)
        self.models.update({'map_fig': map_fig})

        self.fileselector = FileSelection(self.variable)
        self.sources['raw_data'] = self.fileselector.source

    def make_layout(self):
        lay = column(
            self.fileselector.make_layout(),
            gridplot([self.models['map_fig']],
                     [],
                     toolbar_location='left'))
        return lay

    def add_callbacks(self):
        self.sources['raw_data'].on_change(
            'data', wrap_on_change(self.update_map))

    @gen.coroutine
    def update_map(self):
        raw_data = self.sources['raw_data'].data
        logging.debug('Updating map...')
        valid_date = raw_data['valid_date'][0]
        mfmt = '%Y-%m-%d %H:%M MST'
        title = (f'UA WRF {self.variable_options.XLABEL} valid at '
                 f'{valid_date.tz_convert("MST").strftime(mfmt)}')
        map_fig = self.models['map_fig']
        map_fig.title.text = title
        masked_regrid = raw_data['masked_regrid'][0]
        xn = raw_data['xn'][0]
        yn = raw_data['yn'][0]
        dx = xn[1] - xn[0]
        dy = yn[1] - yn[0]
        vals = (np.digitize(masked_regrid.filled(np.inf),
                            self.levels).astype('uint8') - 1)
        self.sources['rgba_img'].data.update({'image': [vals],
                                              'x': [xn[0] - dx / 2],
                                              'y': [yn[0] - dy / 2],
                                              'dw': [xn[-1] - xn[0] + dx],
                                              'dh': [yn[-1] - yn[0] + dy]})
        if self._first:
            map_fig.x_range.start = xn[0]
            map_fig.x_range.end = xn[-1]
            map_fig.y_range.start = yn[0]
            map_fig.y_range.end = yn[-1]
            self._first = False
        logging.debug('Done updating map')



def make_map_figure(width, height, bin_mapper, continuous_mapper, levels):
    map_fig = figure(plot_width=width, plot_height=height,
                     y_axis_type=None, x_axis_type=None,
                     toolbar_location='left',
                     tools='pan, box_zoom, reset, save, wheel_zoom',
                     active_scroll='wheel_zoom',
                     title='', name='mapfig',
                     responsive=True)
    rgba_img_source = ColumnDataSource(data={'image': [], 'x': [], 'y': [],
                                             'dw': [], 'dh': []})
    map_fig.image(image='image', x='x', y='y', dw='dw', dh='dh',
                  source=rgba_img_source, color_mapper=bin_mapper)

    ticker = FixedTicker(ticks=levels[::3])
    cb = ColorBar(color_mapper=continuous_mapper, location=(0, 0),
                  scale_alpha=config.ALPHA, ticker=ticker)
    # Need to use this and not bokeh.tile_providers.STAMEN_TONER
    # https://github.com/bokeh/bokeh/issues/4770
    STAMEN_TONER = WMTSTileSource(
        url='https://stamen-tiles.a.ssl.fastly.net/toner-lite/{Z}/{X}/{Y}.png',
        attribution=(
            'Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
            'under <a href="http://creativecommons.org/licenses/by/3.0">CC BY '
            '3.0</a>. Map data by <a href="http://openstreetmap.org">'
            'OpenStreetMap</a>, under '
            '<a href="http://www.openstreetmap.org/copyright">ODbL</a>'
        )
    )
    map_fig.add_tile(STAMEN_TONER)
    map_fig.add_layout(cb, 'right')

    hover_pt = ColumnDataSource(data={'x': [0], 'y': [0], 'x_idx': [0],
                                      'y_idx': [0]})
    map_fig.x(x='x', y='y', size=10, color=config.RED, alpha=config.ALPHA,
              source=hover_pt, level='overlay')

    return map_fig, {'rgba_img': rgba_img_source, 'hover_pt': hover_pt}


def wrap_on_change(*funcs, timeout=None):
    def wrapper(attr, old, new):
        for f in funcs:
            try:
                if timeout is not None:
                    curdoc().add_timeout_callback(f, timeout)
                else:
                    curdoc().add_next_tick_callback(f)
            except ValueError:
                pass
    return wrapper


class FileSelection(object):
    def __init__(self, variable):
        self.variable = variable
        self.variable_options = config.VAR_OPTS[variable]
        self.file_dates = data.find_fx_times(self.variable_options.VAR)
        self.source = ColumnDataSource(data={
            'masked_regrid': [0], 'xn': [0], 'yn': [0],
            'valid_date': [dt.datetime.now()]})

        self.select_day = Select(title='Initialization Day',
                                 value=self.file_dates[0],
                                 options=self.file_dates)
        self.select_model = DisabledSelect(title='Initialization', value='',
                                           options=[])
        self.select_time = Slider(title='Forecast Time-Step', start=0, end=1,
                                  value=0, name='timeslider')

        self.select_day.on_change('value',
                                  wrap_on_change(self.update_wrf_models,
                                                 self.update_data))
        self.select_model.on_change('value',
                                    wrap_on_change(self.update_file,
                                                   self.update_data))
        self.select_time.on_change('value',
                                   wrap_on_change(self.update_data,
                                                  timeout=100))
        self.update_wrf_models()

    def make_layout(self):
        lay = row([self.select_day, self.select_model, self.select_time])
        return lay

    @property
    def time_step(self):
        return int(self.select_time.value)

    @property
    def valid_datetime(self):
        return self.times[self.time_step]

    @property
    def wrf_model(self):
        return self.select_model.value

    @property
    def initialization_day(self):
        return self.select_day.value

    @gen.coroutine
    def update_wrf_models(self):
        wrf_models = data.get_wrf_models(self.variable_options.VAR,
                                         self.initialization_day)
        self.select_model.options = wrf_models
        thelabel = ''
        for m, disabled in wrf_models:
            if m == self.wrf_model and not disabled:
                thelabel = m
            if not disabled and not thelabel:
                thelabel = m
        self.select_model.value = thelabel

    @gen.coroutine
    def update_file(self):
        self.times = data.load_file(self.variable_options.VAR,
                                    self.wrf_model, self.initialization_day)
        options = [t.strftime('%Y-%m-%d %H:%MZ') for t in self.times]
        self.select_time.end = len(options) - 1
        if self.select_time.value > self.select_time.end:
            self.select_time.value = self.select_time.end

    @gen.coroutine
    def update_data(self):
        logging.info('Getting data...')
        masked_regrid, X, Y = data.load_data(self.valid_datetime)
        xn = X[0]
        yn = Y[:, 0]
        self.source.data.update({'masked_regrid': [masked_regrid],
                                 'xn': [xn], 'yn': [yn],
                                 'valid_date': [self.valid_datetime]})


app = ADVIApp(sys.argv[1])
app.setup_coloring()
app.register_models_and_sources()
lay = app.make_layout()
app.add_callbacks()
doc = curdoc()
doc.add_root(lay)


doc.title = config.TITLE
doc.template_variables.update({
    'menu_vars': config.MENU_VARS,
    'prefix': os.getenv('ADVI_PREFIX', config.PREFIX),
    'ga_tracking_id': os.getenv('ADVI_TRACKING_ID',
                                config.GA_TRACKING_ID)})
try:
    custom_model_code = sys.argv[2]
except IndexError:
    pass
else:
    doc.template_variables['custom_model_code'] = custom_model_code
