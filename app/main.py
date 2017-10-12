# coding: utf-8

import logging
import os
import sys
import warnings


from bokeh import events
from bokeh.colors import RGB
from bokeh.layouts import gridplot, column
from bokeh.models import (
    Range1d, LinearColorMapper, ColorBar, FixedTicker,
    ColumnDataSource, WMTSTileSource)
from bokeh.models.widgets import Div
from bokeh.plotting import figure, curdoc
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable, get_cmap, register_cmap
import numpy as np
from tornado import gen


import config
from models.binned_color_mapper import BinnedColorMapper
from util import wrap_on_change, time_setter
from data import FileSelection

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
        self.tools = 'pan, box_zoom, reset, save'
        self.width = 768
        self.height = int(self.width / 1.6)
        self._first = True
        self.fileselector = FileSelection(self.variable)

    def setup_coloring(self):
        levels = MaxNLocator(
            nbins=self.variable_options.NBINS).tick_values(
                self.variable_options.MIN_VAL, self.variable_options.MAX_VAL)
        self.levels = levels
        cmap = get_cmap(self.variable_options.CMAP)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        self.color_pal = [RGB(*val).to_hex() for val in
                          sm.to_rgba(levels, bytes=True, norm=True)[:-1]]
        self.continuous_mapper = LinearColorMapper(
            self.color_pal, low=sm.get_clim()[0], high=sm.get_clim()[1])
        bin_pal = self.color_pal.copy()
        bin_pal.append('#ffffff')
        self.bin_mapper = BinnedColorMapper(bin_pal, alpha=config.ALPHA)

    def make_sources(self):
        self.sources['hover_pt'] = ColumnDataSource(data={
            'x': [0], 'y': [0], 'x_idx': [0],
            'y_idx': [0], 'time': [0], 'val': [0]})
        self.sources['click_pt'] = ColumnDataSource(data={
            'clickx': [0], 'clicky': [0]})
        self.sources['summary_stats'] = ColumnDataSource(data={
            'current_val': [0], 'mean': [0], 'median': [0],
            'bin_width': [0], 'valid_ms': [0]})
        self.sources['rgba_img'] = ColumnDataSource(
            data={'image': [], 'x': [], 'y': [],
                  'dw': [], 'dh': []})
        self.sources['histogram'] = ColumnDataSource(data={
            'x': np.array(self.levels[:-1]),
            'top': np.array([3.0e6] * len(self.levels[:-1])),
            'color': np.array(self.color_pal)})
        self.sources['timeseries'] = ColumnDataSource(data={
            'x': [0], 'y': [self.variable_options.MAX_VAL]})
        self.sources['raw_data'] = self.fileselector.source

    def register_models_and_sources(self):
        self.make_sources()
        self.make_map_figure()
        self.make_histogram_figure()
        self.models['info_div'] = Div(width=self.width)
        self.make_timeseries_figure()

    def make_map_figure(self):
        map_fig = figure(plot_width=self.width, plot_height=self.height,
                         y_axis_type=None, x_axis_type=None,
                         toolbar_location='left',
                         tools=self.tools + ', wheel_zoom',
                         active_scroll='wheel_zoom',
                         title='', name='mapfig',
                         responsive=True)

        map_fig.image(image='image', x='x', y='y', dw='dw', dh='dh',
                      source=self.sources['rgba_img'],
                      color_mapper=self.bin_mapper)
        ticker = FixedTicker(ticks=self.levels[::3])
        cb = ColorBar(color_mapper=self.continuous_mapper, location=(0, 0),
                      scale_alpha=config.ALPHA, ticker=ticker)
        # Need to use this and not bokeh.tile_providers.STAMEN_TONER
        # https://github.com/bokeh/bokeh/issues/4770
        STAMEN_TONER = WMTSTileSource(
            url='https://stamen-tiles.a.ssl.fastly.net/toner-lite/{Z}/{X}/{Y}.png',  # NOQA
            attribution=(
                'Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
                'under <a href="http://creativecommons.org/licenses/by/3.0">CC'
                ' BY 3.0</a>. Map data by <a href="http://openstreetmap.org">'
                'OpenStreetMap</a>, under '
                '<a href="http://www.openstreetmap.org/copyright">ODbL</a>'
            )
        )
        map_fig.add_tile(STAMEN_TONER)
        map_fig.add_layout(cb, 'right')

        map_fig.x(x='x', y='y', size=10, color=config.RED, alpha=config.ALPHA,
                  source=self.sources['hover_pt'], level='overlay')

        self.models['map_fig'] = map_fig

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

    def make_histogram_figure(self):
        hheight = int(self.width / 2)
        # Make the histogram figure
        hist_fig = figure(plot_width=hheight, plot_height=hheight,
                          toolbar_location='right',
                          x_axis_label=self.variable_options.XLABEL,
                          y_axis_label='Counts',
                          tools=self.tools + ', ywheel_zoom',
                          active_scroll='ywheel_zoom',
                          x_range=Range1d(start=self.variable_options.MIN_VAL,
                                          end=self.variable_options.MAX_VAL),
                          title='Histogram of map pixels')

        # make histograms
        bin_width = self.levels[1] - self.levels[0]
        self.sources['summary_stats'].data.update({'bin_width': [bin_width]})
        bin_centers = self.levels[:-1] + bin_width / 2
        self.sources['histogram'].data.update({'x': bin_centers})
        hist_fig.vbar(x='x', top='top', width=bin_width, bottom=0,
                      fill_color='color', line_color='color',
                      fill_alpha=config.ALPHA,
                      source=self.sources['histogram'])
        self.models['hist_fig'] = hist_fig

    @gen.coroutine
    def update_histogram(self):
        map_fig = self.models['map_fig']
        left = map_fig.x_range.start
        right = map_fig.x_range.end
        bottom = map_fig.y_range.start
        top = map_fig.y_range.end

        raw_data = self.sources['raw_data'].data
        masked_regrid = raw_data['masked_regrid'][0]
        xn = raw_data['xn'][0]
        yn = raw_data['yn'][0]

        left_idx = np.abs(xn - left).argmin()
        right_idx = np.abs(xn - right).argmin() + 1
        bottom_idx = np.abs(yn - bottom).argmin()
        top_idx = np.abs(yn - top).argmin() + 1
        logging.debug('Updating histogram...')
        try:
            new_subset = masked_regrid[bottom_idx:top_idx, left_idx:right_idx]
        except TypeError:
            return
        counts, _ = np.histogram(
            new_subset.clip(max=self.variable_options.MAX_VAL),
            bins=self.levels,
            range=(self.levels.min(), self.levels.max()))

        self.sources['histogram'].data.update({'top': counts})
        logging.debug('Done updating histogram')

        self.sources['summary_stats'].data.update(
            {'mean': [float(new_subset.mean())]})

    @gen.coroutine
    def update_for_map_click(self):
        x = self.sources['click_pt'].data['clickx'][0]
        y = self.sources['click_pt'].data['clicky'][0]

        raw_data = self.sources['raw_data'].data
        xn = raw_data['xn'][0]
        yn = raw_data['yn'][0]
        x_idx = np.abs(xn - x).argmin()
        y_idx = np.abs(yn - y).argmin()

        self.sources['hover_pt'].data.update({
            'x_idx': [x_idx],
            'y_idx': [y_idx],
            'x': [xn[x_idx]],
            'y': [yn[y_idx]],
            })

    def make_timeseries_figure(self):
        hheight = int(self.width / 2)
        tseries_fig = figure(
            height=hheight, width=hheight,
            x_axis_type='datetime',
            tools=self.tools + ', wheel_zoom',
            active_scroll='wheel_zoom',
            toolbar_location='right',
            title='Time-series at selected location',
            y_axis_label=self.variable_options.XLABEL)

        tseries_fig.line(x='x', y='y', color=config.BLUE,
                         source=self.sources['timeseries'])

        tseries_fig.diamond(x='valid_ms', y='current_val', color=config.RED,
                            source=self.sources['summary_stats'],
                            level='overlay', size=6)
        self.models['timeseries_fig'] = tseries_fig

    @gen.coroutine
    def update_timeseries(self):
        x_idx = self.sources['hover_pt'].data['x_idx'][0]
        y_idx = self.sources['hover_pt'].data['y_idx'][0]

        line_data = self.fileselector.get_timeseries(x_idx, y_idx)
        if len(line_data.dropna().values) < 1:
            self.sources['timeseries'].data.update({
                'y': [self.variable_options.MAX_VAL], 'x': [0]})
        else:
            self.sources['timeseries'].data.update({
                'y': line_data.values,
                'x': time_setter(line_data.index)})

        masked_regrid = self.sources['raw_data'].data['masked_regrid'][0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            val = float(masked_regrid[y_idx, x_idx])
        self.sources['summary_stats'].data.update({
            'current_val': [val],
            'valid_ms': [
                time_setter(self.sources['raw_data'].data['valid_date'][0])]})

    @gen.coroutine
    def update_info_text(self):
        info_text = """
<div class="well">
<b>Selected Value:</b> {current_val:0.1f} <b>Area Mean:</b> {mean:0.1f} <b>Bin Width</b> {bin_width:0.1f}
</div>
"""  # NOQA
        summary_stats = self.sources['summary_stats'].data
        self.models['info_div'].text = info_text.format(
            current_val=float(summary_stats['current_val'][0]),
            mean=summary_stats['mean'][0],
            bin_width=summary_stats['bin_width'][0]
        )

    def make_layout(self):
        lay = column(
            self.fileselector.make_layout(),
            gridplot([self.models['map_fig']],
                     [self.models['timeseries_fig'], self.models['hist_fig']],
                     toolbar_location='left'),
            self.models['info_div']
            )
        return lay

    def add_callbacks(self):
        self.sources['raw_data'].on_change(
            'data', wrap_on_change(self.update_map,
                                   self.update_histogram,
                                   self.update_for_map_click))

        map_fig = self.models['map_fig']
        for what, on in [(map_fig.x_range, 'start'),
                         (map_fig.x_range, 'end'),
                         (map_fig.y_range, 'start'),
                         (map_fig.y_range, 'end')]:
            what.on_change(on, wrap_on_change(
                self.update_histogram, timeout=100))

        self.sources['hover_pt'].on_change(
            'data', wrap_on_change(self.update_timeseries))
        self.sources['summary_stats'].on_change(
            'data', wrap_on_change(self.update_info_text, timeout=100))

        def move_click_marker(event):
            self.sources['click_pt'].data.update({'clickx': [event.x],
                                                  'clicky': [event.y]})
            self.update_for_map_click()

        map_fig.on_event(events.Tap, move_click_marker)
        map_fig.on_event(events.Press, move_click_marker)


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
