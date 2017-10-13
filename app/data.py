import datetime as dt
import logging
from pathlib import Path
import os


import numpy as np
import pandas as pd
import tables
from bokeh.layouts import row
from bokeh.models import Slider, ColumnDataSource
from bokeh.models.widgets import Select
from tornado import gen


import config
from models.disabled_select import DisabledSelect
from util import wrap_on_change


class H5File(object):
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.t = tables.open_file(self.path, mode='r')
        return self.t

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.t.close()
        except:
            pass


DATA_DIRECTORY = os.getenv('ADVI_DATADIR',
                           config.DATA_DIRECTORY)


def load_file(variable, model, fx_date='latest'):
    dir = os.path.expanduser(DATA_DIRECTORY)
    if fx_date == 'latest':
        p = Path(dir)
        model_dir = sorted([pp for pp in p.rglob(f'*{model}')],
                           reverse=True)[0]
    else:
        model_dir = os.path.join(
            dir, dt.datetime.strptime(fx_date,
                                      '%Y-%m-%d').strftime('%Y/%m/%d'),
            strpmodel(model))

    path = os.path.join(model_dir,
                        f'{variable}.h5')

    global h5file
    h5file = H5File(path)
    global times
    with h5file as h5:
        times = pd.DatetimeIndex(
            h5.get_node('/times')[:]).tz_localize('UTC')
    return times


def load_data(valid_time):
    strformat = '%Y%m%dT%H%MZ'
    with h5file as h5:
        regridded_data = h5.get_node(f'/{valid_time.strftime(strformat)}')[:]
        regridded_data[np.isnan(regridded_data)] = -999

        X = h5.get_node('/X')[:]
        Y = h5.get_node('/Y')[:]
    masked_regrid = np.ma.masked_less(regridded_data, -998)
    return masked_regrid, X, Y


def load_tseries(xi, yi):
    strformat = '%Y%m%dT%H%MZ'
    rd = []
    with h5file as h5:
        for t in times:
            rd.append(h5.get_node(f'/{t.strftime(strformat)}')[yi, xi])
    data = pd.Series(rd, index=times)
    return data


def find_fx_times(variable):
    p = Path(DATA_DIRECTORY).expanduser()
    out = set()
    for pp in sorted(p.rglob(f'*WRF*')):
        try:
            datetime = dt.datetime.strptime(''.join(pp.parts[-4:-1]),
                                            '%Y%m%d')
        except ValueError:
            logging.debug('%s does not conform to expected format', pp)
            continue
        if not pp.joinpath(f'{variable}.h5').exists():
            logging.debug('No h5 file for %s in %s', variable, pp)
            continue
        out.add(datetime.strftime('%Y-%m-%d'))
    return sorted(out, reverse=True)


def strfmodel(modelstr):
    return f'{modelstr[3:6]} {modelstr[7:]}'


def strpmodel(model):
    m = model.split(' ')
    return f'WRF{m[0]}_{m[1]}'


def get_wrf_models(variable, date_string):
    dir = os.path.join(DATA_DIRECTORY,
                       dt.datetime.strptime(
                           date_string, '%Y-%m-%d').strftime('%Y/%m/%d'))
    p = Path(dir).expanduser()
    disabled = {model: True for model in config.POSSIBLE_MODELS}
    for pp in p.iterdir():
        if pp.joinpath(f'{variable}.h5').exists():
            m = pp.parts[-1]
            disabled[m] = False
    mld = [(strfmodel(k), v) for k, v in disabled.items()]
    return mld


class FileSelection(object):
    def __init__(self, variable):
        self.variable = variable
        self.variable_options = config.VAR_OPTS[variable]
        self.file_dates = find_fx_times(self.variable_options.VAR)
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
                                                 self.update_file,
                                                 self.update_data))
        self.select_model.on_change('value',
                                    wrap_on_change(self.update_file,
                                                   self.update_data))
        self.select_time.on_change('value',
                                   wrap_on_change(self.update_data,
                                                  timeout=100))
        self.update_wrf_models()

    def make_layout(self):
        lay = row(self.select_day, self.select_model, self.select_time,
                  sizing_mode=config.SIZING_MODE,
                  width=768, height=100)
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
        wrf_models = get_wrf_models(self.variable_options.VAR,
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
        self.times = load_file(self.variable_options.VAR,
                               self.wrf_model, self.initialization_day)
        options = [t.strftime('%Y-%m-%d %H:%MZ') for t in self.times]
        self.select_time.end = len(options) - 1
        if self.select_time.value > self.select_time.end:
            self.select_time.value = self.select_time.end

    @gen.coroutine
    def update_data(self):
        masked_regrid, X, Y = load_data(self.valid_datetime)
        xn = X[0]
        yn = Y[:, 0]
        self.source.data.update({'masked_regrid': [masked_regrid],
                                 'xn': [xn], 'yn': [yn],
                                 'valid_date': [self.valid_datetime]})

    def get_timeseries(self, xi, yi):
        return load_tseries(xi, yi)
