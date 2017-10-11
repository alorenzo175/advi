from collections import OrderedDict
import datetime as dt
import logging
from pathlib import Path
import os


import numpy as np
import pandas as pd
import tables


import config


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
