from bokeh.plotting import curdoc
import pandas as pd


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


def time_setter(index):
    if isinstance(index, pd.DatetimeIndex):
        return (index.tz_convert('MST').tz_localize(None).values.astype(int) /
                10**6)
    elif isinstance(index, pd.Timestamp):
        return index.tz_convert('MST').tz_localize(None).value / 10**6
    else:
        raise AttributeError
