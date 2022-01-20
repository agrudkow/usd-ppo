"""
This code is based on a code from spinningup (spinningup/spinup/utils/plot.py) and adjusted to our use case.
"""
import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def plot_data(
    data: pd.DataFrame,
    xaxis: str = "Epoch",
    value: str = "AverageEpRet",
    condition: str = "Condition1",
    smooth: int = 1,
    **kwargs,
):
  """Plot data timeseries-like data."""
  if smooth > 1:
    """
    Smooth data with moving window average.
    that is,
        smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
    where the "smooth" param is width of that window (2k+1)
    """
    y = np.ones(smooth)
    for datum in data:
      x = np.asarray(datum[value])
      z = np.ones(len(x))
      smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
      datum[value] = smoothed_x

  if isinstance(data, list):
    data = pd.concat(data, ignore_index=True)

  sns.set(
      context='notebook',
      style="darkgrid",
      font_scale=1.5,
  )
  sns.tsplot(
      data=data,
      time=xaxis,
      value=value,
      unit="Unit",
      condition=condition,
      ci='sd',
      **kwargs,
  )
  """
  If you upgrade to any version of Seaborn greater than 0.8.1, switch from tsplot to lineplot replacing L29 with:
    sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)
  Changes the colorscheme and the default legend style, though.
  """
  plt.legend(
      loc='upper left',
      ncol=1,
      handlelength=1,
      borderaxespad=0.,
      bbox_to_anchor=(1, 1),
      prop={'size': 18},
  )
  """
  For the version of the legend used in the Spinning Up benchmarking page, swap L38 with:
    plt.legend(loc='upper center', ncol=6, handlelength=1, mode="expand", borderaxespad=0., prop={'size': 33})
  """

  xscale = np.max(np.asarray(data[xaxis])) > 5e3
  if xscale:
    # Just some formatting niceness: x-axis scale in scientific notation if max x is large
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

  plt.tight_layout(pad=0.5)


def get_datasets(logdir, condition=None):
  """
    Recursively look through logdir for output files. Assumes that any file "progress.txt" is a valid hit.
  """
  global exp_idx
  global units
  datasets = []
  delimiter = ',' if 'rllib' in logdir else None
  for root, _, files in os.walk(logdir):
    if 'progress.txt' in files:
      exp_name = None
      try:
        config_path = open(os.path.join(root, 'config.json'))
        config = json.load(config_path)
        if 'exp_name' in config:
          exp_name = config['exp_name']
      except Exception:
        print('No file named config.json')
      condition1 = condition or exp_name or 'exp'
      condition2 = condition1 + '-' + str(exp_idx)
      exp_idx += 1
      if condition1 not in units:
        units[condition1] = 0
      unit = units[condition1]
      units[condition1] += 1

      try:
        exp_data = pd.read_table(
            os.path.join(root, 'progress.txt'),
            delimiter=delimiter,
        )
      except Exception:
        print('Could not read from %s' % os.path.join(root, 'progress.txt'))
        continue
      performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
      exp_data.insert(len(exp_data.columns), 'Unit', unit)
      exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
      exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
      exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance])
      datasets.append(exp_data)
  return datasets


def get_all_datasets(
    all_logdirs,
    legend=None,
    select=None,
    exclude=None,
    verify_logdirs=False,
):
  """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, pull data from it;
        2) if not, check to see if the entry is a prefix for a real directory, and pull data from that.
    """
  logdirs = []
  for logdir in all_logdirs:
    if os.path.isdir(logdir) and logdir[-1] == os.sep:
      logdirs += [logdir]
    else:
      basedir = os.path.dirname(logdir)
      prefix = logdir.split(os.sep)[-1]
      listdir = os.listdir(basedir)
      logdirs += sorted([os.path.join(basedir, x) for x in listdir if prefix in x])
  """
  Enforce selection rules, which check logdirs for certain substrings. Makes it easier to look at graphs
  from particular ablations, if you launch many jobs at once with similar names.
  """
  if select is not None:
    logdirs = [log for log in logdirs if all(x in log for x in select)]
  if exclude is not None:
    logdirs = [log for log in logdirs if all(not (x in log) for x in exclude)]

  if verify_logdirs:
    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
      print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)

  assert not (legend) or (len(legend) == len(logdirs)), "Must give a legend title for each set of experiments."

  # Load data from logdirs
  data = []
  if legend:
    for log, leg in zip(logdirs, legend):
      data += get_datasets(log, leg)
  else:
    for log in logdirs:
      data += get_datasets(log)
  return data


def make_plots(
    all_logdirs: List[str],
    legend: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    file_name: Optional[str] = None,
    xaxis: str = 'TotalEnvInteracts',
    values: Union[str, List[str]] = 'AverageEpRet',
    count: bool = False,
    linewidth: float = 2.5,
    smooth: int = 1,
    select: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    estimator: str = 'mean',
    verify_logdirs: bool = False,
):
  """
  Plot results for provided experiments.

  Args:
    all_logdirs (List[str]): List of directories containing experiment's results.
    legend (Optional[List[str]]): List containing labels for provided list of experiments placed in legend section.
    title (Optional[str]): Plot title.
    figsize (Tuple[int, int]): Size of resulting plot.
    file_name (Optional[str]): File where the plot will be saved. If None, no plot will be saved.
    xaxis (str): Column from `progress` file used as a X axis.
    values (Union[str, List[str]]): Column (or columns) from `progress` file used as a value/Y axis.
    count (bool): Optional flag. By default, the plotter shows y-values which are averaged across all results that share
      an ``exp_name``, which is typically a set of identical experiments that only vary in random seed. But if you'd
      like to see all of those curves separately, set `count` to True.
    linewidth (float): Matplotlib parametr - set the line width in points.
    smooth (int): Smooth data by averaging it over a fixed window. This parameter says how wide the averaging window
      will be.
    select (Optional[List[str]]): Optional selection rule: the plotter will only show curves from logdirs that contain
      all of these substrings.
    exclude (Optional[List[str]]): Optional exclusion rule: plotter will only show curves from logdirs that do not
      contain these substrings.
    estimator (str): Operation done over all results that share the same experiment - choose what to show on main
      curve: mean? max? min? (operation from NumPy)
    verify_logdirs (bool): Wether to print found directories with results.
  """
  data = get_all_datasets(
      all_logdirs,
      legend,
      select,
      exclude,
      verify_logdirs,
  )
  values = values if isinstance(values, list) else [values]
  condition = 'Condition2' if count else 'Condition1'
  estimator = getattr(np, estimator)  # choose what to show on main curve: mean? max? min?
  for value in values:
    plt.figure(figsize=figsize)
    plot_data(
        data,
        xaxis=xaxis,
        value=value,
        condition=condition,
        smooth=smooth,
        estimator=estimator,
        linewidth=linewidth,
    )
  plt.title(title)
  if file_name:
    plt.savefig(file_name, bbox_inches='tight')
  plt.show()
