#date: 2022-06-22T17:14:24Z
#url: https://api.github.com/gists/fb1688ecaa6c5078081923b7d5a10c9f
#owner: https://api.github.com/users/GregSikoraPhD

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from typing import Union, Tuple

def eipf(data: npt.NDArray, 
         grid_x: Union[int, npt.NDArray] = 100, 
         grid_y: Union[int, npt.NDArray] = None, 
         plot: bool = False) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, go.Figure]:
  # Grid X:
  if type(grid_x) == int:
    data_min = min(data)
    data_max = max(data)
    step = (data_max-data_min)/grid_x
    args_x = np.arange(data_min, data_max+step, step)
  else:
    args_x = grid_x
  # Grid Y:
  if not grid_y:
    args_y = args_x
  elif type(grid_y) == int:
    data_min = min(data)
    data_max = max(data)
    step = (data_max-data_min)/grid_y
    args_y = np.arange(data_min, data_max+step, step)
  else:
    args_y = grid_y
  # Eipf:
  eipf = np.ones((len(args_y),len(args_x)))*np.nan
  for i, a in enumerate(args_x):
    for j, b in enumerate(args_y):
      if a < b:
        eipf[j, i] = 100*len(data[(data >= a) & (data <= b)])/sample_len
  # Plot:      
  if plot:
    fig = go.Figure(data=go.Heatmap(x=args_x, y=args_y, z=eipf, colorscale='rainbow',
                                colorbar=dict(title='[%]'), name='', hovertemplate = '%{z:.2f}% measurements in [%{x:.2f}, %{y:.2f}]'))
    fig.update_layout(
      title='Empirical interval probability function, % of measurements in [x,y]',
      width=800, 
      height=800,
      xaxis=dict(title='x', range=[data_min, data_max]),
      yaxis=dict(title='y', range=[data_min, data_max]),
    )
    fig.update_xaxes(showspikes=True, spikemode="across")
    fig.update_yaxes(showspikes=True, spikemode="across")
  return args_x, args_y, eipf, fig