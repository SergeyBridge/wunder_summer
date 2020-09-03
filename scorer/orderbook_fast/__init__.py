import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})

from .orderbook_fast import OrderBook, EventPlayer
from .orderbook_fast import *