

import pandas as pd
from abc import ABCMeta, abstractmethod

from typing import TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')

'''
Given: 
    [data]: a dataframe describes program traces from different initial states
Return:
    [invlist]: a list of pairs (inv,loss), where 
               [inv] is a candidated invariant, and [loss] is its loss on [data]. 
'''
class Learner:
    @abstractmethod
    def learn_inv(self, data:PandasDataFrame):
        pass
