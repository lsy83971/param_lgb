import math
import numpy as np
import pandas as pd


gg = pd.Series(range(100000))
from datetime import datetime
print(datetime.now())
sb = list(range(5000))
for i in range(10000):
    gg.loc[sb]
print(datetime.now())

sb = gg.index.isin(range(5000))
print(datetime.now())
for i in range(10000):
    gg.loc[sb]
print(datetime.now())


j = pd.Series([1, 2, 3, 4])
sb = pd.Series(dir(j))
sb[sb.str.contains("iter")]

k = j.__iter__()
k = j.iteritems()
k.__next__()

# 1. quantile(10)
# 2. < s <=s
# 3.




class _gd_y(pd.Series):
    """
    sigmoid function: y1=(1/(1+exp(x)))
    objective function: obj=ln(abs(y-y1))
    
    this class is to calculate the
    First and Second deravatives
    of obj to x.
    """
    
    def gd(self, y):
        _exp = np.exp(self)
        self.g1 = 1 / (1 + _exp) - y
        self.g2 = -_exp / (self.g1 ** 2)




x = pd.DataFrame(np.random.random([1000, 10]))
y = (pd.Series(np.random.random(1000)) > 0.8)

param = pd.DataFrame(np.random.random([1000, 5]))
quant = 10


sbb = _gd_t(min_cnt=50)
sbb.data(param, x, y)



for i in sbb.iter_sep():
    print(i)

i[0]
i[1]
i[2]. value_counts()

class _gd_t:
    def __init__(self, quant=10, min_cnt=1000, ycls=None):
        self.quant = quant
        self.min_cnt = min_cnt
        pass

    def data(self, param, x, y, weight=None, y1=None):
        """
        ×¢ÈëÑµÁ·Êý¾Ý
        """
        self.param = param
        self.param2 = (param ** 2)
        self.x = x
        self.y = y
        
        if y1 is None:
            y_logit = math.log(1 - y.mean())
            y1 = pd.Series(y_logit, index=y.index)
        self.y1 = _gd_y(y1)
        self.y1.gd(self.y)

    def fit(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        for i in self.find_sep(x):
            pass

    def iter_sep(self, mask=None):
        for i in self.x.columns: ## TODO:random columns
            _xi1 = self.x[i]
            if mask is not None:
                _xi = self.x[i]. loc[mask]
            else:
                _xi = _xi1

            _sep = _xi. quantile([i / self.quant for i in range(1, self.quant)]).drop_duplicates()

            if mask is not None:
                gp_mask = mask.astype(int)
                cnt = mask.sum()
            else:
                cnt = self.x.shape[0]
                gp_mask = pd.Series(1, index=self.x.index)


            sep1 = list()
            for j in _sep:
                if mask is not None:
                    gp_mask1 = (_xi1 < j) & mask
                else:
                    gp_mask1 = (_xi1 < j)

                cnt1 = gp_mask1.sum()
                if (cnt1 >= self.min_cnt):
                    if (cnt - cnt1 >= self.min_cnt):
                        gp_mask += gp_mask1
                        sep1.append(j)

            if len(_sep) > 0:
                yield (i, _sep, gp_mask)

    def iter_sep_obj(self, mask=None):
        for i, j, _m in self.iter_sep(mask):
            mask = _m > 0
            p1 = self.param.loc[_mask]
            p2 = self.param2.loc[_mask]
            g1 = self.y1.g1.loc[_mask]
            g2 = self.y1.g2.loc[_mask]            
            m = _m.loc[_mask]

            g1_p1 = p1.multiply(g1, axis=0)
            g2_p2 = p2.multiply(g2, axis=0)
            g1_gp = g1_p1.groupby(m).agg(sum).cumsum()
            g2_gp = g2_p2.groupby(m).agg(sum).cumsum()
            g1_t = g1_gp.iloc[ - 1]
            g2_t = g2_gp.iloc[ - 1]
            g1_gp_l = g1_gp.iloc[: -1]
            g2_gp_l = g2_gp.iloc[: -1]            
            g1_gp_r = (g1_t - g1_gp_l)
            g2_gp_r = (g2_t - g2_gp_l)

            obj_l = g1_gp_l ** 2 / g2_gp_l
            obj_r = g1_gp_r ** 2 / g2_gp_r
            
            obj_t = g1_t ** 2 / g2_t
            obj_delta = obj_t - obj_l + obj_r

            yield i, j, obj_delta, g2_gp_l, g2_gp_r





for i in sbb.iter_sep():
    print(i)

i[0]
i[1]
i[2]

                
    def obj(self, m1, i):


        
        pass

    def obj_delta(self, m1, m2, m3):

        if mask is not None:
            _m1 = _m & mask
        else:
            _m1 = _m

        if mask is not None:
            _m2 = (~_m) & mask
        else:
            _m2 = (~_m)
        
        self.obj(_m1) + self.obj(_m2) - self.obj(_m3)


    

    def trans(self):
        pass

        











