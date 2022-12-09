import math
import numpy as np
import pandas as pd

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

class _gd_t:
    def __init__(self,
                 quant=10,
                 min_cnt=1000,
                 l2=100,
                 trace=2000,
                 max_depth=4,
                 max_nodes=17
                 ):
        self.quant = quant
        self.min_cnt = min_cnt
        self.l2 = l2
        self.trace = trace
        self.max_depth = max_depth
        self.max_nodes = max_nodes


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
            cntl = list()
            cntr = list()
            for j in _sep:
                if mask is not None:
                    gp_mask1 = (_xi1 < j) & mask
                else:
                    gp_mask1 = (_xi1 < j)

                cnt1 = gp_mask1.sum()
                if (cnt1 >= self.min_cnt):
                    if ((cnt - cnt1) >= self.min_cnt):
                        gp_mask += gp_mask1
                        sep1.append(j)
                        cntl.append(cnt1)
                        cntr.append(cnt - cnt1)

            if len(_sep) > 0:
                yield (i, sep1, cntl, cntr, gp_mask)

    def iter_sep_obj(self, mask=None):
        res_list = list()
        for i, j, cntl, cntr, _m in self.iter_sep(mask):
            _mask = _m > 0
            p1 = self.param.loc[_mask]
            p2 = self.param2.loc[_mask]
            g1 = self.y1.g1.loc[_mask]
            g2 = self.y1.g2.loc[_mask]
            m0 = _m.loc[_mask]
            m = m0. reset_index(drop=True)

            idx_gp = {k[0]:k[1]. index for k in m.groupby(m)}
            
            g2_matrix = p2.multiply(g2, axis=0).values[:, :, None]*p2.values[:, None, :]
            g2_agg = np.array([g2_matrix[k2]. sum(axis=0) for k1, k2 in idx_gp.items()])
            g2_cumsum = g2_agg.cumsum(axis=0)
            
            g1_matrix = p1.multiply(g1, axis=0)            
            g1_cumsum = g1_matrix.groupby(m0).agg(sum).cumsum()
            
            g1_t = g1_cumsum.iloc[ - 1]
            g2_t = g2_cumsum[ - 1]
            
            g1_l = g1_cumsum.iloc[: -1]
            g2_l = g2_cumsum[: -1]
            g1_r = (g1_t - g1_l)
            g2_r = (g2_t - g2_l)

            n_sample = len(idx_gp) - 1
            n_idx = g1_t.shape[0]
            l2_eye = -np.eye(n_idx)*self.l2

            g2_r_inv = np.linalg.inv(g2_r + l2_eye)
            gd_r = -np.matmul(g1_r.values[:, None, :], g2_r_inv)[:, 0, :] / 2            
            obj_r = (g1_r * gd_r).sum(axis=1) / 2
            trace_r = np.diagonal(g2_r, axis1=1, axis2=2).sum(axis=1)            
            
            g2_l_inv = np.linalg.inv(g2_l + l2_eye)
            gd_l = -np.matmul(g1_l.values[:, None, :], g2_l_inv)[:, 0, :] / 2
            obj_l = (g1_l * gd_l).sum(axis=1) / 2
            trace_l = np.diagonal(g2_l, axis1=1, axis2=2).sum(axis=1)
            
            g2_t_inv = np.linalg.inv(g2_t + l2_eye)
            gd_t = -(g1_t@g2_t_inv) / 2
            obj_t = (g1_t * gd_t).sum() / 2
            delta_obj = obj_r + obj_l - obj_t

            res = pd.DataFrame(columns=["idx", "delta", "trace_l", "trace_r", "threshold", "cntl", "cntr"], index=range(n_sample))
            res["delta"] = delta_obj.values
            res["trace_l"] = trace_l
            res["trace_r"] = trace_r
            res["threshold"] = j
            res["idx"] = i
            res["cntl"] = cntl
            res["cntr"] = cntr
            res_list.append(res)

        if len(res_list) > 0:
            return pd.concat(res_list)
        else:
            return pd.DataFrame([])

class _gd_node_info:
    def __init__(self, info=None):
        if info is None:
            self.info = []
            self.root = self
            self.leaves = [self]
        else:
            self.info = info
            
    def save(self):
        pass

    def trans(self):
        pass

    def calc_mask(self, x):
        if len(self.info) == 0:
            self.mask = pd.Series(True, index=x.index)
        else:
            idx, v, lr= self.info[ - 1]
            if lr == "l":
                cond = x[idx] < v
            else:
                cond = x[idx] >= v
            self.mask = self.p.mask & cond
            

class _gd_node(_gd_node_info):
    def __init__(self, gd, info=None, p=None):
        self.gd = gd
        if info is None:
            self.info = []
            self.root = self
            self.leaves = [self]
        else:
            self.info = info
            self.p = p
            self.root = self.p.root

        self.calc_mask()

    def split(self, idx, v):
        info_l = self.info.copy()
        info_l.append((idx, v, "l"))
        self.l = _gd_node(self.gd, info_l, p=self)
        self.l.root = self.root

        info_r = self.info.copy()
        info_r.append((idx, v, "r"))
        self.r = _gd_node(self.gd, info_r, p=self)
        self.r.root = self.root
        
        self.root.leaves.pop(self.root.leaves.index(self))
        self.root.leaves.append(self.l)
        self.root.leaves.append(self.r)
        
    def calc_mask(self):
        super().calc_mask(self.gd.x)

    def obj(self):
        if hasattr(self, "obj_info"):
            return
        self.obj_info = self.gd.iter_sep_obj(self.mask)
        if self.obj_info.shape[0] == 0:
            self.obj_info_bst = None
        self.obj_info_bst = self.obj_info.iloc[self.obj_info["delta"]. argmax()]

    def best_select(self):
        bd = self.best_dict
        bd1 = bd[(bd["trace_l"] < -self.gd.trace) & (bd["trace_r"] < -self.gd.trace)]
        if bd1.shape[0] <= 0:
            return None, None
        
        idx0 = bd1.index[bd1["delta"]. argmax()]
        it = bd1.loc[idx0]
        return idx0, it
        
    def best_split(self):
        best_dict = dict()
        for j, i in enumerate(self.leaves):
            i.obj()
            bst = i.obj_info_bst
            if bst is None:
                continue
            best_dict[j] = bst

        if len(best_dict) == 0:
            return None
        
        best_dict = pd.DataFrame(best_dict).T
        self.best_dict = best_dict
        idx0, it = self.best_select()
        if idx0 is None:
            return None
        
        idx = it["idx"]
        v = it["threshold"]
        self.leaves[idx0]. split(idx, v)
        return 1

    def rec_split(self):
        while True:
            _res = self.best_split()
            if _res is None:
                break


if __name__ == "__main__":
    x = pd.DataFrame(np.random.random([10000, 10]))
    y = (pd.Series(np.random.random(10000)) > 0.8)
    param = pd.DataFrame(np.random.random([10000, 10]))
    sbb = _gd_t(min_cnt=1000)
    sbb.data(param, x, y)
    sbb = _gd_t(min_cnt=50)
    sbb.data(param, x, y)
    gn = _gd_node(sbb)
    gn.rec_split()
    [i.mask.sum() for i in gn.leaves]




