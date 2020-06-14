#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import OrderedDict
import configparser
import sys
import pandas as pd


class LayerDict(OrderedDict):
    def __init__(self):
        super().__init__()
        self.seq = 0
        self.layer = 0

    def __setitem__(self, key, val):
        if key == '# downsample':
            key = 'next_down'
            val = ['1']
        elif key.startswith('#'):
            return
        if isinstance(val, dict):
            if key == 'net':
                return
            if key != 'shortcut':
                self.layer += 1
                val['layer'] = [str(self.layer)]
            self.seq += 1
            key = '%d:' % self.seq + key
        super().__setitem__(key, val)
        return


def main(cname):
    cfg = configparser.ConfigParser(
        defaults=None, dict_type=LayerDict,
        strict=False, empty_lines_in_values=False,
        comment_prefixes=(';'), allow_no_value=True
    )
    cfg.read(cname)
    next_down = False
    cfgdict = OrderedDict()
    for i, s in enumerate(cfg.sections()):
        layer_no, layer_type = s.split(':')
        vdict = dict(cfg.items(s))
        # downsampling
        if next_down:
            vdict['downsampling'] = 1
            next_down = False
        if vdict.get('next_down') == '1':
            next_down = True
            del vdict['next_down']
        # register
        cfgdict[s] = vdict
    df = pd.DataFrame(cfgdict).T
    oname = cname.split('.')[0] + '.csv'
    all_columns = [
        'layer', 'filters', 'size', 'stride', 'pad', 'activation',
        'downsampling', 'batch_normalize', 'from', 'layers'
    ]
    columns = list()
    df_columns = list(df.columns)
    for c in all_columns:
        if c in df_columns:
            columns.append(c)
    df.to_csv(oname, encoding='utf_8_sig', columns=columns)
    return


if __name__ == '__main__':
    main(sys.argv[1])
