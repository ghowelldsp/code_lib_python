#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ghowell
"""

import os

def makeDirectories(fpath):
    """
    Makes all directories in the specified path. The path can include the filename too, although the
    file will not be created.

    :param fpath: path to file
    """
    
    # remove filename if present
    dirs = os.path.dirname(fpath)
    
    # split path into sub directories
    sdirs = dirs.split('/')
    
    # create all sub directories if not aleady present
    curPath = ''
    for sdir in sdirs:
        curPath += sdir + '/'
        if (sdir == '.') or (sdir == '..'):
            continue
        isExist = os.path.exists(curPath)
        if (not isExist):
            os.mkdir(curPath)
    
if __name__ == "__main__":
    
    # test file paths
    fpath1 = './_test_data_/test_dir/test0/file.txt'
    fpath2 = './_test_data_/test_dir/test1/'
    fpath3 = './_test_data_/test_dir'
    
    makeDirectories(fpath1)
    makeDirectories(fpath2)
    makeDirectories(fpath3)
