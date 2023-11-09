#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ghowell
"""

import numpy as np
import re
import warnings

def addMacro(name, value):
    
    # create empty macro lists
    mout = []
    
    # determine which data type value is
    if isinstance(value, int):
        # determine datatype suffix
        if (value >= 0):
            dsuffix = 'U'
        else:
            dsuffix = ''
    elif isinstance(value, float):
        dsuffix = 'F'
    else:
        warnings.warn(f'variable {name} with type {type(value)} is not a supported data type')
    
    # append macro
    mout.append(f'#define {name: <40}({value}{dsuffix})')
    
    return mout

def addVariables(name, data, rowName, colName, maxLineWidth):
    
    # create empty variable and macro lists
    vout = []
    mout = []
    
    # unsupported datatypes list
    unsupportedDataTypes = ['complex128', 'complex256']
    
    # get arraya specs
    datatype = data.dtype
    dataDim = data.ndim
    
    # check for unsupported datatypes
    if datatype in unsupportedDataTypes:
        warnings.warn(f'variable {name} with type {datatype} is not a supported data type')
    
    # write array or matrix
    if (dataDim == 1):
        
        nrows = data.size
        
        # create row macros definitions
        mout.append(addMacro(rowName, nrows))
        
        # write definition line
        defLine = f'{datatype}_t {name}[{rowName}] = '
        vout.append(defLine + '{')
        
        # write row of data
        dataLine = np.array2string(data, max_line_width=maxLineWidth, precision=23, separator=', ')
        dataLine = re.sub(r"\n", "\n   ", dataLine)
        vout.append('    ' + dataLine[1:-1])
        vout.append('    };\n')
        
    elif (dataDim == 2):
               
        nrows, ncols = data.shape
        
        # create row and column macros definitions
        mout.append(addMacro(rowName, nrows))
        mout.append(addMacro(colName, ncols))
        
        # write definition line
        defLine = f'{datatype}_t {name}[{rowName}][{colName}] = '
        vout.append(defLine + '{')
        
        # write row of data
        for i in range(nrows):
            dataLine = np.array2string(data[i,:], max_line_width=maxLineWidth, precision=23, separator=', ')
            dataLine = re.sub(r"\n", "\n    ", dataLine)
            if (i != nrows-1):
                vout.append('    {' + dataLine[1:-1] + '},')
            else:
                vout.append('    {' + dataLine[1:-1] + '}')
                vout.append('    };\n')
        
    else:
        warning.warn(f'variable {name} has {dataDim} is not a supported number of dimensions, max dimension = 2')
        
    return vout, mout
    
def createCTestHeader(file, macros, variables, maxLineWidth):  
    
    # create header data lists
    hout = []
    mout = []
    vout = []
    
    # create variable list
    for variable in variables:
        voutVar, moutVar = addVariables(variable[0], variable[1], variable[2], variable[3], maxLineWidth)
        for line in voutVar:
            vout.append(line)
        for line in moutVar:
            mout.append(line[0])
    
    # create macro list
    for macro in macros:
        moutVar = addMacro(macro[0], macro[1])
        mout.append(moutVar[0])
        
    # remove duplicate macros
    mout = list(dict.fromkeys(mout))
    
    # add info
    hout.append('/' + '*' * (maxLineWidth-2))
    hout.append(' *')
    hout.append(' * C Test Header')
    hout.append(' *')
    hout.append(' ' + '*' * (maxLineWidth-3) + '/\n')
    
    # add includes
    title = 'INCLUDES'
    hout.append('/** ' + title + ' ' + '*' * (maxLineWidth - 7 - len(title)) + '/\n')
    hout.append('#include <stdint.h>')
    hout.append('')
    
    # add macros
    title = 'MACROS'
    hout.append('/** ' + title + ' ' + '*' * (maxLineWidth - 7 - len(title)) + '/\n')
    for line in mout:
        hout.append(line)
    hout.append('')
    
    # add typedefs
    title = 'TYPEDEFS'
    hout.append('/** ' + title + ' ' + '*' * (maxLineWidth - 7 - len(title)) + '/\n')
    hout.append('typedef float float32_t;')
    hout.append('typedef double float64_t;')
    hout.append('')
    
    # add variables
    title = 'VARIABLES'
    hout.append('/** ' + title + ' ' + '*' * (maxLineWidth - 7 - len(title)) + '/\n')
    for line in vout:
        hout.append(line)
    hout.append('')

    # write data to file
    hout = '\n'.join(hout)
    with open(file, 'w') as f:
        f.write(hout)

if __name__ == "__main__":
    
    maxLineWidth = 120  
    
    print('C Test Header Started')
    
    print('... creating macros')
    
    macroVal1 = 10
    macroVal2 = -20
    macroVal3 = 3.14
    
    macros = [['MACRO_INT', macroVal1], 
              ['MACRO_UINT', macroVal2],
              ['MACRO_FLOAT', macroVal3]]
    
    print('... creating variables')
    
    arrayf32 = np.array([1.2453, 5.7, 9.3], dtype=np.float32)
    matrixf32 = np.array([[1.2, 5.7, 9.3],[1.2, 5.7, 9.3]], dtype=np.float32)
    matrixi32 = np.array([[1, 5, 9],[1, 5, 9]], dtype=np.int32)
    matrixf64 = np.random.rand(100, 100)
    
    variables = [['arrayf32', arrayf32, 'N_CHANNELS_ARRAY', 'N'], 
                 ['matrixf32_1', matrixf32, 'N_CHANNELS_MATRIX', 'N'],
                 ['matrixf32_2', matrixf32, 'N_CHANNELS_MATRIX', 'N'],
                 ['matrixi32', matrixi32, 'N_CHANNELS_MATRIX', 'N'],
                 ['matrixf64', matrixf64, 'N_CHANNELS_RAND', 'N_CHANNELS_RAND']]
    
    print('... creating header')
    
    createCTestHeader('./data/header.h', macros, variables, maxLineWidth)
    
    print('C Test Header Finished')
        