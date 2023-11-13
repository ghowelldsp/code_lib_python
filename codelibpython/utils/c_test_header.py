#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ghowell
"""

import numpy as np
import re
import os
import warnings

def writeDataLine(data, vout, nDim):
        
    # split the data into block with a max length of 1000 as array2string doesn't handle large arrays
    maxLen = 1000
    nBlocks = int(len(data)/maxLen)
    nRemain = len(data) % maxLen
    
    # make the space offset larger for arrays that have greater than 1 dimesion to account for the inner 
    # curly braces
    if (nDim > 1):
        spaceOffset = ' ' * 8
    else:
        spaceOffset = ' ' * 4
    
    # write blocks of data
    for i in range(nBlocks):
        dataLine = np.array2string(data[maxLen*i:maxLen*(i+1)], 
                                   max_line_width=116, 
                                   formatter={'float_kind':'{:.21}'.format},
                                   separator=', ')
        dataLine = re.sub(r"\n", "\n   ", dataLine)
        dataLine = spaceOffset + dataLine[1:-1]
        if ((i == nBlocks-1) and (nRemain == 0)):
            vout.append(dataLine)
        else:
            vout.append(dataLine + ',')
    
    # write remaining data elements
    if (nRemain):
        dataLine = np.array2string(data[-nRemain:], 
                                   max_line_width=116, 
                                   formatter={'float_kind':'{:.21}'.format}, 
                                   separator=', ')
        dataLine = re.sub(r"\n", "\n   ", dataLine)
        vout.append(spaceOffset + dataLine[1:-1])

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

def addVariables1d(vout, mout, name, data, dataType, xName):
    
    # length of dimensions
    xLen = data.size
    
    # create row macros definitions
    mout.append(addMacro(xName, xLen))
    
    # create empty dout matricies
    if (name == 'doutRef'):
        for i in range(ndepth):
            vout.append(f'{dataType} dout[{xName}] = ' + '{0};\n')
    
    # write definition line
    vout.append(f'{dataType} {name}[{xName}] =')
    vout.append('{') 
    writeDataLine(data, vout, 1)  
    vout.append('};\n')

def addVariables2d(vout, mout, name, data, dataType, xName, yName):
    
    # create name with first letter capitalised
    cname = f'{name[0].capitalize()}{name[1:]}'
    
    # length of dimensions
    xLen, yLen = data.shape
        
    # create row and column macros definitions
    mout.append(addMacro(xName, xLen))
    mout.append(addMacro(yName, yLen))
    
    # write definition line
    vout.append(f'{dataType} {name}[{xName}][{yName}] =')
    vout.append('{')
    
    # write row of data
    for i in range(xLen):
        vout.append('    {')
        writeDataLine(data[i,:], vout, 2)
        if (i < xLen-1):
            vout.append('    },')
        else:
            vout.append('    }')
    vout.append('};\n')
    
    # create pointer to pointer matricies
    vout.append(f'{dataType} *pp{cname}[{xName}] =')
    vout.append('{')
    for i in range(xLen):
        line = f'    {name}[{i}]'
        if (i < xLen-1):
            vout.append(line + ',')
        else:
            vout.append(line)
    vout.append('};\n')
    
    # if the variable is data out reference create dout variables
    if (name == 'doutRef'):
        
        # create empty 2d matrix
        vout.append(f'{dataType} dout[{xName}][{yName}] = ' + '{0};\n')
        
        # create pointer to pointer matrix
        vout.append(f'{dataType} *ppDout[{xName}] =')
        vout.append('{')
        for i in range(xLen):
            line = f'    dout[{i}]'
            if (i < xLen-1):
                vout.append(line + ',')
            else:
                vout.append(line)
        vout.append('};\n')
    
    # create structure
    vout.append(f'{name}_t s{cname} =')
    vout.append('{')
    vout.append(f'    .p{cname} = {name}[0],')
    if (name == 'doutRef'):
        vout.append(f'    .pp{cname} = pp{cname},')
        vout.append(f'    .pDout = dout[0],')
        vout.append(f'    .ppDout = ppDout')
    else:
        vout.append(f'    .pp{cname} = pp{cname}')
    vout.append('};\n')

def addVariables3d(vout, mout, name, data, dataType, xName, yName, zName):
    
    # create name with first letter capitalised
    cname = f'{name[0].capitalize()}{name[1:]}'
    
    # length of dimensions
    xLen, yLen, zLen = data.shape
        
    # create row and column macros definitions
    mout.append(addMacro(xName, xLen))
    mout.append(addMacro(yName, yLen))
    mout.append(addMacro(zName, zLen))
    
    # create data matricies
    for i in range(zLen):
        
        # write definition line
        vout.append(f'{dataType} {name}_{i}[{xName}][{yName}] =')
        vout.append('{')
        # write row of data
        for j in range(xLen):
            vout.append('    {')
            writeDataLine(data[j,:,i], vout, 3)
            if (j < xLen-1):
                vout.append('    },')
            else:
                vout.append('    }')
        vout.append('};\n')
         
        # create pointer to pointer matricies
        vout.append(f'{dataType} *pp{cname}_{i}[{xName}] = ')
        vout.append('{')
        for j in range(xLen):
            line = f'    {name}_{i}[{j}]'
            if (j < xLen-1):
                vout.append(line + ',')
            else:
                vout.append(line)
        vout.append('};\n')
            
    # if the variable is data out reference create dout variables
    if (name == 'doutRef'):
        for i in range(zLen):
        
            # create empty 2d matrix
            vout.append(f'{dataType} dout_{i}[{xName}][{yName}] = ' + '{0};\n')
        
            # create pointer to pointer matrix
            vout.append(f'{dataType} *ppDout_{i}[{xName}] =')
            vout.append('{')
            for j in range(xLen):
                line = f'    dout_{i}[{j}]'
                if (j < xLen-1):
                    vout.append(line + ',')
                else:
                    vout.append(line)
            vout.append('};\n')
            
    # create structure
    vout.append(f'{name}_t s{cname}[{zName}] =')
    vout.append('{')
    for i in range(zLen):
        vout.append('    {')
        vout.append(f'        .p{cname} = {name}_{i}[0],')
        if (name == 'doutRef'):
            vout.append(f'        .pp{cname} = pp{cname}_{i},')
            vout.append(f'        .pDout = dout_{i}[0],')
            vout.append(f'        .ppDout = ppDout_{i}')
        else:
            vout.append(f'        .pp{cname} = pp{cname}_{i}')
        if (i < zLen-1):
            vout.append('    },')
        else:
            vout.append('    }')
    vout.append('};\n')

def addVariables(name, data, dimNames):
    
    # create empty variable and macro lists
    vout = []
    mout = []
    tout = []
    
    # unsupported datatypes list
    unsupportedDataTypes = ['complex128', 'complex256']
    
    # get array specs
    dataType = f'{data.dtype}_t'
    dataDim = data.ndim
    
    # check for unsupported datatypes
    if dataType in unsupportedDataTypes:
        warnings.warn(f'variable {name} with type {dataType} is not a supported data type')
        
    vout.append(f'/** {name} */')
    
    # create name with first letter capitalised
    cname = f'{name[0].capitalize()}{name[1:]}'
    
    # write array or matrix
    if (dataDim == 1):
        # create 1d variables
        addVariables1d(vout, mout, name, data, dataType, dimNames[0])
        
    elif (dataDim > 1):
        # create typedef structure
        tout.append(f'typedef struct {name}_t')
        tout.append('{')
        tout.append(f'    {dataType} *p{cname};')
        tout.append(f'    {dataType} **pp{cname};')
        if (name == 'doutRef'):
            tout.append(f'    {dataType} *pDout;')
            tout.append(f'    {dataType} **ppDout;')
        tout.append('}' + f' {name}_t;\n')

        if (dataDim == 2):
            # create 2d variables
            addVariables2d(vout, mout, name, data, dataType, dimNames[0], dimNames[1])
            
        elif (dataDim == 3):
            # create 3d variables
            addVariables3d(vout, mout, name, data, dataType, dimNames[0], dimNames[1], dimNames[2])
        
        else:
            warnings.warn(f'variable {name} has {dataDim} is not a supported number of dimensions, max dimensions = 3')
             
    return vout, mout, tout
    
def createCTestHeader(file, macros, variables):  
    
    # create header data lists
    hout = []
    mout = []
    vout = []
    tout = []
    
    # create variable list
    for variable in variables:
        
        voutVar, moutVar, toutVar = addVariables(variable[0], variable[1], variable[2:len(variable)])
        for line in voutVar:
            vout.append(line)
        for line in moutVar:
            mout.append(line[0])
        for line in toutVar:
            tout.append(line)
    
    # create macro list
    for macro in macros:
        moutVar = addMacro(macro[0], macro[1])
        mout.append(moutVar[0])
        
    # remove duplicate macros
    mout = list(dict.fromkeys(mout))
    
    # add info
    hout.append('/***********************************************************************************************************************')
    hout.append(' *')
    hout.append(f' * @file    {os.path.basename(file)}')
    hout.append(' *')
    hout.append(' * @brief   Test Variables')
    hout.append(' *')
    hout.append(' * @par')
    hout.append(' * @author  G. Howell')
    hout.append(' *')
    hout.append(' **********************************************************************************************************************/')
    hout.append('')
    
    # add includes
    hout.append('/*------------------------------------------- INCLUDES ---------------------------------------------------------------*/')
    hout.append('')
    hout.append('#include <stdint.h>')
    hout.append('')
    
    # add macros
    hout.append('/*------------------------------------------- MACROS AND DEFINES -----------------------------------------------------*/')
    hout.append('')
    for line in mout:
        hout.append(line)
    hout.append('')
    
    # add typedefs
    hout.append('/*------------------------------------------- TYPEDEFS ---------------------------------------------------------------*/')
    hout.append('')
    hout.append('typedef float float32_t;')
    hout.append('typedef double float64_t;')
    hout.append('')
    for line in tout:
        hout.append(line)
    hout.append('')
    
    # add variables
    hout.append('/*------------------------------------------- EXPORTED VARIABLES -----------------------------------------------------*/')
    hout.append('')
    for line in vout:
        hout.append(line)
    hout.append('')

    # write data to file
    hout = '\n'.join(hout)
    with open(file, 'w') as f:
        f.write(hout)

if __name__ == "__main__":
    
    print('C Test Header Started')
    
    print('... creating macros')
    
    macroVal1 = 10
    macroVal2 = -20
    macroVal3 = 3.14
    
    macros = [['MACRO_INT', macroVal1], 
              ['MACRO_UINT', macroVal2],
              ['MACRO_FLOAT', macroVal3]]
    
    print('... creating variables')
    
    # 1d data
    data_1d_f32 = np.array([1.2453, 5.7, 9.3], dtype=np.float32)
    data_1d_i32 = np.array([1, 5, 9], dtype=np.int32)
    data_long_1d_i32 = np.random.randn(3000).astype(np.int32)
    data_long_1d_f32 = np.random.randn(3000).astype(np.float32)
    
    # 2d data
    data_2d_f32 = np.array([[1.2, 5.7, 9.3],[5.3, 8.8, 2.4]], dtype=np.float32)
    
    # 3d data
    data_3d_f32 = np.random.randn(4, 3, 2).astype(np.float32)
    data_long_3d_f32 = np.random.randn(5, 50, 3).astype(np.float32)
    
    variables = [['data_1d_f32', data_1d_f32, 'N'], 
                 ['data_1d_i32', data_1d_i32, 'N'],
                 ['data_long_1d_i32', data_long_1d_i32, 'N'],
                 ['data_long_1d_f32', data_long_1d_f32, 'N'],
                 ['data_2d_f32', data_2d_f32, 'N_CHANNELS_MATRIX', 'N'],
                 ['doutRef', data_2d_f32, 'N_CHANNELS_MATRIX', 'N'],
                 ['data_3d_f32', data_3d_f32, 'N_CHANNELS_MATRIX', 'N', 'N_BANDS'],
                 ['doutRef', data_3d_f32, 'N_CHANNELS_MATRIX', 'N', 'N_BANDS'],
                 ['data_long_3d_f32', data_long_3d_f32, 'N_X', 'N_Y', 'N_Z']]
    
    print('... creating header')
    
    createCTestHeader('./data/header.h', macros, variables)
    
    print('C Test Header Finished')
        