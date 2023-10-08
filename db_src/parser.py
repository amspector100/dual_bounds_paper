import os
import sys
import numpy as np

### Utility functions which improve parsing
def str2bool(v):
    """ Helper function, converts strings to boolean vals""" 
    if isinstance(v, bool):
        return v
    elif isinstance(v, str):
        if v.lower() in ['yes', 'true', 't', 'y', '1']:
            return True
        elif v.lower() in ['no', 'false', 'f', 'n', '0']:
            return False
    return v

def obj2int(v):
    try:
        v = float(v)
        if v.is_integer():
            v = int(v)
        return v
    except: 
        return str2bool(v)

def val2list(val):
    """ Turns discrete values into lists, otherwise returns """
    if not isinstance(val, list):
        if isinstance(val, np.ndarray):
            if len(val.shape) == 1:
                return val
        return [val] 
    return val


def parse_args(args):
    """ Homemade argument parser 
    Usage: --argname value
    Value should be one of:
        - integer/float
        - arbitrary string
        - Alternatively, string of the form
        "start{num}end{num}numvals{num}" 
        which indicates that the parameter ought 
        to be varied along a linear interpolation
        from the specified start/end with the specified
        number of values.
        - Alternatively, a string like
        "[str1, str2, str3]" which will be parsed as
        ["str1", "str2", "str3"]
    The --description argument must be the last argument, as 
    all arguments after the --description argument will be 
    assumed to be part of the description.
    """
    
    # Get rid of script name
    args = args[1:]

    # Initialize kwargs constructor
    kwargs = {}
    key = None
    description_index = None # At the end, can write description

    #Parse
    for i, arg in enumerate(args):
        arg = str(arg).lower()

        # Even placements should be keyword
        if i % 2 == 0:
            if str(arg)[0:2] != '--':
                raise ValueError(
                    f'{i}th arg ({arg}) should be a kwarg name preceded by "--"'
                )
            key = arg[2:]

            # Description
            if key == 'description':
                description_index = i
                break

        # Parse values
        if i % 2 == 1:

            # Check if it's a list written out
            if arg[0] == '[':
                # Strip brackets, whitspace, and split by commas
                value = arg.replace('[', '').replace(']', '')
                value = value.replace(' ', '').split(',')
                # Try to process
                value = [obj2int(v) for v in value]

            # Check if it's start{num}end{num}numvals{num}
            elif arg[0:5] == 'start':
                start = float(arg.replace('start', '').split('end')[0])
                end = float(arg.split('end')[-1].split('numvals')[0])
                numvals = int(arg.split('numvals')[-1])
                value = np.linspace(
                    start, end, numvals
                )

            # Apply obj2int (preserves strings, infers floats, bools, ints)
            else:
                value = obj2int(arg)


            kwargs[key] = val2list(value)

    # Parse description 
    description = ''
    if description_index is not None:
        description += (' ').join(args[description_index+1:])
        description += '\n'
    description = 'Arguments were: \n'
    i = 0
    arg_msg = ''
    max_ind = len(args) if description_index is None else description_index
    while i < max_ind:
        description += f'{args[i]}={args[i+1]}\n'
        i += 2
    kwargs['description'] = description
    return kwargs