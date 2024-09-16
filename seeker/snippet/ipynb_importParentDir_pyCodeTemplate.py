#date: 2024-09-16T16:51:30Z
#url: https://api.github.com/gists/c4028cf4e3de861d0dda7c7edf552b57
#owner: https://api.github.com/users/birdflyi

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.7  # Your python version
 
# @Time   : ${DATE} ${TIME}
# @Author : 'Lou Zehua'  # Your name
# @File   : ${NAME}.py 
 
import os
import sys
 
if '__file__' not in globals():
    # !pip install ipynbname  # Remove comment symbols to solve the ModuleNotFoundError
    import ipynbname
 
    nb_path = ipynbname.path()
    __file__ = str(nb_path)
cur_dir = os.path.dirname(__file__)
pkg_rootdir = os.path.dirname(cur_dir)  # Should be the root directory of your project.
if pkg_rootdir not in sys.path:  # To resolve the ModuleNotFoundError
    sys.path.append(pkg_rootdir)
    print('-- Add root directory "{}" to system path.'.format(pkg_rootdir))
