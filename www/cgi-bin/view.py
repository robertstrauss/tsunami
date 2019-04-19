#!/usr/bin/python
import cgi
import sys
import pickle
form = cgi.FieldStorage()

print("Content-type: text/html")
print("")

def make_form():
    sform = ['latstart', 'latend', 'lonstart', 'lonend', 'tsulat', 'tsuamp', 'tsutype', 'tsuwx', 'tsuwy', 'theta']
    dc = {}
    for i in sform:
       tmp= form.getfirst(i, 'empty')
       dc[i] = cgi.escape(tmp)
    return dc

f = make_form()



print('<a href="../output.html">click here to view output</a>')


with open('runinfo.dict', 'w') as pklfile:
    pickle.dump(f, pklfile)


#
# cargs = []
#
# for k,v in f.items():
#     cargs.append(k)
#     cargs.append(v)
#
#
# import subprocess
#
# subprocess.Popen([sys.executable, './controll.py'] + cargs) # 'python3',
#
