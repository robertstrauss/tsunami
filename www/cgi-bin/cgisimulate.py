#!/usr/bin/python
import cgi
# form = cgi.FieldStorage()

print("Content-type: text/html")
print("")
print('asdf')

def make_form():
    sform = ['latstart', 'latend', 'lonstart', 'lonend', 'tsulat', 'tsuamp', 'tsutype', 'tsuwx', 'wsuwy', 'theta']
    dc = {}
    for i in sform:
       tmp= form.getfirst(i, 'empty')
       dc[i] = cgi.escape(tmp)
    return dc


print(make_form())

f = make_form()



if ('empty' in f.values()):# || simtime > 10000):
    print("you did not fill out all the boxes, or entered an invalid")
    exit()

# import subprocess

# subprocess.call(['python3', *['--'+n+' '+f[n] for n in f.keys()]])
