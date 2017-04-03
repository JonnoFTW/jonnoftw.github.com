---
layout: post
title: "OpenCL External Tool for IntelliJ"
description: ""
category: 
tags: ['intellij', 'python', 'pycharm']
---
{% include JB/setup %}

In this post I will describe how to setup an external tool in your IntelliJ IDE that will give you an easy way to compile your kernel 
source and provide hyperlinked error messages. The code for `compile.py` is here (make sure you do `chmod +x compile.py`): 

```
#!/usr/bin/env python
from __future__ import print_function
import pyopencl as cl
import sys
import os
import re

if __name__ == "__main__":
    dev = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([dev])
    fname = sys.argv[1]
    with open(fname, 'r') as src_file:
        src = src_file.read()
        fl = fname.lower()
        if fl.endswith('.py'):
            # pick out the string from the file that looks like a cl kernel

            docstrings = list(re.finditer('"""', src))
            allstrings = [(src[i[0].regs[0][1] + 1:i[1].regs[0][0] - 1], i[0].regs[0][0]) for i in zip(docstrings[::2], docstrings[1::2])]
            strings, positions = zip(*[s for s in allstrings if "//CL-KERNEL" in s[0]])
            cl_src = "\n".join(strings)
        elif fl.endswith('.cl'):
            cl_src = src
        print("Compiling {} on {}".format(fname, dev))
        try:
            prog = cl.Program(ctx, cl_src).build(options=['-I', os.path.dirname(src_file.name)])
    
        except cl.RuntimeError as e:
            out = str(e)
            out = re.sub(r'"/tmp/(.*)\.cl"', src_file.name, out).replace('\n\n', '\n')
            line_offset = src[:positions[0]].count("\n") + 1
            out = re.sub(r', line (\d+):', lambda m:":"+str(int(m.group(1)) + line_offset), out)
            print(out)#, file=sys.stderr)
    exit(1) # highlighting only occurs on non-zero exit status

```
Save this script somewhere and make sure you have `pyopencl` and python installed. If you are using pyopencl, you will need to mark the triple quoted
strings that contain the kernel source code with a `//CL-KERNEL` somewhere in the code. The code could easily be modified to extract the kernel source
from other languages such as Java.

In your IntelliJ IDE, go File -> Settings -> Tools -> External Tools -> Click the create button and fill out the form like so: 

![IDE screenshot](https://i.imgur.com/9xL9KAm.png)

Next, create a filter from the "Output Filters" button. Create a new filter that has the following regex
```
$FILE_PATH$:$LINE$
```

Now you can easily right click on a file, click External Tools -> Compile OpenCL

![full run](https://i.imgur.com/xWabMqA.png)
