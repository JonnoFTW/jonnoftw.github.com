---
layout: post
title: "Linux linker error: _ZTVN10__cxxabiv117__class_type_infoE"
description: ""
category: 
tags: [linux, python, linker error]
---
{% include JB/setup %}

If you ever get an error that looks like `_ZTVN10__cxxabiv117__class_type_infoE` or similar, the fix I found that works is to add `-lstdc++` to
your linker invocation. In python this means you need to modify the `setup.py` invocation of `Extension` to have:

```
libraries=['stdc++']
```

I hope this helps someone else.
