---
layout: post
title: "libfreenect Python Depth Image"
description: ""
category: 
tags: []
---
{% include JB/setup %}

## Viewing the Kinect Depth Reading Using Python

[libfreenect2](https://github.com/OpenKinect/libfreenect) is a useful tool for reading the output of a Kinect camera. It offers some sample programs that 
will display the output of the camera along with its depth readings. If you want to use the python bindings though, viewing the depth reading in the same
way as the example program is not possible and requires extra coding. In this post I will supply code that does this in a performant manner. Follow the 
instructions in the repo to install the library.

# Pure Python Implementation

This code is basically a python translation of the code here: https://github.com/OpenKinect/libfreenect/blob/master/examples/glview.c#L350-L402

The difference here is that we generate every depth value to a colour before we start. The code uses numpy and pygame:

```
import pygame
import numpy as np
import sys
from freenect import sync_get_depth as get_depth


def make_gamma():
    """
    Create a gamma table
    """
    num_pix = 2048 # there's 2048 different possible depth values
    npf = float(num_pix)
    _gamma = np.empty((num_pix, 3), dtype=np.uint16)
    for i in xrange(num_pix):
        v = i / npf
        v = pow(v, 3) * 6
        pval = int(v * 6 * 256)
        lb = pval & 0xff
        pval >>= 8
        if pval == 0:
            a = np.array([255, 255 - lb, 255 - lb], dtype=np.uint8)
        elif pval == 1:
            a = np.array([255, lb, 0], dtype=np.uint8)
        elif pval == 2:
            a = np.array([255 - lb, lb, 0], dtype=np.uint8)
        elif pval == 3:
            a = np.array([255 - lb, 255, 0], dtype=np.uint8)
        elif pval == 4:
            a = np.array([0, 255 - lb, 255], dtype=np.uint8)
        elif pval == 5:
            a = np.array([0, 0, 255 - lb], dtype=np.uint8)
        else:
            a = np.array([0, 0, 0], dtype=np.uint8)

        _gamma[i] = a
    return _gamma


gamma = make_gamma()


if __name__ == "__main__":
    fpsClock = pygame.time.Clock()
    FPS = 30 # kinect only outputs 30 fps
    disp_size = (640, 480)
    pygame.init()
    screen = pygame.display.set_mode(disp_size)
    font = pygame.font.Font('slkscr.ttf', 32) # provide your own font 
    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                sys.exit()
        fps_text = "FPS: {0:.2f}".format(fpsClock.get_fps())
        # draw the pixels

        depth = np.rot90(get_depth()[0])
        pixels = gamma[depth]
        temp_surface = pygame.Surface(disp_size)
        pygame.surfarray.blit_array(temp_surface, pixels)
        pygame.transform.scale(temp_surface, disp_size, screen)
        screen.blit(font.render(fps_text, 1, (255, 255, 255)), (30, 30))
        pygame.display.flip()
        fpsClock.tick(FPS)
```
