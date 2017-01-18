---
layout: post
title: "Markov Chain Image Generation"
description: "Generating Images Using a Markov Chain"
category: 
tags: [python, image generation]
---
{% include JB/setup %}

## Overview

In this post I will describe a method of generating images using a Markov Chain built from a training image. We train a markov chain to store pixel colours as the node values and the count of neighbouring pixel colours becomes the connection weight to neighbour nodes. To generate an image, we randomly walk through the chain and paint a pixel in the output image. The result is images that have a similar colour pallette to the original, but none of the coherence. They still look nice though.


### Description

At a high level, this algorithm works in the following manner:

1. Read in a training `image` into an array of pixels
2. Create a mapping `colourMap` from pixel colour to a pixel counter colour.
3. For each `pixel` in `image`:
  1. For each `neighbour` of `pixel`:
    1. `mapping[pixel.colour][neighbour.colour] += 1`
4. Create a blank `newImage`
5. Choose a random `startingPosition` within the bounds of `newImage`
6. Choose a random `startingState` from `mapping`
7. Create a `stack` with `[startingPosition]`
8. `image[startingPosition] = startingState`
9. while `stack` is not empty:
  1. `pixel = stack.pop()`
  2. For each `neighbour` of `pixel`:
   1. if `neighbour.isColoured`:
     2. continue
   1. `neighbour.isColoured = true`
   1. Push `neighbour` to `stack`
   2. `neighbour.colour = ` randomly chosen colour from `mapping[pixel.colour].keys()`. Probability distribution is `mapping[pixel.colour].counts() / sum(mapping[pixel.colour].counts())`
10. Display `newImage`


### Outputs

Here's some example outputs:

![HomerInput](http://i.imgur.com/ql46UZL.png)
![HomerGenerated](http://i.imgur.com/QDqCIF9.png)

An image generated from my black and white portrait:

![MeGenerated](http://i.imgur.com/RtyeYAJ.png)

![Moon](http://i.imgur.com/j49UuqV.jpg?1)
![MoonGenerated](http://i.imgur.com/0JiVWKh.png)

### Code
To run this you'll need python and the following packages installed:

```
pip install pillow numpy pyprind pygame
```
Feel free to mess around with `bucket_size` and `four_neighbour`. In my testing I found they didn't give much difference in ouput.

The final code is here:


```python
from PIL import Image
import numpy as np
import pyprind
import random
import os
import pygame
from collections import defaultdict, Counter


class MarkovChain(object):
    def __init__(self, bucket_size=10, four_neighbour=True):
        self.weights = defaultdict(Counter)
        self.bucket_size = bucket_size
        self.four_neighbour = four_neighbour

    def normalize(self, pixel):
        return pixel // self.bucket_size

    def denormalize(self, pixel):
        return pixel * self.bucket_size

    def get_neighbours(self, x, y):
        if self.four_neighbour:
            return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        else:
            return [(x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1),
                    (x + 1, y + 1),
                    (x - 1, y - 1),
                    (x - 1, y + 1),
                    (x + 1, y - 1)]

    def get_neighbours_dir(self, x, y):
        if self.four_neighbour:
            return {'r': (x + 1, y), 'l': (x - 1, y), 'b': (x, y + 1), 't': (x, y - 1)}
        else:
            return dict(zip(['r', 'l', 'b', 't', 'br', 'tl', 'tr', 'bl'],
                            [(x + 1, y),
                             (x - 1, y),
                             (x, y + 1),
                             (x, y - 1),
                             (x + 1, y + 1),
                             (x - 1, y - 1),
                             (x - 1, y + 1),
                             (x + 1, y - 1)]))

    def train(self, img):
        """
        Train on the input PIL image
        :param img:
        :return:
        """
        width, height = img.size
        img = np.array(img)[:, :, :3]
        prog = pyprind.ProgBar((width * height), width=64, stream=1)
        for x in range(height):
            for y in range(width):
                # get the left, right, top, bottom neighbour pixels
                pix = tuple(self.normalize(img[x, y]))
                prog.update()
                for neighbour in self.get_neighbours(x, y):
                    try:
                        self.weights[pix][tuple(self.normalize(img[neighbour]))] += 1
                    except IndexError:
                        continue
        self.directional = False

    def train_direction(self, img):
        self.weights = defaultdict(lambda: defaultdict(Counter))
        width, height = img.size
        img = np.array(img)[:, :, :3]
        prog = pyprind.ProgBar((width * height), width=64, stream=1)
        for x in range(height):
            for y in range(width):
                pix = tuple(self.normalize(img[x, y]))
                prog.update()
                for dir, neighbour in self.get_neighbours_dir(x, y).items():
                    try:
                        self.weights[pix][dir][tuple(self.normalize(img[neighbour]))] += 1
                    except IndexError:
                        continue
        self.directional = True

    def generate(self, initial_state=None, width=512, height=512):
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'MP4v')
        writer = cv2.VideoWriter('markov_img.mp4', fourcc, 24, (width, height))
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 0)
        pygame.init()

        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Markov Image')
        screen.fill((0, 0, 0))

        if initial_state is None:
            initial_state = random.choice(list(self.weights.keys()))
        if type(initial_state) is not tuple and len(initial_state) != 3:
            raise ValueError("Initial State must be a 3-tuple")
        img = Image.new('RGB', (width, height), 'white')
        img = np.array(img)
        img_out = np.array(img.copy())

        # start filling out the image
        # start at a random point on the image, set the neighbours and then move into a random, unchecked neighbour,
        # only filling in unmarked pixels
        initial_position = (np.random.randint(0, width), np.random.randint(0, height))
        img[initial_position] = initial_state
        stack = [initial_position]
        coloured = set()
        i = 0
        prog = pyprind.ProgBar((width * height), width=64, stream=1)
        # input()
        while stack:
            x, y = stack.pop()
            if (x, y) in coloured:
                continue
            else:
                coloured.add((x, y))
            try:
                cpixel = img[x, y]
                node = self.weights[tuple(cpixel)]  # a counter of neighbours
                img_out[x, y] = self.denormalize(cpixel)
                prog.update()
                i += 1
                screen.set_at((x, y), img_out[x, y])
                if i % 128 == 0:
                    pygame.display.flip()
                    # writer.write(cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))
                    pass
            except IndexError:
                continue

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sys.exit()


            if self.directional:
                keys = {dir: list(node[dir].keys()) for dir in node}
                neighbours = self.get_neighbours_dir(x, y).items()
                counts = {dir: np.array(list(node[dir].values()), dtype=np.float32) for dir in keys}
                key_idxs = {dir: np.arange(len(node[dir])) for dir in keys}
                ps = {dir: counts[dir] / counts[dir].sum() for dir in keys}
            else:
                keys = list(node.keys())
                neighbours = self.get_neighbours(x, y)
                counts = np.array(list(node.values()), dtype=np.float32)
                key_idxs = np.arange(len(keys))
                ps = counts / counts.sum()
            np.random.shuffle(neighbours)
            for neighbour in neighbours:
                try:
                    if self.directional:
                        direction = neighbour[0]
                        neighbour = neighbour[1]
                        if neighbour not in coloured:
                            col_idx = np.random.choice(key_idxs[direction], p=ps[direction])
                            img[neighbour] = keys[direction][col_idx]
                    else:
                        col_idx = np.random.choice(key_idxs, p=ps)
                        if neighbour not in coloured:
                            img[neighbour] = keys[col_idx]
                except IndexError:
                    pass
                except ValueError:
                    continue
                if 0 <= neighbour[0] < width and 0 <= neighbour[1] < height:
                    stack.append(neighbour)
        writer.release()
        return Image.fromarray(img_out)


if __name__ == "__main__":
    import sys

    chain = MarkovChain(bucket_size=16, four_neighbour=True)
    try:
        fnames = sys.argv[1:]
    except IndexError:
        fnames = ['me.jpg']
    for fname in fnames:
        im = Image.open(fname)
        # im.show()
        print("Training " + fname)
        chain.train(im)
    # print chain.weights
    print("\nGenerating")
    chain.generate().show()


```
