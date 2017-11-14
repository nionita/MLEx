# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:56:32 2017

@author: nicu
"""

import math
import random
import numpy as np
import cv2

"""
Coordinates are screen-like, i.e. x=0, y=0 is top, left corner
"""

"""
This is a simple, deterministic sensor, which can detect
the presence of the object based on euclidian distance
and returns 1 if the object was detected or 0 otherwise
"""
class Sensor():
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def detect(self, x, y):
        dx = x - self.x
        dy = y - self.y
        r = math.sqrt(dx * dx + dy * dy)
        return 1 if r <= self.r else 0

"""
A room has 2 dimensions and a list of sensors
"""
class Room():

    MAXEXT = 800
    MINEXT = 400

    def __init__(self, x, y, s):
        self.x = x
        self.y = y
        self.sensors = s
        mi, ma = (x, y) if x > y else (y, x)
        if ma > Room.MAXEXT:
            self.scale = Room.MAXEXT / ma
        elif mi < Room.MINEXT:
            self.scale = Room.MINEXT / mi
        else:
            self.scale = 1.

    def in_range(self, x, y):
        return 0 <= x <= self.x and 0 <= y <= self.y

    def detect(self, x, y):
        detects = []
        for s in self.sensors:
            detects.append(s.detect(x, y))
        return detects

    def img_coord(self, x, y):
        kx = int(x * self.scale)
        ky = int(y * self.scale)
        return kx, ky

    def draw(self):
        kx, ky = self.img_coord(self.x, self.y)
        img = np.zeros((ky, kx, 3), dtype=np.uint8)
        for s in self.sensors:
            cx, cy = self.img_coord(s.x, s.y)
            cv2.circle(img, (cx, cy), 3, (100, 100, 100), -1)
        return img

    def draw_detects(self, img, det, ps):
        for s, a in zip(self.sensors, det):
            if a:
                cx, cy = self.img_coord(s.x, s.y)
                cv2.circle(img, (cx, cy), 3, (10, 255, 20), -1)
        for p in ps:
            cx, cy = self.img_coord(p.x, p.y)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

"""
A person will appear at one random side of the given room
and will go to another random side of that room with a random speed
"""
class Person():

    SPEED = 1
    SVAR = 0.15
    SMIN = 0.2

    def __init__(self, room):
        # Select a random start and end side
        se = random.sample(('N', 'S', 'E', 'W'), 2)
        sx, sy = self.randpos(se[0], room)
        self.x = sx
        self.y = sy
        ex, ey = self.randpos(se[1], room)
        dx = ex - sx
        dy = ey - sy
        speed = 0
        while speed < Person.SMIN:
            speed = random.gauss(Person.SPEED, Person.SVAR)
        d = math.sqrt(dx * dx + dy * dy)
        t = d / speed
        self.vx = dx / t
        self.vy = dy / t

    def step(self):
        self.x += self.vx
        self.y += self.vy

    def randpos(self, side, room):
        if side == 'N':
            y = 0
            x = random.uniform(0, room.x)
        elif side == 'S':
            y = room.y
            x = random.uniform(0, room.x)
        elif side == 'E':
            x = room.x
            y = random.uniform(0, room.y)
        else:
            x = 0
            y = random.uniform(0, room.y)
        return x, y

"""
Episodes with only one person in a room
"""
def episode(room):
    person = Person(room)
    #print('Start at', person.x, person.y)
    rimg = room.draw()
    while room.in_range(person.x, person.y):
        cimg = rimg.copy()
        det = room.detect(person.x, person.y)
        room.draw_detects(cimg, det, [person])
        cv2.imshow('room', cimg)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        person.step()

def rand_room(x, y, n, r):
    sensors = []
    for _ in range(n):
        s = Sensor(random.uniform(0, x), random.uniform(0, y), r)
        sensors.append(s)
    room = Room(x, y, sensors)
    return room

def grid_room(x, y, nx, ny, r):
    sensors = []
    dx = x / nx
    dy = y / ny
    for ix in range(nx):
        sx = dx / 2 + dx * ix
        for iy in range(ny):
            sy = dy / 2 + dy * iy
            s = Sensor(sx, sy, r)
            sensors.append(s)
    room = Room(x, y, sensors)
    return room

def comb_detect(det1, det2):
    l1 = len(det1)
    l2 = len(det2)
    if l1 > l2:
        return comb_detect_eq(det1[:l2], det2) + det1[l2:]
    elif l2 > l1:
        return comb_detect_eq(det1, det2[:l1]) + det2[l1:]
    else:
        return comb_detect_eq(det1, det2)

def comb_detect_eq(det1, det2):
    det = []
    for a, b in zip(det1, det2):
        det.append(max(a, b))
    return det

"""
Experiment with many persons traversing a square
"""
def square(room, time):
    rimg = room.draw()
    rest = np.random.poisson(lam=time)
    persons = []
    lp = 0
    while True:
        cimg = rimg.copy()
        rest -= 1
        if rest < 0:
            p = Person(room)
            persons.append(p)
            rest = np.random.poisson(lam=time)
        det = []
        rps = []
        for p in persons:
            det = comb_detect(det, room.detect(p.x, p.y))
            p.step()
            if room.in_range(p.x, p.y):
                rps.append(p)
        room.draw_detects(cimg, det, persons)
        cv2.imshow('room', cimg)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        persons = rps
        if len(persons) != lp:
            lp = len(persons)
            print(lp, 'persons')

if __name__ == '__main__':
#    #room = rand_room(100, 150, 100, 15)
    room = grid_room(300, 150, 20, 10, 25)
#    for i in range(10):
#        print('Episode', i)
#        episode(room)
    square(room, 50)