#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：mmpose -> aniposefly
@Author ：Wu X.
@Email  : wux024@nenu.edu.cn
@Date   ：2024/4/6 14:13
'''
dataset_info = dict(
    dataset_name='Anipose Fly',
    paper_info=dict(
        author='Singapore University of Technology and Design, Singapore.'
        ' Xun Long Ng, Kian Eng Ong, Qichen Zheng,'
        ' Yun Ni, Si Yong Yeo, Jun Liu.',
        title='Anipose: A toolkit for robust markerless 3D pose estimation',
        container='Cell Reports',
        year='2021',
        homepage='https://github.com/lambdaloop/anipose',
        version='1.0 (2019-08)',
        date_created='2019-08',
    ),
    keypoint_info={
        0:
        dict(
            name='L1A',
            id=0,
            color=(225, 0, 255),
            type='upper',
            swap='R1A'),
        1:
        dict(
            name='L1B',
            id=1,
            color=[220, 20, 60],
            type='upper',
            swap='R1B'),
        2:
        dict(
            name='L1C',
            id=2,
            color=[0, 255, 255],
            type='upper',
            swap='R1C'),
        3:
        dict(
            name='L1D',
            id=3,
            color=(0, 255, 42),
            type='upper',
            swap='R1D'),
        4:
        dict(
            name='L1E',
            id=4,
            color=[221, 160, 221],
            type='upper',
            swap='R1E'),
        5:
        dict(
            name='L2A',
            id=5,
            color=[135, 206, 250],
            type='upper',
            swap='R2A'),
        6:
        dict(
            name='L2B',
            id=6,
            color=[50, 205, 50],
            type='upper',
            swap='R2B'),
        7:
        dict(
            name='L2C',
            id=7,
            color=[255, 182, 193],
            type='upper',
            swap='R2C'),
        8:
        dict(
            name='L2D',
            id=8,
            color=[0, 191, 255],
            type='upper',
            swap='R2D'),
        9:
        dict(
            name='L2E',
            id=9,
            color=[255, 105, 180],
            type='upper',
            swap='R2E'),
        10:
        dict(
            name='L3A',
            id=10,
            color=[30, 144, 255],
            type='upper',
            swap='R3A'),
        11:
        dict(
            name='L3B',
            id=11,
            color=[255, 20, 147],
            type='upper',
            swap='R3B'),
        12:
        dict(
            name='L3C',
            id=12,
            color=[0, 0, 255],
            type='upper',
            swap='R3C'),
        13:
        dict(
            name='L3D',
            id=13,
            color=(185, 3, 221),
            type='upper',
            swap='R3D'),
        14:
        dict(
            name='L3E',
            id=14,
            color=[255, 215, 0],
            type='upper',
            swap='R3E'),
        15:
        dict(
            name='R1A',
            id=15,
            color=[147, 112, 219],
            type='upper',
            swap='L1A'),
        16:
        dict(
            name='R1B',
            id=16,
            color=[255, 165, 0],
            type='upper',
            swap='L1B'),
        17:
        dict(
            name='R1C',
            id=17,
            color=[138, 43, 226],
            type='upper',
            swap='L1C'),
        18:
        dict(
            name='R1D',
            id=18,
            color=[255, 140, 0],
            type='upper',
            swap='L1D'),
        19:
        dict(
            name='R1E',
            id=19,
            color=[128, 0, 128],
            type='upper',
            swap='L1E'),
        20:
        dict(
            name='R2A',
            id=20,
            color=(0, 251, 255),
            type='upper',
            swap='L2A'),
        21:
        dict(
            name='R2B',
            id=21,
            color=[32, 178, 170],
            type='upper',
            swap='L2B'),
        22:
        dict(
            name='R2C',
            id=22,
            color=(0, 102, 102),
            type='upper',
            swap='L2C'),
        23:
        dict(
            name='R2D',
            id=23,
            color=(0, 102, 102),
            type='upper',
            swap='L2D'),
        24:
        dict(
            name='R2E',
            id=24,
            color=(0, 102, 102),
            type='upper',
            swap='L2E'),
        25:
        dict(
            name='R3A',
            id=25,
            color=(0, 102, 102),
            type='upper',
            swap='L3A'),
        26:
        dict(
            name='R3B',
            id=26,
            color=(0, 102, 102),
            type='upper',
            swap='L3B'),
        27:
        dict(
            name='R3C',
            id=27,
            color=(0, 102, 102),
            type='upper',
            swap='L3C'),
        28:
        dict(
            name='R3D',
            id=28,
            color=(0, 102, 102),
            type='upper',
            swap='L3D'),
        29:
        dict(
            name='R3E',
            id=29,
            color=(0, 102, 102),
            type='upper',
            swap='L3E'),
    },
    skeleton_info={
        0:
        dict(link=('L1A', 'L1B'), id=0, color=[220, 20, 60]),
        1:
        dict(link=('L1B', 'L1C'), id=1, color=[0, 255, 255]),
        2:
        dict(link=('L1C', 'L1D'), id=2, color=[221, 160, 221]),
        3:
        dict(link=('L1D', 'L1E'), id=3, color=[135, 206, 250]),
        4:
        dict(link=('L2A', 'L2B'), id=4, color=[221, 160, 221]),
        5:
        dict(link=('L2B', 'L2C'), id=5, color=[135, 206, 250]),
        6:
        dict(link=('L2C', 'L2D'), id=6, color=(225, 0, 255)),
        7:
        dict(link=('L2D', 'L2E'), id=7, color=(185, 3, 221)),
        8:
        dict(link=('L3A', 'L3B'), id=8, color=(0, 251, 255)),
        9:
        dict(link=('L3B', 'L3C'), id=9, color=[32, 178, 170]),
        10:
        dict(link=('L3C', 'L3D'), id=10, color=[255, 182, 193]),
        11:
        dict(link=('L3D', 'L3E'), id=11, color=[0, 191, 255]),
        12:
        dict(link=('R1A', 'R1B'), id=12, color=[255, 105, 180]),
        13:
        dict(link=('R1B', 'R1C'), id=13, color=[30, 144, 255]),
        14:
        dict(link=('R1C', 'R1D'), id=14, color=[255, 20, 147]),
        15:
        dict(link=('R1D', 'R1E'), id=15, color=[0, 0, 255]),
        16:
        dict(link=('R2A', 'R2B'), id=16, color=[255, 215, 0]),
        17:
        dict(link=('R2B', 'R2C'), id=17, color=[147, 112, 219]),
        18:
        dict(link=('R2C', 'R2D'), id=18, color=[255, 165, 0]),
        19:
        dict(link=('R2D', 'R2E'), id=19, color=[138, 43, 226]),
        20:
        dict(link=('R3A', 'R3B'), id=20, color=[255, 140, 0]),
        21:
        dict(link=('R3B', 'R3C'), id=21, color=[128, 0, 128]),
        22:
        dict(link=('R3C', 'R3D'), id=22, color=[128, 0, 128]),
        23:
        dict(link=('R3D', 'R3E'), id=23, color=[128, 0, 128])
    },
    joint_weights=[1.]*30,
    sigmas=[1.0/30.0]*30)
