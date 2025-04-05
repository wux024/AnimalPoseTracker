#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：mmpose -> aniposemouse
@Author ：Wu X.
@Email  : wux024@nenu.edu.cn
@Date   ：2024/4/6 14:13
'''
dataset_info = dict(
    dataset_name='Anipose Mouse',
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
            name='l-base',
            id=0,
            color=(225, 0, 255),
            type='upper',
            swap='r-base'),
        1:
        dict(
            name='l-edge',
            id=1,
            color=[220, 20, 60],
            type='upper',
            swap='r-edge'),
        2:
        dict(
            name='l-middle',
            id=2,
            color=[0, 255, 255],
            type='upper',
            swap='r-middle'),
        3:
        dict(
            name='r-base',
            id=3,
            color=(0, 255, 42),
            type='upper',
            swap='l-base'),
        4:
        dict(
            name='r-edge',
            id=4,
            color=[221, 160, 221],
            type='upper',
            swap='l-edge'),
        5:
        dict(
            name='r-middle',
            id=5,
            color=[135, 206, 250],
            type='upper',
            swap='l-middle'),
    },
    skeleton_info={
        0:
        dict(link=('l-base', 'l-edge'), id=0, color=[220, 20, 60]),
        1:
        dict(link=('l-edge', 'l-middle'), id=1, color=[0, 255, 255]),
        2:
        dict(link=('r-base', 'r-edge'), id=2, color=[221, 160, 221]),
        3:
        dict(link=('r-edge', 'r-middle'), id=3, color=[135, 206, 250]),
    },
    joint_weights=[1.]*6,
    sigmas=[1.0/6.0]*6)