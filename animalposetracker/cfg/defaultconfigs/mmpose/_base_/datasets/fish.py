#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：mmpose -> fish
@Author ：Wu X.
@Email  : wux024@nenu.edu.cn
@Date   ：2024/4/6 14:14
'''
dataset_info = dict(
    dataset_name='Fish ',
    paper_info=dict(
        author='Mathis, Alexander and Biasi, Thomas and '
        'Schneider, Steffen and '
        'Yuksekgonul, Mert and Rogers, Byron and '
        'Bethge, Matthias and '
        'Mathis, Mackenzie W',
        title='Pretraining boosts out-of-domain robustness '
        'for pose estimation',
        container='Proceedings of the IEEE/CVF Winter Conference on '
        'Applications of Computer Vision',
        year='2021',
        homepage='http://www.mackenziemathislab.org/horse10',
    ),
    keypoint_info={
        0:
        dict(name='tip', id=0, color=[255, 153, 255], type='upper', swap=''),
        1:
        dict(name='gill', id=1, color=[255, 153, 255], type='upper', swap=''),
        2:
        dict(
            name='peduncle',
            id=2,
            color=[255, 102, 255],
            type='upper',
            swap=''),
        3:
        dict(
            name='dorsal fin tip',
            id=3,
            color=[255, 102, 255],
            type='upper',
            swap=''),
        4:
        dict(
            name='caudal tip',
            id=4,
            color=[255, 102, 255],
            type='upper',
            swap=''),
    },
    skeleton_info={
        0:
        dict(link=('tip', 'gill'), id=0, color=[255, 153, 255]),
        1:
        dict(link=('gill', 'peduncle'), id=1, color=[255, 153, 255]),
        2:
        dict(link=('peduncle', 'dorsal fin tip'), id=2, color=[255, 153, 255]),
        3:
        dict(link=('dorsal fin tip', 'caudal tip'), id=3, color=[255, 153, 255]),
    },
    joint_weights=[1.] * 5,
    sigmas=[1.0 / 5.0] * 5)
