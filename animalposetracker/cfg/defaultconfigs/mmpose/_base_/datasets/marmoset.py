#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：mmpose -> marmoset
@Author ：Wu X.
@Email  : wux024@nenu.edu.cn
@Date   ：2024/4/6 14:15
'''
dataset_info = dict(
    dataset_name='Marmoset',
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
            dict(
                name='Front', id=0, color=[0, 255, 0], type='upper', swap=''),
        1:
            dict(
                name='Right',
                id=1,
                color=[255, 128, 0],
                type='upper',
                swap='Left'),
        2:
            dict(name='Middle', id=2, color=[51, 153, 255], type='upper', swap=''),
        3:
            dict(name='Left', id=3, color=[51, 153, 255], type='upper', swap='Right'),
        4:
            dict(
                name='FL1',
                id=4,
                color=[51, 153, 255],
                type='lower',
                swap='FR1'),
        5:
            dict(
                name='BL1',
                id=5,
                color=[51, 153, 255],
                type='upper',
                swap='BR1'),
        6:
            dict(
                name='FR1',
                id=6,
                color=[51, 153, 255],
                type='upper',
                swap='FL1'),
        7:
            dict(
                name='BR1',
                id=7,
                color=[0, 255, 0],
                type='upper',
                swap='BL1'),
        8:
            dict(
                name='BL2',
                id=8,
                color=[0, 255, 0],
                type='upper',
                swap='BR2'),
        9:
            dict(
                name='BR2',
                id=9,
                color=[255, 128, 0],
                type='upper',
                swap='BL2'),
        10:
            dict(
                name='FL2',
                id=10,
                color=[0, 255, 0],
                type='lower',
                swap='FR2'),
        11:
            dict(
                name='FR2',
                id=11,
                color=[255, 128, 0],
                type='lower',
                swap='FL2'),
        12:
            dict(
                name='Body1',
                id=12,
                color=[255, 128, 0],
                type='lower',
                swap=''),
        13:
            dict(
                name='Body2',
                id=13,
                color=[0, 255, 0],
                type='lower',
                swap=''),
        14:
            dict(
                name='Body3', id=14, color=[0, 255, 0], type='lower',swap='')
    },
    skeleton_info={
        0: dict(link=('Right', 'Middle'), id=0, color=[0, 0, 255]),
        1: dict(link=('Left', 'FL1'), id=1, color=[0, 0, 255]),
        2: dict(link=('Right', 'Left'), id=2, color=[0, 0, 255]),
        3: dict(link=('Left', 'Body2'), id=3, color=[0, 255, 0]),
        4: dict(link=('Body2', 'Body3'), id=4, color=[0, 255, 0]),
        5: dict(link=('Body3', 'BL1'), id=5, color=[0, 255, 255]),
        6: dict(link=('BL1', 'BL2'), id=6, color=[0, 255, 255]),
        7: dict(link=('BL2', 'FL2'), id=7, color=[0, 255, 255]),
        8: dict(link=('FR1', 'BR2'), id=8, color=[6, 156, 250]),
        9: dict(link=('BL1', 'FR1'), id=9, color=[6, 156, 250]),
        10: dict(link=('Body2', 'BR1'), id=10, color=[6, 156, 250]),
        11: dict(link=('BR1', 'Body1'), id=11, color=[0, 255, 255]),
        12: dict(link=('Body2', 'BL1'), id=12, color=[0, 255, 255]),
        13: dict(link=('BL1', 'FR2'), id=13, color=[0, 255, 255]),
    },
    joint_weights=[1.] * 15,
    sigmas=[1.0 / 15.0] * 15)