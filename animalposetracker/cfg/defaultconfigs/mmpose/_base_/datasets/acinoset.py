#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：mmpose -> acinoset
@Author ：Wu X.
@Email  : wux024@nenu.edu.cn
@Date   ：2024/4/6 14:12
'''
dataset_info = dict(
    dataset_name='AcinoSet',
    paper_info=dict(
        author='Daniel Joska and Liam Clark '
               'and Naoya Muramatsu and Ricardo Jericevich '
               'and Fred Nicolls and Alexander Mathis '
               'and Mackenzie W. Mathis and Amir Patel',
        title='AcinoSet: '
              'A 3D Pose Estimation Dataset and Baseline Models '
              'for Cheetahs in the Wild',
        container='IEEE International '
                  'Conference on Robotics and Automation (ICRA)',
        year='2021',
        homepage='https://github.com/African-Robotics-Unit/AcinoSet',
        version='1.0 (2021-06)',
        date_created='2021-06',
    ),
    keypoint_info={
        0:
        dict(
            name='r_eye',
            id=0,
            color=(225, 0, 255),
            type='upper',
            swap='l_eye'),
        1:
        dict(
            name='l_eye',
            id=1,
            color=[220, 20, 60],
            type='upper',
            swap='r_eye'),
        2:
        dict(
            name='r_shoulder',
            id=2,
            color=[0, 255, 255],
            type='upper',
            swap='l_shoulder'),
        3:
        dict(
            name='r_front_knee',
            id=3,
            color=(0, 255, 42),
            type='lower',
            swap='l_front_knee'),
        4:
        dict(
            name='r_front_ankle',
            id=4,
            color=[221, 160, 221],
            type='lower',
            swap='l_front_ankle'),
        5:
        dict(
            name='r_front_paw',
            id=5,
            color=[135, 206, 250],
            type='lower',
            swap='l_front_paw'),
        6:
        dict(
            name='spine',
            id=6,
            color=[50, 205, 50],
            type='upper',
            swap=''),
        7:
        dict(
            name='r_hip',
            id=7,
            color=[255, 182, 193],
            type='upper',
            swap='l_hip'),
        8:
        dict(
            name='r_back_knee',
            id=8,
            color=[0, 191, 255],
            type='lower',
            swap='l_back_knee'),
        9:
        dict(
            name='r_back_ankle',
            id=9,
            color=[255, 105, 180],
            type='lower',
            swap='l_back_ankle'),
        10:
        dict(
            name='r_back_paw',
            id=10,
            color=[30, 144, 255],
            type='lower',
            swap='l_back_paw'),
        11:
        dict(
            name='tail1',
            id=11,
            color=[255, 20, 147],
            type='upper',
            swap=''),
        12:
        dict(
            name='tail2',
            id=12,
            color=[0, 0, 255],
            type='upper',
            swap=''),
        13:
        dict(
            name='l_shoulder',
            id=13,
            color=(185, 3, 221),
            type='upper',
            swap='r_shoulder'),
        14:
        dict(
            name='l_front_knee',
            id=14,
            color=[255, 215, 0],
            type='lower',
            swap='r_front_knee'),
        15:
        dict(
            name='l_front_ankle',
            id=15,
            color=[147, 112, 219],
            type='lower',
            swap='r_front_ankle'),
        16:
        dict(
            name='l_front_paw',
            id=16,
            color=[255, 165, 0],
            type='lower',
            swap='r_front_paw'),
        17:
        dict(
            name='l_hip',
            id=17,
            color=[138, 43, 226],
            type='upper',
            swap='r_hip'),
        18:
        dict(
            name='l_back_knee',
            id=18,
            color=[255, 140, 0],
            type='lower',
            swap='r_back_knee'),
        19:
        dict(
            name='l_back_ankle',
            id=19,
            color=[128, 0, 128],
            type='lower',
            swap='r_back_ankle'),
        20:
        dict(
            name='l_back_paw',
            id=20,
            color=(0, 251, 255),
            type='lower',
            swap='r_back_paw'),
        21:
        dict(
            name='lure',
            id=21,
            color=[32, 178, 170],
            type='upper',
            swap=''),
        22:
        dict(
            name='tail_base',
            id=22,
            color=(0, 102, 102),
            type='upper',
            swap=''),
        23:
        dict(
            name='nose',
            id=23,
            color=(0, 102, 102),
            type='upper',
            swap=''),
        24:
        dict(
            name='neck_base',
            id=24,
            color=(0, 102, 102),
            type='upper',
            swap='')
    },
    skeleton_info={
        0:
        dict(link=('nose', 'neck_base'), id=0, color=[220, 20, 60]),
        1:
        dict(link=('neck_base', 'spine'), id=1, color=[0, 255, 255]),
        2:
        dict(
            link=('spine', 'tail_base'),
            id=2,
            color=[221, 160, 221]),
        3:
        dict(
            link=('tail_base', 'tail1'),
            id=3,
            color=[135, 206, 250]),
        4:
        dict(
            link=('tail1', 'tail2'),
            id=4,
            color=[221, 160, 221]),
        5:
        dict(
            link=('nose', 'r_eye'),
            id=5,
            color=[135, 206, 250]),
        6:
        dict(
            link=('r_eye', 'neck_base'), id=6,
            color=(225, 0, 255)),
        7:
        dict(
            link=('nose', 'l_eye'),
            id=7,
            color=(185, 3, 221)),
        8:
        dict(
            link=('l_eye', 'neck_base'), id=8,
            color=(0, 251, 255)),
        9:
        dict(
            link=('neck_base', 'r_shoulder'),
            id=9,
            color=[32, 178, 170]),
        10:
        dict(
            link=('r_shoulder', 'r_front_knee'),
            id=10,
            color=[255, 182, 193]),
        11:
        dict(
            link=('r_front_knee', 'r_front_ankle'),
            id=11,
            color=[0, 191, 255]),
        12:
        dict(
            link=('neck_base', 'l_shoulder'), id=12, color=[255, 105,
                                                                180]),
        13:
        dict(
            link=('l_shoulder', 'l_front_knee'),
            id=13,
            color=[30, 144, 255]),
        14:
        dict(link=('l_front_knee', 'l_front_ankle'), id=14, color=[255, 20, 147]),
        15:
        dict(link=('r_shoulder', 'r_shoulder'), id=15, color=[0, 0, 255]),
        16:
        dict(link=('r_shoulder', 'spine'), id=16, color=[255, 215, 0]),
        17:
        dict(
            link=('spine', 'r_hip'), id=17, color=[147, 112, 219]),
        18:
        dict(link=('l_shoulder', 'spine'), id=18, color=[255, 165, 0]),
        19:
        dict(link=('spine', 'l_hip'), id=19, color=[138, 43, 226]),
        20:
        dict(link=('r_hip', 'l_hip'), id=20, color=[255, 140, 0]),
        21:
        dict(link=('tail_base', 'r_hip'), id=21, color=[128, 0, 128]),
        22:
        dict(link=('r_hip', 'r_back_knee'), id=22, color=[128, 0, 128]),
        23:
        dict(link=('r_back_knee', 'r_back_ankle'), id=23, color=[128, 0, 128]),
        24:
        dict(link=('tail_base', 'l_hip'), id=24, color=[128, 0, 128]),
        25:
        dict(link=('l_hip', 'l_back_knee'), id=25, color=[128, 0, 128]),
        26:
        dict(link=('l_back_knee', 'l_back_ankle'), id=26, color=[128, 0, 128])

    },
    joint_weights=[1.]*25,
    sigmas=[1.0/25.0]*25)
