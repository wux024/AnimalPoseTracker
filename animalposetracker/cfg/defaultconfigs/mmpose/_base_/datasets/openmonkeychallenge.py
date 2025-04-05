#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：mmpose -> openmonkeychallenge
@Author ：Wu X.
@Email  : wux024@nenu.edu.cn
@Date   ：2024/4/6 14:15
'''
dataset_info = dict(
    dataset_name='OpenMonkeyChallenge',
    paper_info=dict(
        author='Yu, Hang and Xu, Yufei and Zhang, Jing and '
        'Zhao, Wei and Guan, Ziyu and Tao, Dacheng',
        title='OpenMonkeyChallenge: '
              'Dataset and Benchmark Challenges '
              'for Pose Estimation of Non-human Primates',
        container='International Journal of Computer Vision',
        year='2023',
        homepage='https://www-users.cse.umn.edu/~hspark/omc/index.html',
    ),
    keypoint_info={
        0:
        dict(
            name='R_Eye', id=0, color=[0, 255, 0], type='upper', swap='L_Eye'),
        1:
        dict(
            name='L_Eye',
            id=1,
            color=[255, 128, 0],
            type='upper',
            swap='R_Eye'),
        2:
        dict(name='Nose', id=2, color=[51, 153, 255], type='upper', swap=''),
        3:
        dict(name='Head', id=3, color=[51, 153, 255], type='upper', swap=''),
        4:
        dict(
            name='Neck',
            id=4,
            color=[51, 153, 255],
            type='lower',
            swap=''),
        5:
        dict(
            name='R_Shoulder',
            id=5,
            color=[51, 153, 255],
            type='upper',
            swap='L_Shoulder'),
        6:
        dict(
            name='R_Elbow',
            id=6,
            color=[51, 153, 255],
            type='upper',
            swap='L_Elbow'),
        7:
        dict(
            name='R_Wrist',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='L_Wrist'),
        8:
        dict(
            name='L_Shoulder',
            id=8,
            color=[0, 255, 0],
            type='upper',
            swap='R_Shoulder'),
        9:
        dict(
            name='L_Elbow',
            id=9,
            color=[255, 128, 0],
            type='upper',
            swap='R_Elbow'),
        10:
        dict(
            name='L_Wrist',
            id=10,
            color=[0, 255, 0],
            type='lower',
            swap='R_Wrist'),
        11:
        dict(
            name='Hip',
            id=11,
            color=[255, 128, 0],
            type='lower',
            swap=''),
        12:
        dict(
            name='R_Knee',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='L_Knee'),
        13:
        dict(
            name='R_Ankle',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='L_Ankle'),
        14:
        dict(
            name='L_Knee', id=14, color=[0, 255, 0], type='lower',
            swap='R_Knee'),
        15:
        dict(
            name='L_Ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='R_Ankle'),
        16:
        dict(
            name='Tail',
            id=16,
            color=[0, 255, 0],
            type='lower',
            swap=''),
    },
    skeleton_info={
        0: dict(link=('L_Eye', 'Nose'), id=0, color=[0, 0, 255]),
        1: dict(link=('R_Eye', 'Nose'), id=1, color=[0, 0, 255]),
        2: dict(link=('Head', 'Neck'), id=2, color=[0, 0, 255]),
        3: dict(link=('Neck', 'Neck'), id=3, color=[0, 255, 0]),
        4: dict(link=('Neck', 'R_Shoulder'), id=4, color=[0, 255, 0]),
        5: dict(link=('R_Shoulder', 'R_Elbow'), id=5, color=[0, 255, 255]),
        6: dict(link=('R_Elbow', 'R_Wrist'), id=6, color=[0, 255, 255]),
        7: dict(link=('Neck', 'L_Shoulder'), id=7, color=[0, 255, 255]),
        8: dict(link=('L_Shoulder', 'L_Elbow'), id=8, color=[6, 156, 250]),
        9: dict(link=('L_Elbow', 'L_Wrist'), id=9, color=[6, 156, 250]),
        10: dict(link=('Neck', 'Hip'), id=10, color=[6, 156, 250]),
        11: dict(link=('Hip', 'R_Knee'), id=11, color=[0, 255, 255]),
        12: dict(link=('R_Knee', 'R_Ankle'), id=12, color=[0, 255, 255]),
        13: dict(link=('Hip', 'L_Knee'), id=13, color=[0, 255, 255]),
        14: dict(link=('L_Knee', 'L_Ankle'), id=14, color=[6, 156, 250]),
        15: dict(link=('Hip', 'Tail'), id=15, color=[6, 156, 250]),
    },
    joint_weights=[1.0]*17,
    sigmas=[
        0.025, 0.025, 0.026, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079, 0.072,
        0.062, 0.107, 0.087, 0.089, 0.087, 0.089, 0.062
    ])