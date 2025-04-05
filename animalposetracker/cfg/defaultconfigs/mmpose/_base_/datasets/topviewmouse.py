#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：mmpose -> topviewmouse.py
@Author ：Wu X.
@Email  : wux024@nenu.edu.cn
@Date   ：2024/6/25 12:00
'''

dataset_info = dict(
    dataset_name="Topview5K",
    paper_info=dict(
        author=' Shaokai Ye and Anastasiia Filippova and Jessy Lauer and '
                'Steffen Schneider and Maxime Vidal and Tian Qiu and '
                'Alexander Mathis and Mackenzie Weygandt Mathis',
        title='SuperAnimal pretrained pose estimation models for behavioral analysis',
        container='Nature Communications',
        year='2024',
        homepage='https://zenodo.org/records/10618947'
    ),
    keypoint_info = {
            0: {
                "name": "nose",
                "id": 0,
                "color": [
                    255.0,
                    0.0,
                    0.0
                ],
                "type": "upper",
                "swap": ""
            },
            1: {
                "name": "left_ear",
                "id": 1,
                "color": [
                    255.0,
                    57.9267406190483,
                    0.0
                ],
                "type": "upper",
                "swap": "right_ear"
            },
            2: {
                "name": "right_ear",
                "id": 2,
                "color": [
                    255.0,
                    115.8534812380966,
                    0.0
                ],
                "type": "upper",
                "swap": "left_ear"
            },
            3: {
                "name": "left_ear_tip",
                "id": 3,
                "color": [
                    255.0,
                    173.78022185714494,
                    0.0
                ],
                "type": "upper",
                "swap": "right_ear_tip"
            },
            4: {
                "name": "right_ear_tip",
                "id": 4,
                "color": [
                    255.0,
                    231.7069624761932,
                    0.0
                ],
                "type": "upper",
                "swap": "left_ear_tip"
            },
            5: {
                "name": "left_eye",
                "id": 5,
                "color": [
                    220.36629690475846,
                    255.0,
                    0.0
                ],
                "type": "upper",
                "swap": "right_eye"
            },
            6: {
                "name": "right_eye",
                "id": 6,
                "color": [
                    162.43955628571013,
                    255.0,
                    0.0
                ],
                "type": "upper",
                "swap": "left_eye"
            },
            7: {
                "name": "neck",
                "id": 7,
                "color": [
                    104.51281566666178,
                    255.0,
                    0.0
                ],
                "type": "upper",
                "swap": ""
            },
            8: {
                "name": "mid_back",
                "id": 8,
                "color": [
                    46.586075047613534,
                    255.0,
                    0.0
                ],
                "type": "upper",
                "swap": ""
            },
            9: {
                "name": "mouse_center",
                "id": 9,
                "color": [
                    1.5322781428550762,
                    255.0,
                    12.872943714289848
                ],
                "type": "upper",
                "swap": ""
            },
            10: {
                "name": "mid_backend",
                "id": 10,
                "color": [
                    0.0,
                    255.0,
                    69.26707021385477
                ],
                "type": "upper",
                "swap": ""
            },
            11: {
                "name": "mid_backend2",
                "id": 11,
                "color": [
                    0.0,
                    255.0,
                    127.19344589637149
                ],
                "type": "upper",
                "swap": ""
            },
            12: {
                "name": "mid_backend3",
                "id": 12,
                "color": [
                    0.0,
                    255.0,
                    185.11982157888798
                ],
                "type": "upper",
                "swap": ""
            },
            13: {
                "name": "tail_base",
                "id": 13,
                "color": [
                    0.0,
                    255.0,
                    243.0461972614046
                ],
                "type": "upper",
                "swap": ""
            },
            14: {
                "name": "tail1",
                "id": 14,
                "color": [
                    0.0,
                    209.02713742857958,
                    255.0
                ],
                "type": "upper",
                "swap": ""
            },
            15: {
                "name": "tail2",
                "id": 15,
                "color": [
                    0.0,
                    151.1003968095313,
                    255.0
                ],
                "type": "upper",
                "swap": ""
            },
            16: {
                "name": "tail3",
                "id": 16,
                "color": [
                    0.0,
                    93.17365619048316,
                    255.0
                ],
                "type": "upper",
                "swap": ""
            },
            17: {
                "name": "tail4",
                "id": 17,
                "color": [
                    0.0,
                    35.24691557143491,
                    255.0
                ],
                "type": "upper",
                "swap": ""
            },
            18: {
                "name": "tail5",
                "id": 18,
                "color": [
                    22.67982504761347,
                    0.0,
                    255.0
                ],
                "type": "upper",
                "swap": ""
            },
            19: {
                "name": "left_shoulder",
                "id": 19,
                "color": [
                    80.6065656666618,
                    0.0,
                    255.0
                ],
                "type": "upper",
                "swap": "right_shoulder"
            },
            20: {
                "name": "left_midside",
                "id": 20,
                "color": [
                    138.5333062857101,
                    0.0,
                    255.0
                ],
                "type": "upper",
                "swap": "right_midside"
            },
            21: {
                "name": "left_hip",
                "id": 21,
                "color": [
                    196.46004690475843,
                    0.0,
                    255.0
                ],
                "type": "upper",
                "swap": "right_hip"
            },
            22: {
                "name": "right_shoulder",
                "id": 22,
                "color": [
                    249.483095841269,
                    0.0,
                    250.09630831746205
                ],
                "type": "upper",
                "swap": "left_shoulder"
            },
            23: {
                "name": "right_midside",
                "id": 23,
                "color": [
                    255.0,
                    0.0,
                    197.68647185714468
                ],
                "type": "upper",
                "swap": "left_midside"
            },
            24: {
                "name": "right_hip",
                "id": 24,
                "color": [
                    255.0,
                    0.0,
                    139.7597312380966
                ],
                "type": "upper",
                "swap": "left_hip"
            },
            25: {
                "name": "tail_end",
                "id": 25,
                "color": [
                    255.0,
                    0.0,
                    81.8329906190483
                ],
                "type": "upper",
                "swap": ""
            },
            26: {
                "name": "head_midpoint",
                "id": 26,
                "color": [
                    255.0,
                    0.0,
                    23.90625
                ],
                "type": "upper",
                "swap": ""
            }
        },
        joint_weights=[1.0]*27,
        sigmas=[1.0/27.0]*27,
        skeleton_info={
            0: {
                "link": [
                    "nose",
                    "left_ear"
                ],
                "id": 0,
                "color": [
                    255.0,
                    0.0,
                    0.0
                ]
            },
            1: {
                "link": [
                    "nose",
                    "right_ear"
                ],
                "id": 1,
                "color": [
                    255.0,
                    0.0,
                    0.0
                ]
            },
            2: {
                "link": [
                    "left_ear",
                    "right_ear"
                ],
                "id": 2,
                "color": [
                    255.0,
                    57.9267406190483,
                    0.0
                ]
            },
            3: {
                "link": [
                    "left_ear",
                    "tail_base"
                ],
                "id": 3,
                "color": [
                    255.0,
                    57.9267406190483,
                    0.0
                ]
            },
            4: {
                "link": [
                    "right_ear",
                    "tail_base"
                ],
                "id": 4,
                "color": [
                    255.0,
                    115.8534812380966,
                    0.0
                ]
            }
        }
)