dataset_info = dict(
    dataset_name='AWA Pose',
    paper_info=dict(
        author='Prianka, Banik and Lin, Li and Xishuang, Dong',
        title='A Novel Dataset for '
              'Keypoint Detection of quadruped Animals from Images ',
        container='arXiv',
        year='2021',
        homepage='https://github.com/prinik/AwA-Pose',
    ),
    keypoint_info={
        0:
        dict(
            name='nose',
            id=0,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        1:
        dict(
            name='upper_jaw',
            id=1,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        2:
        dict(
            name='lower_jaw',
            id=2,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        3:
        dict(
            name='mouth_end_right',
            id=3,
            color=[0, 255, 0],
            type='upper',
            swap='mouth_end_left'),
        4:
        dict(
            name='mouth_end_left',
            id=4,
            color=[0, 255, 0],
            type='upper',
            swap='mouth_end_right'),
        5:
        dict(
            name='right_eye',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='left_eye'),
        6:
        dict(
            name='right_earbase',
            id=6,
            color=[51, 153, 255],
            type='upper',
            swap='left_earbase'),
        7:
        dict(
            name='right_earend',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='left_earend'),
        8:
        dict(
            name='right_antler_base',
            id=8,
            color=[0, 255, 0],
            type='upper',
            swap='left_antler_base'),
        9:
        dict(
            name='right_antler_end',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='left_antler_end'),
        10:
        dict(
            name='left_eye',
            id=10,
            color=[0, 255, 0],
            type='upper',
            swap='right_eye'),
        11:
        dict(
            name='left_earbase',
            id=11,
            color=[0, 255, 0],
            type='upper',
            swap='right_earbase'),
        12:
        dict(
            name='left_earend',
            id=12,
            color=[0, 255, 0],
            type='upper',
            swap='right_earend'),
        13:
        dict(
            name='left_antler_base',
            id=13,
            color=[0, 255, 0],
            type='upper',
            swap='right_antler_base'),
        14:
        dict(
            name='left_antler_end',
            id=14,
            color=[0, 255, 0],
            type='upper',
            swap='right_antler_end'),
        15:
        dict(
            name='neck_base',
            id=15,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        16:
        dict(
            name='neck_end',
            id=16,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        17:
        dict(
            name='throat_base',
            id=17,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        18:
        dict(
            name='throat_end',
            id=18,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        19:
        dict(
            name='back_base',
            id=19,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        20:
        dict(
            name='back_end',
            id=20,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        21:
        dict(
            name='back_middle',
            id=21,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        22:
        dict(
            name='tail_base',
            id=22,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        23:
        dict(
            name='tail_end',
            id=23,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        24:
        dict(
            name='front_left_thai',
            id=24,
            color=[0, 255, 0],
            type='lower',
            swap='front_right_thai'),
        25:
        dict(
            name='front_left_knee',
            id=25,
            color=[0, 255, 0],
            type='lower',
            swap='front_right_knee'),
        26:
        dict(
            name='front_left_paw',
            id=26,
            color=[0, 255, 0],
            type='lower',
            swap='front_right_paw'),
        27:
        dict(
            name='front_right_thai',
            id=27,
            color=[0, 255, 0],
            type='lower',
            swap='front_left_thai'),
        28:
        dict(
            name='front_right_paw',
            id=28,
            color=[0, 255, 0],
            type='lower',
            swap='front_left_paw'),
        29:
        dict(
            name='front_right_knee',
            id=29,
            color=[0, 255, 0],
            type='lower',
            swap='front_left_knee'),
        30:
        dict(
            name='back_left_knee',
            id=30,
            color=[0, 255, 0],
            type='lower',
            swap='back_right_knee'),
        31:
        dict(
            name='back_left_paw',
            id=31,
            color=[0, 255, 0],
            type='lower',
            swap='back_right_paw'),
        32:
        dict(
            name='back_left_thai',
            id=32,
            color=[0, 255, 0],
            type='lower',
            swap='back_right_thai'),
        33:
        dict(
            name='back_right_thai',
            id=33,
            color=[0, 255, 0],
            type='lower',
            swap='back_left_thai'),
        34:
        dict(
            name='back_right_paw',
            id=34,
            color=[0, 255, 0],
            type='lower',
            swap='back_left_paw'),
        35:
        dict(
            name='back_right_knee',
            id=35,
            color=[0, 255, 0],
            type='lower',
            swap='back_left_knee'),
        36:
        dict(
            name='belly_bottom',
            id=36,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        37:
        dict(
            name='body_middle_right',
            id=37,
            color=[0, 255, 0],
            type='lower',
            swap='body_middle_left'),
        38:
        dict(
            name='body_middle_left',
            id=38,
            color=[0, 255, 0],
            type='lower',
            swap='body_middle_right'),
    },
    skeleton_info={
        0: dict(link=('left_eye', 'right_eye'), id=0, color=[0, 255, 0]),
        1: dict(link=('left_eye', 'left_earbase'), id=1, color=[0, 255, 0]),
        2: dict(link=('right_eye', 'right_earbase'), id=2, color=[0, 255, 0]),
        3: dict(link=('right_earbase', 'right_earend'), id=3, color=[0, 255, 0]),
        4: dict(link=('left_earbase', 'left_earend'), id=4, color=[0, 255, 0]),
        5: dict(link=('nose', 'right_eye'), id=5, color=[0, 255, 0]),
        6: dict(link=('nose', 'left_eye'), id=6, color=[0, 255, 0]),
        7: dict(link=('upper_jaw', 'lower_jaw'), id=7, color=[0, 255, 0]),
        8: dict(link=('mouth_end_right', 'mouth_end_left'), id=8, color=[0, 255, 0]),
        9: dict(link=('right_earbase', 'neck_base'), id=9, color=[0, 255, 0]),
        10: dict(link=('left_earbase', 'neck_base'), id=10, color=[0, 255, 0]),
        11: dict(link=('neck_base', 'neck_end'), id=11, color=[0, 255, 0]),
        12: dict(link=('neck_end', 'back_base'), id=12, color=[0, 255, 0]),
        13: dict(link=('back_base', 'back_middle'), id=13, color=[0, 255, 0]),
        14: dict(link=('back_middle', 'back_end'), id=14, color=[0, 255, 0]),
        15: dict(link=('back_end', 'tail_base'), id=15, color=[0, 255, 0]),
        16: dict(link=('tail_base', 'tail_end'), id=16, color=[0, 255, 0]),
        17: dict(link=('lower_jaw', 'throat_base'), id=17, color=[0, 255, 0]),
        18: dict(link=('throat_base', 'throat_end'), id=18, color=[0, 255, 0]),
        19: dict(link=('front_left_paw', 'front_left_knee'), id=19, color=[0, 255, 0]),
        20: dict(link=('front_left_knee', 'front_left_thai'), id=20, color=[0, 255, 0]),
        21: dict(link=('front_right_paw', 'front_right_knee'), id=21, color=[0, 255, 0]),
        22: dict(link=('front_right_knee', 'front_right_thai'), id=22, color=[0, 255, 0]),
        23: dict(link=('back_left_paw', 'back_left_knee'), id=23, color=[0, 255, 0]),
        24: dict(link=('back_left_knee', 'back_left_thai'), id=24, color=[0, 255, 0]),
        25: dict(link=('back_right_paw', 'back_right_knee'), id=25, color=[0, 255, 0]),
        26: dict(link=('back_right_knee', 'back_right_thai'), id=26, color=[0, 255, 0]),
        27: dict(link=('belly_bottom', 'front_right_thai'), id=27, color=[0, 255, 0]),
        28: dict(link=('belly_bottom', 'front_left_thai'), id=28, color=[0, 255, 0]),
        29: dict(link=('belly_bottom', 'back_right_thai'), id=29, color=[0, 255, 0]),
        30: dict(link=('belly_bottom', 'back_left_thai'), id=30, color=[0, 255, 0]),
        31: dict(link=('body_middle_right', 'front_right_thai'), id=31, color=[0, 255, 0]),
        32: dict(link=('body_middle_right', 'back_right_thai'), id=32, color=[0, 255, 0]),
        33: dict(link=('body_middle_right', 'back_base'), id=33, color=[0, 255, 0]),
        34: dict(link=('body_middle_right', 'back_end'), id=34, color=[0, 255, 0]),
        35: dict(link=('body_middle_left', 'back_left_thai'), id=35, color=[0, 255, 0]),
        36: dict(link=('body_middle_left', 'front_left_thai'), id=36, color=[0, 255, 0]),
        37: dict(link=('body_middle_left', 'back_base'), id=37, color=[0, 255, 0]),
        38: dict(link=('body_middle_left', 'back_end'), id=38, color=[0, 255, 0]),
        39: dict(link=('throat_end', 'front_left_thai'), id=39, color=[0, 255, 0]),
        40: dict(link=('throat_end', 'front_right_thai'), id=40, color=[0, 255, 0]),
        41: dict(link=('right_earbase', 'right_antler_base'), id=41, color=[0, 255, 0]),
        42: dict(link=('right_antler_base', 'right_antler_end'), id=42, color=[0, 255, 0]),
        43: dict(link=('left_earbase', 'left_antler_base'), id=43, color=[0, 255, 0]),
        44: dict(link=('left_antler_base', 'left_antler_end'), id=44, color=[0, 255, 0]),
    },
    joint_weights=[1.0]*39,
            sigmas=[
            0.026,
            0.067,
            0.067,
            0.067,
            0.067,
            0.025,
            0.067,
            0.067,
            0.067,
            0.067,
            0.025,
            0.067,
            0.067,
            0.067,
            0.067,
            0.035,
            0.067,
            0.035,
            0.067,
            0.067,
            0.035,
            0.067,
            0.035,
            0.067,
            0.079,
            0.072,
            0.062,
            0.079,
            0.072,
            0.062,
            0.089,
            0.107,
            0.107,
            0.087,
            0.087,
            0.089,
            0.067,
            0.067,
            0.067
        ],
)
