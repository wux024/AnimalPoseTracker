dataset_info = dict(
    dataset_name='StanfordExtra',
    paper_info=dict(
        author=' Benjamin, Biggs and Oliver, Boyne and '
               'James, Charles and Andrew, Fitzgibbon and '
               'Roberto, Cipolla',
        title='Who Left the Dogs Out? '
              '3D Animal Reconstruction with Expectation Maximization in the Loop',
        container='European Conference on Computer Vision (ECCV)',
        year='2020',
        homepage='https://github.com/benjiebob/WLDO',
    ),
    keypoint_info={
        0:
        dict(
            name='FrontLeftLeg0',
            id=0,
            color=[0, 255, 0],
            type='lower',
            swap='FrontRightLeg0'),
        1:
        dict(
            name='FrontLeftLeg1',
            id=1,
            color=[0, 255, 0],
            type='lower',
            swap='FrontRightLeg1'),
        2:
        dict(
            name='FrontLeftLeg2',
            id=2,
            color=[0, 255, 0],
            type='lower',
            swap='FrontRightLeg2'),
        3:
        dict(
            name='FrontRightLeg0',
            id=3,
            color=[0, 255, 0],
            type='lower',
            swap='FrontLeftLeg0'),
        4:
        dict(
            name='FrontRightLeg1',
            id=4,
            color=[0, 255, 0],
            type='lower',
            swap='FrontLeftLeg0'),
        5:
        dict(
            name='FrontRightLeg2',
            id=5,
            color=[0, 255, 0],
            type='lower',
            swap='FrontLeftLeg2'),
        6:
        dict(
            name='BackLeftLeg0',
            id=6,
            color=[51, 153, 255],
            type='lower',
            swap='BackRightLeg0'),
        7:
        dict(
            name='BackLeftLeg1',
            id=7,
            color=[0, 255, 0],
            type='lower',
            swap='BackRightLeg1'),
        8:
        dict(
            name='BackLeftLeg2',
            id=8,
            color=[0, 255, 0],
            type='lower',
            swap='BackRightLeg2'),
        9:
        dict(
            name='BackRightLeg0',
            id=9,
            color=[0, 255, 0],
            type='lower',
            swap='BackLeftLeg0'),
        10:
        dict(
            name='BackRightLeg1',
            id=10,
            color=[0, 255, 0],
            type='lower',
            swap='BackLeftLeg1'),
        11:
        dict(
            name='BackRightLeg2',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='BackLeftLeg2'),
        12:
        dict(
            name='Tail0',
            id=12,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        13:
        dict(
            name='Tail1',
            id=13,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        14:
        dict(
            name='LeftEar0',
            id=14,
            color=[255, 128, 0],
            type='upper',
            swap='RightEar0'),
        15:
        dict(
            name='RightEar0',
            id=15,
            color=[255, 128, 0],
            type='upper',
            swap='LeftEar0'),
        16:
        dict(
            name='Spout0',
            id=16,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        17:
        dict(
            name='Spout1',
            id=17,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        18:
        dict(
            name='LeftEar1',
            id=18,
            color=[255, 128, 0],
            type='upper',
            swap='RightEar1'),
        19:
        dict(
            name='RightEar1',
            id=19,
            color=[255, 128, 0],
            type='upper',
            swap='LeftEar1')
    },
    skeleton_info={
        0: dict(link=('FrontLeftLeg0', 'FrontLeftLeg1'), id=0, color=[0, 255, 0]),
        1: dict(link=('FrontLeftLeg1', 'FrontLeftLeg2'), id=1, color=[0, 255, 0]),
        2: dict(link=('FrontRightLeg0', 'FrontRightLeg1'), id=2, color=[0, 255, 0]),
        3: dict(link=('FrontRightLeg1', 'FrontRightLeg2'), id=3, color=[0, 255, 0]),
        4: dict(link=('BackLeftLeg0', 'BackLeftLeg1'), id=4, color=[0, 255, 0]),
        5: dict(link=('BackLeftLeg1', 'BackLeftLeg2'), id=5, color=[0, 255, 0]),
        6: dict(link=('BackRightLeg0', 'BackRightLeg1'), id=6, color=[0, 255, 0]),
        7: dict(link=('BackRightLeg1', 'BackRightLeg2'), id=7, color=[0, 255, 0]),
        8: dict(link=('Tail0', 'Tail1'), id=8, color=[255, 128, 0]),
        9: dict(link=('LeftEar0', 'LeftEar1'), id=9, color=[255, 128, 0]),
        10: dict(link=('RightEar0', 'RightEar1'), id=10, color=[255, 128, 0]),
    },
    joint_weights=[1.0]*20,
    sigmas=[1.0 / 20.0] * 20
)
