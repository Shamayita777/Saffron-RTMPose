dataset_info = dict(
    dataset_name='saffron',

    paper_info=dict(
        author='Shamayita Moitra',
        title='Saffron Cutting Point Dataset',
        year='2026'
    ),

    keypoint_info={
        0: dict(
            name='cut_point',
            id=0,
            color=[255, 0, 0],
            type='upper',
            swap=''
        ),
    },

    skeleton_info={},

    joint_weights=[1.0],

    sigmas=[0.025]
)