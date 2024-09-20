_base_ = [
    "../../../configs/_base_/datasets/kitti-mono3d.py",
    "../../../configs/_base_/default_runtime.py",
    "../../../configs/_base_/schedules/cyclic-20e.py",
]
backbone_norm_cfg = dict(type="LN", requires_grad=True)
custom_imports = dict(imports=["projects.MonoDETR.monodetr"])

randomness = dict(seed=1, deterministic=False, diff_rank_seed=False)
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False
)
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
metainfo = dict(classes=class_names)

input_modality = dict(use_camera=True)
model = dict(
    type="MonoDETR",
    num_classes=3,
    num_queries=50,
    num_feature_levels=4,
    depthaware_transformer=dict(
        type="DepthAwareTransformer",
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        two_stage=False,
        two_stage_num_proposals=50,
        group_num=11,
        use_dab=False,
        two_stage_dino=False,
    ),
    pos_embedding=dict(
        type="PositionalEncoding",
        hidden_dim=256,
        position_embedding="sine",
    ),
    depth_predictor=dict(
        type="DepthPredictor",
        num_depth_bins=80,
        depth_min=1e-3,
        depth_max=60.0,
        hidden_dim=256,
    ),
    backbone=dict(
        type="Backbone",
        name="resnet50",
        train_backbone=True,
        return_interm_layers=True,
        dilation=False,
    ),
    matcher=dict(
        type="HungarianMatcher",
        cost_class=2,
        cost_3dcenter=10,
        cost_bbox=5,
        cost_giou=2,
    ),
    loss=dict(
        type="SetCriterion",
        num_classes=3,
        focal_alpha=0.25,
    ),
)
# data_root = "data/nuscenes/"

# dataset_type = "NuScenesDataset"
# data_root = "/mnt/nas3/Data/nuScenes/v1.0-trainval/"

backend_args = None

# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + "nuscenes_dbinfos_train.pkl",
#     rate=1.0,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(
#             car=5,
#             truck=5,
#             bus=5,
#             trailer=5,
#             construction_vehicle=5,
#             traffic_cone=5,
#             barrier=5,
#             motorcycle=5,
#             bicycle=5,
#             pedestrian=5,
#         ),
#     ),
#     classes=class_names,
#     sample_groups=dict(
#         car=2,
#         truck=3,
#         construction_vehicle=7,
#         bus=4,
#         trailer=6,
#         barrier=2,
#         motorcycle=6,
#         bicycle=6,
#         pedestrian=2,
#         traffic_cone=2,
#     ),
#     points_loader=dict(
#         type="LoadPointsFromFile",
#         coord_type="LIDAR",
#         load_dim=5,
#         use_dim=[0, 1, 2, 3, 4],
#         backend_args=backend_args,
#     ),
#     backend_args=backend_args,
# )
# ida_aug_conf = {
#     "resize_lim": (0.47, 0.625),
#     "final_dim": (320, 800),
#     "bot_pct_lim": (0.0, 0.0),
#     "rot_lim": (0.0, 0.0),
#     "H": 900,
#     "W": 1600,
#     "rand_flip": True,
# }

# train_pipeline = [
#     dict(
#         type="LoadMultiViewImageFromFiles", to_float32=True, backend_args=backend_args
#     ),
#     dict(
#         type="LoadAnnotations3D",
#         with_bbox_3d=True,
#         with_label_3d=True,
#         with_attr_label=False,
#     ),
#     dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
#     dict(type="ObjectNameFilter", classes=class_names),
# dict(type="ResizeCropFlipImage", data_aug_conf=ida_aug_conf, training=True),
# dict(
#     type="GlobalRotScaleTransImage",
#     rot_range=[-0.3925, 0.3925],
#     translation_std=[0, 0, 0],
#     scale_ratio_range=[0.95, 1.05],
#     reverse_angle=False,
#     training=True,
# ),
#     dict(
#         type="Pack3DDetInputs",
#         keys=[
#             "img",
#             "gt_bboxes",
#             "gt_bboxes_labels",
#             "attr_labels",
#             "gt_bboxes_3d",
#             "gt_labels_3d",
#             "centers_2d",
#             "depths",
#         ],
#     ),
# ]
# test_pipeline = [
#     dict(
#         type="LoadMultiViewImageFromFiles", to_float32=True, backend_args=backend_args
#     ),
#     dict(type="ResizeCropFlipImage", data_aug_conf=ida_aug_conf, training=False),
#     dict(type="Pack3DDetInputs", keys=["img"]),
# ]

# train_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     dataset=dict(
#         type=dataset_type,
#         data_prefix=dict(
#             pts="samples/LIDAR_TOP",
#             CAM_FRONT="samples/CAM_FRONT",
#             CAM_FRONT_LEFT="samples/CAM_FRONT_LEFT",
#             CAM_FRONT_RIGHT="samples/CAM_FRONT_RIGHT",
#             CAM_BACK="samples/CAM_BACK",
#             CAM_BACK_RIGHT="samples/CAM_BACK_RIGHT",
#             CAM_BACK_LEFT="samples/CAM_BACK_LEFT",
#         ),
#         pipeline=train_pipeline,
#         box_type_3d="LiDAR",
#         metainfo=metainfo,
#         test_mode=False,
#         modality=input_modality,
#         use_valid_flag=True,
#         backend_args=backend_args,
#     ),
# )
# test_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         data_prefix=dict(
#             pts="samples/LIDAR_TOP",
#             CAM_FRONT="samples/CAM_FRONT",
#             CAM_FRONT_LEFT="samples/CAM_FRONT_LEFT",
#             CAM_FRONT_RIGHT="samples/CAM_FRONT_RIGHT",
#             CAM_BACK="samples/CAM_BACK",
#             CAM_BACK_RIGHT="samples/CAM_BACK_RIGHT",
#             CAM_BACK_LEFT="samples/CAM_BACK_LEFT",
#         ),
#         pipeline=test_pipeline,
#         box_type_3d="LiDAR",
#         metainfo=metainfo,
#         test_mode=True,
#         modality=input_modality,
#         use_valid_flag=True,
#         backend_args=backend_args,
#     )
# )
# val_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         data_prefix=dict(
#             pts="samples/LIDAR_TOP",
#             CAM_FRONT="samples/CAM_FRONT",
#             CAM_FRONT_LEFT="samples/CAM_FRONT_LEFT",
#             CAM_FRONT_RIGHT="samples/CAM_FRONT_RIGHT",
#             CAM_BACK="samples/CAM_BACK",
#             CAM_BACK_RIGHT="samples/CAM_BACK_RIGHT",
#             CAM_BACK_LEFT="samples/CAM_BACK_LEFT",
#         ),
#         pipeline=test_pipeline,
#         box_type_3d="LiDAR",
#         metainfo=metainfo,
#         test_mode=True,
#         modality=input_modality,
#         use_valid_flag=True,
#         backend_args=backend_args,
#     )
# )

# Different from original PETR:
# We don't use special lr for image_backbone
# This seems won't affect model performance
optim_wrapper = dict(
    # TODO Add Amp
    # type='AmpOptimWrapper',
    # loss_scale='dynamic',
    optimizer=dict(type="AdamW", lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
)

num_epochs = 24

param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, begin=0, end=500, by_epoch=False),
    dict(
        type="CosineAnnealingLR",
        # TODO Figure out what T_max
        T_max=num_epochs,
        by_epoch=True,
    ),
]

train_cfg = dict(max_epochs=num_epochs, val_interval=num_epochs)

find_unused_parameters = False

# pretrain_path can be found here:
# https://drive.google.com/file/d/1ABI5BoQCkCkP4B0pO5KBJ3Ni0tei0gZi/view
# load_from = "/mnt/d/fcos3d_vovnet_imgbackbone-remapped.pth"
# resume = False
