cfg_all = {
    "w18":{
        'MODEL':{
            "EXTRA":{
                "STAGE1":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":1,
                    "BLOCK":"BOTTLENECK",
                    "NUM_BLOCKS":[4],
                    "NUM_CHANNELS":[64],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE2":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":2,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4],
                    "NUM_CHANNELS":[18,36],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE3":{
                    "NUM_MODULES":4,
                    "NUM_BRANCHES":3,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4,4],
                    "NUM_CHANNELS":[18,36,72],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE4":{
                    "NUM_MODULES":3,
                    "NUM_BRANCHES":4,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4,4,4],
                    "NUM_CHANNELS":[18,36,72,144],
                    "FUSE_METHOD":"SUM"
                }
            }
        },
        "pretrained":'./pretrained/hrnetv2_w18_imagenet_pretrained.pth'
    },
    "w18_small_v1":{
        'MODEL':{
            "EXTRA":{
                "STAGE1":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":1,
                    "BLOCK":"BOTTLENECK",
                    "NUM_BLOCKS":[1],
                    "NUM_CHANNELS":[32],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE2":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":2,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[2,2],
                    "NUM_CHANNELS":[16,32],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE3":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":3,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[2,2,2],
                    "NUM_CHANNELS":[16,32,64],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE4":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":4,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[2,3,3,3],
                    "NUM_CHANNELS":[16,32,64,128],
                    "FUSE_METHOD":"SUM"
                }
            }
        },
        "pretrained":'./pretrained/hrnet_w18_small_model_v1.pth'
    },
    "w18_small_v2":{
        'MODEL':{
            "EXTRA":{
                "STAGE1":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":1,
                    "BLOCK":"BOTTLENECK",
                    "NUM_BLOCKS":[2],
                    "NUM_CHANNELS":[64],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE2":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":2,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[2,2],
                    "NUM_CHANNELS":[18,36],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE3":{
                    "NUM_MODULES":3,
                    "NUM_BRANCHES":3,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[2,2,2],
                    "NUM_CHANNELS":[18,36,72],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE4":{
                    "NUM_MODULES":2,
                    "NUM_BRANCHES":4,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[2,2,2,2],
                    "NUM_CHANNELS":[18,36,72,144],
                    "FUSE_METHOD":"SUM"
                }
            }
        },
        "pretrained":'./pretrained/hrnet_w18_small_model_v2.pth'
    },
    "w30":{
        'MODEL':{
            "EXTRA":{
                "STAGE1":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":1,
                    "BLOCK":"BOTTLENECK",
                    "NUM_BLOCKS":[4],
                    "NUM_CHANNELS":[64],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE2":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":2,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4],
                    "NUM_CHANNELS":[30,60],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE3":{
                    "NUM_MODULES":4,
                    "NUM_BRANCHES":3,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4,4],
                    "NUM_CHANNELS":[30,60,120],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE4":{
                    "NUM_MODULES":3,
                    "NUM_BRANCHES":4,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4,4,4],
                    "NUM_CHANNELS":[30,60,120,240],
                    "FUSE_METHOD":"SUM"
                }
            }
        },
        "pretrained":'./pretrained/hrnetv2_w30_imagenet_pretrained.pth'
    },
    "w32":{
        'MODEL':{
            "EXTRA":{
                "STAGE1":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":1,
                    "BLOCK":"BOTTLENECK",
                    "NUM_BLOCKS":[4],
                    "NUM_CHANNELS":[64],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE2":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":2,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4],
                    "NUM_CHANNELS":[32,64],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE3":{
                    "NUM_MODULES":4,
                    "NUM_BRANCHES":3,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4,4],
                    "NUM_CHANNELS":[32,64,128],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE4":{
                    "NUM_MODULES":3,
                    "NUM_BRANCHES":4,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4,4,4],
                    "NUM_CHANNELS":[32,64,128,256],
                    "FUSE_METHOD":"SUM"
                }
            }
        },
        #"pretrained":'./pretrained/hrnetv2_w32_imagenet_pretrained.pth'
        "pretrained":'/data/models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth'
    },
    "w40":{
        'MODEL':{
            "EXTRA":{
                "STAGE1":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":1,
                    "BLOCK":"BOTTLENECK",
                    "NUM_BLOCKS":[4],
                    "NUM_CHANNELS":[64],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE2":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":2,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4],
                    "NUM_CHANNELS":[40,80],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE3":{
                    "NUM_MODULES":4,
                    "NUM_BRANCHES":3,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4,4],
                    "NUM_CHANNELS":[40,80,160],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE4":{
                    "NUM_MODULES":3,
                    "NUM_BRANCHES":4,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4,4,4],
                    "NUM_CHANNELS":[40,80,160,320],
                    "FUSE_METHOD":"SUM"
                }
            }
        },
        "pretrained":'./pretrained/hrnetv2_w40_imagenet_pretrained.pth'
    },
    "w44":{
        'MODEL':{
            "EXTRA":{
                "STAGE1":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":1,
                    "BLOCK":"BOTTLENECK",
                    "NUM_BLOCKS":[4],
                    "NUM_CHANNELS":[64],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE2":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":2,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4],
                    "NUM_CHANNELS":[44,88],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE3":{
                    "NUM_MODULES":4,
                    "NUM_BRANCHES":3,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4,4],
                    "NUM_CHANNELS":[44,88,176],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE4":{
                    "NUM_MODULES":3,
                    "NUM_BRANCHES":4,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4,4,4],
                    "NUM_CHANNELS":[44,88,176,352],
                    "FUSE_METHOD":"SUM"
                }
            }
        },
        "pretrained":'./pretrained/hrnetv2_w44_imagenet_pretrained.pth'
    },
    "w48":{
        'MODEL':{
            "EXTRA":{
                "STAGE1":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":1,
                    "BLOCK":"BOTTLENECK",
                    "NUM_BLOCKS":[4],
                    "NUM_CHANNELS":[64],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE2":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":2,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4],
                    "NUM_CHANNELS":[48,96],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE3":{
                    "NUM_MODULES":4,
                    "NUM_BRANCHES":3,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4,4],
                    "NUM_CHANNELS":[48,96,192],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE4":{
                    "NUM_MODULES":3,
                    "NUM_BRANCHES":4,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4,4,4],
                    "NUM_CHANNELS":[48,96,192,384],
                    "FUSE_METHOD":"SUM"
                }
            }
        },
        "pretrained":'./pretrained/hrnetv2_w48_imagenet_pretrained.pth'
    },
    "w64":{
        'MODEL':{
            "EXTRA":{
                "STAGE1":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":1,
                    "BLOCK":"BOTTLENECK",
                    "NUM_BLOCKS":[4],
                    "NUM_CHANNELS":[64],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE2":{
                    "NUM_MODULES":1,
                    "NUM_BRANCHES":2,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4],
                    "NUM_CHANNELS":[64,128],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE3":{
                    "NUM_MODULES":4,
                    "NUM_BRANCHES":3,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4,4],
                    "NUM_CHANNELS":[64,128,256],
                    "FUSE_METHOD":"SUM"
                },
                "STAGE4":{
                    "NUM_MODULES":3,
                    "NUM_BRANCHES":4,
                    "BLOCK":"BASIC",
                    "NUM_BLOCKS":[4,4,4,4],
                    "NUM_CHANNELS":[64,128,256,512],
                    "FUSE_METHOD":"SUM"
                }
            }
        },
        "pretrained":'./pretrained/hrnetv2_w64_imagenet_pretrained.pth'
    }
}
