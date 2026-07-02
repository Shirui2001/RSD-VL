BASE_PATH = "/data/wenxinma"
DATA_PATH = {
    "Brain": f"{BASE_PATH}/data/MedAD/Brain_AD",
    "Liver": f"{BASE_PATH}/data/MedAD/Liver_AD",
    "Retina": f"{BASE_PATH}/data/MedAD/Retina_RESC_AD",
    "Colon_clinicDB": f"{BASE_PATH}/data/Colon/CVC-ClinicDB",
    "Colon_colonDB": f"{BASE_PATH}/data/Colon/CVC-ColonDB",
    "Colon_cvc300": f"{BASE_PATH}/data/Colon/CVC-300",
    "Colon_Kvasir": f"{BASE_PATH}/data/Colon/Kvasir",
    "BTAD": f"{BASE_PATH}/data/BTech_Dataset_transformed",
    "MPDD": f"{BASE_PATH}/data/MPDD",
    "MVTec": f"{BASE_PATH}/data/mvtec_ad",
    "VisA": f"{BASE_PATH}/data/VisA_20220922",
    # Explicit paths for road scenario (Cityscapes + COCO OOD + RoadAnomaly)
    "Road": "/wuhongwei/sr/AA-CLIP-main/dataset",
    "RoadSynth": "/wuhongwei/sr/AA-CLIP-main/dataset",
    "RoadAnomaly": "/wuhongwei/sr/AA-CLIP-main/dataset/Validation_Dataset/RoadAnomaly",
    "RoadAnomaly21": "/wuhongwei/sr/AA-CLIP-main/dataset/Validation_Dataset/RoadAnomaly21",
    "RoadObsticle21": "/wuhongwei/sr/AA-CLIP-main/dataset/Validation_Dataset/RoadObsticle21",
    "FS_LostFound_full": "/wuhongwei/sr/AA-CLIP-main/dataset/Validation_Dataset/FS_LostFound_full",
    "fs_static": "/wuhongwei/sr/AA-CLIP-main/dataset/Validation_Dataset/fs_static",
}

CLASS_NAMES = {
    "Brain": ["Brain"],
    "Liver": ["Liver"],
    "Retina": ["Retina"],
    "Colon_clinicDB": ["Colon_clinicDB"],
    "Colon_colonDB": ["Colon_colonDB"],
    "Colon_Kvasir": ["Kvasir"],
    "Colon_cvc300": ["CVC-300"],
    "MVTec": [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "transistor",
        "toothbrush",
        "wood",
        "zipper",
    ],
    "VisA": [
        "candle",
        "pcb3",
        "capsules",
        "pipe_fryum",
        "pcb4",
        "macaroni2",
        "pcb2",
        "chewinggum",
        "macaroni1",
        "cashew",
        "fryum",
        "pcb1",
    ],
    "MPDD": [
        "connector",
        "tubes",
        "metal_plate",
        "bracket_white",
        "bracket_brown",
        "bracket_black",
    ],
    "BTAD": ["01", "02", "03"],
    # Road/RoadSynth: two semantic classes, normal road and unknown obstacles
    "Road": ["road", "unknown"],
    "RoadSynth": ["road", "unknown"],
    "RoadAnomaly": ["road", "unknown"],
    "RoadAnomaly21": ["road", "unknown"],
    "RoadObsticle21": ["road", "unknown"],
    "FS_LostFound_full": ["road", "unknown"],
    "fs_static": ["road", "unknown"],
}
DOMAINS = {
    "VisA": "Industrial",
    "BTAD": "Industrial",
    "MPDD": "Industrial",
    "MVTec": "Industrial",
    "Brain": "Medical",
    "Liver": "Medical",
    "Retina": "Medical",
    "Colon_clinicDB": "Medical",
    "Colon_colonDB": "Medical",
    "Colon_Kvasir": "Medical",
    "Colon_cvc300": "Medical",
    "Road": "Road",
    "RoadSynth": "Road",
    "RoadAnomaly": "Road",
    "RoadAnomaly21": "Road",
    "RoadObsticle21": "Road",
    "FS_LostFound_full": "Road",
    "fs_static": "Road",
}
REAL_NAMES = {
    "Brain": {"Brain": "scan"},
    "Liver": {"Liver": "scan"},
    "Retina": {"Retina": "scan"},
    "MVTec": {
        "bottle": "dark bottle",
        "cable": "top view of three cables",
        "capsule": "black and orange capsule",
        "carpet": "gray carpet",
        "grid": "metal or plastic mesh",
        "hazelnut": "single brown hazelnut",
        "leather": "brown leather",
        "metal_nut": "metal nut which has four notched edges",
        "pill": "oval white pill with small red speckles and the letters 'FF' engraved",
        "screw": "screw",
        "tile": "speckled tile surface",
        "transistor": "a three-legged transistor placed vertically",
        "toothbrush": "toothbrush head",
        "wood": "wood surface",
        "zipper": "a black zipper",
    },
    "VisA": {
        "candle": "candle",
        "pcb3": "infrared sensor pcb module",
        "capsules": "capsules",
        "pipe_fryum": "pipe-shaped fryum",
        "pcb4": "battery charging pcb module",
        "macaroni2": "scattered yellow macaroni",
        "pcb2": "integrated circuits board",
        "chewinggum": "chewing gum",
        "macaroni1": "orange macaroni",
        "cashew": "cashew nut",
        "fryum": "wheel-shaped fryum snack",
        "pcb1": "dual ultrasonic distance sensor pcb module",
    },
    "Colon_clinicDB": {
        "Colon_clinicDB": "colon endoscopy image",
    },
    "Colon_colonDB": {
        "Colon_colonDB": "colon endoscopy image",
    },
    "Colon_cvc300": {"CVC-300": "colon endoscopy image"},
    "Colon_Kvasir": {"Kvasir": "colon endoscopy image"},
    "MPDD": {
        "connector": "metal clamps with black adjustment knobs",
        "tubes": "scattered metal objects",
        "metal_plate": "blue rectangular metal plate with a notch on one side",
        "bracket_white": "white, elongated triangular metal bracket with a smooth, matte finish",
        "bracket_brown": "brown L-shaped metal bracket with smooth, glossy finish and multiple mounting holes along its arms",
        "bracket_black": "black ornamental metal bracket with spiral design attached to a rectangular frame",
    },
    "BTAD": {
        "01": "Bright concentric rings in neon yellow and blue tones against a dark blue background, resembling a stylized wave or energy field radiating outward.",
        "02": "vertical fabric lines in warm, dusty pink and beige tones",
        "03": "oval concentric circular rings in gradient shades of blue and white",
    },
    "Road": {
        "road": "road surface in urban driving",
        "unknown": "a physical obstacle on the road",
    },
    "RoadSynth": {
        "road": "road surface in urban driving",
        "unknown": "a physical obstacle on the road",
    },
    "RoadAnomaly": {
        "road": "road surface in urban driving",
        "unknown": "a physical obstacle on the road",
    },
    "RoadAnomaly21": {
        "road": "road surface in urban driving",
        "unknown": "a physical obstacle on the road",
    },
    "RoadObsticle21": {
        "road": "road surface in urban driving",
        "unknown": "a physical obstacle on the road",
    },
    "FS_LostFound_full": {
        "road": "road surface in urban driving",
        "unknown": "a physical obstacle on the road",
    },
    "fs_static": {
        "road": "road surface in urban driving",
        "unknown": "a physical obstacle on the road",
    },
}

# Dataset-specific prompt overrides for road scenario
PROMPTS_BY_DATASET = {
    "Road": {
        "prompt_normal": [
            "A normal {}",
            "This is a typical {} on a road",
            "A standard view of a {} in an urban driving scene",
        ],
        "prompt_abnormal": [
            "An abnormal {}",
            "This is an unusual {} on a road",
        ],
        "prompt_templates": [
            "a photo of {}.",
            "a street scene with {}.",
            "a dashcam view of {}.",
        ],
    },
    "RoadSynth": {
        "prompt_normal": [
            "A normal {}",
            "This is a typical {} on a road",
            "A standard view of a {} in an urban driving scene",
        ],
        "prompt_abnormal": [
            "An abnormal {}",
            "This is an unusual {} on a road",
        ],
        "prompt_templates": [
            "{}.",
            "a photo of {}.",
        ],
    },
    "RoadAnomaly": {
        "prompt_normal": [
            "A normal {}",
            "This is a typical {} on a road",
            "A standard view of a {} in an urban driving scene",
        ],
        "prompt_abnormal": [
            "An abnormal {}",
            "This is an unusual {} on a road",
        ],
        "prompt_templates": [
            "{}.",
            "a photo of {}.",
        ],
    },
    "RoadAnomaly21": {
        "prompt_normal": [
            "A normal {}",
            "This is a typical {} on a road",
            "A standard view of a {} in an urban driving scene",
        ],
        "prompt_abnormal": [
            "An abnormal {}",
            "This is an unusual {} on a road",
        ],
        "prompt_templates": [
            "{}.",
            "a photo of {}.",
        ],
    },
    "RoadObsticle21": {
        "prompt_normal": [
            "A normal {}",
            "This is a typical {} on a road",
            "A standard view of a {} in an urban driving scene",
        ],
        "prompt_abnormal": [
            "An abnormal {}",
            "This is an unusual {} on a road",
        ],
        "prompt_templates": [
            "{}.",
            "a photo of {}.",
        ],
    },
    "FS_LostFound_full": {
        "prompt_normal": [
            "A normal {}",
            "This is a typical {} on a road",
            "A standard view of a {} in an urban driving scene",
        ],
        "prompt_abnormal": [
            "An abnormal {}",
            "This is an unusual {} on a road",
        ],
        "prompt_templates": [
            "{}.",
            "a photo of {}.",
        ],
    },
    "fs_static": {
        "prompt_normal": [
            "A normal {}",
            "This is a typical {} on a road",
            "A standard view of a {} in an urban driving scene",
        ],
        "prompt_abnormal": [
            "An abnormal {}",
            "This is an unusual {} on a road",
        ],
        "prompt_templates": [
            "{}.",
            "a photo of {}.",
        ],
    },
}
PROMPTS = {
    "prompt_normal": ["{}", "a {}", "the {}"],
    "prompt_abnormal": [
        "a damaged {}",
        "a broken {}",
        "a {} with flaw",
        "a {} with defect",
        "a {} with damage",
    ],
    "prompt_templates": [
        "{}.",
        "a photo of {}.",
    ],
}