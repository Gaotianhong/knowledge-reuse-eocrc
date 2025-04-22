# Instance parameters
IMG_SIZE = 224

# Modality stacking order
MODE_ORDER = ['N', 'A', 'P']

# Number of slices above and below in the 3D scan
SLICE = 1

# CRC path
CRC_MATCH_PATH = 'data/crc_match.json'
CRC_INDEXES_PATH = 'data/crc_indexes.json'
CRC_BBOX_PATH = 'data/crc_bbox.json'

CRC_ALL_PATH = 'data/crc_all.json'
CRC_HARD_INDEXES_PATH = 'data/crc_hard_indexes.json'

# CRC LNM Path
CRC_LNM_MATCH_PATH = 'data/LNM/crc_lnm_match.json'
CRC_LNM_LABEL_PATH = 'data/LNM/crc_lnm_label.json'
CRC_LNM_BBOX_PATH = 'data/LNM/crc_lnm_bbox.json'

# Label list
LABEL_LIST = [0, 1]
# Label mapping 0: N0 1: N1 2: N2 3: normal
LABEL_MAP = {0: 0, 1: 1, 2: 1, 3: 2}