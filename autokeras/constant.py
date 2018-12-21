class Constant:
    # Data

    VALIDATION_SET_SIZE = 0.08333
    CUTOUT_HOLES = 1
    CUTOUT_RATIO = 0.5

    # Searcher

    MAX_MODEL_NUM = 1000
    BETA = 2.576
    KERNEL_LAMBDA = 1.0
    T_MIN = 0.0001
    N_NEIGHBOURS = 8
    MAX_MODEL_SIZE = (1 << 25)
    MAX_LAYER_WIDTH = 4096
    MAX_LAYERS = 200
    GRID_SEARCH = 'grid'
    BAYESIAN_SEARCH = 'bo'

    # Grid Dimensions
    LENGTH_DIM = 0
    WIDTH_DIM = 1

    # Default Search Space
    DEFAULT_LENGTH_SEARCH = [50,75,100]
    DEFAULT_WIDTH_SEARCH = [64,128,256]

    # Model Defaults

    DENSE_DROPOUT_RATE = 0.5
    CONV_DROPOUT_RATE = 0.25
    MLP_DROPOUT_RATE = 0.25
    CONV_BLOCK_DISTANCE = 2
    DENSE_BLOCK_DISTANCE = 1
    MODEL_LEN = 3
    MLP_MODEL_LEN = 3
    MLP_MODEL_WIDTH = 5
    MODEL_WIDTH = 64
    POOLING_KERNEL_SIZE = 2

    # ModelTrainer

    DATA_AUGMENTATION = True
    MAX_ITER_NUM = 200
    MIN_LOSS_DEC = 1e-4
    MAX_NO_IMPROVEMENT_NUM = 5
    MAX_BATCH_SIZE = 128
    LIMIT_MEMORY = False
    SEARCH_MAX_ITER = 200

    # text preprocessor

    EMBEDDING_DIM = 100
    MAX_SEQUENCE_LENGTH = 400
    MAX_NB_WORDS = 5000
    EXTRACT_PATH = "glove/"
    STORE_PATH = ''

    # Download file name

    FILE_PATH = "glove.zip"
    PRE_TRAIN_FILE_LINK = "http://nlp.stanford.edu/data/glove.6B.zip"
    PRE_TRAIN_FILE_NAME = "glove.6B.100d.txt"

    # constants for pretrained model of face detection
    FACE_DETECTION_PRETRAINED = {
        'PRETRAINED_MODEL_LINKS': [
            'https://raw.githubusercontent.com/kuaikuaikim/DFace/master/model_store/pnet_epoch.pt',
            'https://raw.githubusercontent.com/kuaikuaikim/DFace/master/model_store/rnet_epoch.pt',
            'https://raw.githubusercontent.com/kuaikuaikim/DFace/master/model_store/onet_epoch.pt'
        ],
        'FILE_PATHS': [
            'pnet.pt',
            'rnet.pt',
            'onet.pt'
        ]
    }

    # Image Resize

    MAX_IMAGE_SIZE = 128 * 128

    # SYS Constant

    SYS_LINUX = 'linux'
    SYS_WINDOWS = 'windows'
    SYS_GOOGLE_COLAB = 'goog_colab'
