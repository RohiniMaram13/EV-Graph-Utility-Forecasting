from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'processed_v2'
RESULTS_DIR = ROOT / 'results_v2'
MODEL_DIR = ROOT / 'models_v2'

LOOKBACK = 168
HORIZON = 24
STEP = 24
TRAIN_DAYS = 250
VAL_DAYS = 50
TEST_DAYS = 65
TOTAL_DAYS = TRAIN_DAYS + VAL_DAYS + TEST_DAYS

HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 16
MAX_EPOCHS = 100
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4

PEAK_HOURS = list(range(17, 23))
COUPLING_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]
DEFAULT_SEEDS = [42, 52, 62]

SOCIO_FEATURES = ['median_income', 'commute_time', 'total_households', 'multi_car_houses']
ALL_STATIC_FEATURES = ['ev_count'] + SOCIO_FEATURES
