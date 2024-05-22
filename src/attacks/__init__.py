from src.attacks.data_source import DataSource

from src.attacks.features_extraction.extractor import FeatureExtractor
from src.attacks.features_extraction.carlini import CarliniLossThresholdExtractor
from src.attacks.features_extraction.secmi import SecMIExtractor
from src.attacks.features_extraction.combination_attack import (
    CombinationAttackExtractor,
)
from src.attacks.features_extraction.autoencoder_rec import (
    AutoEncReconLossThresholdExtractor,
)
from src.attacks.features_extraction.ssl_attack import SSLAttackExtractor
from src.attacks.features_extraction.pia import PIAExtractor
from src.attacks.features_extraction.gradient_masking import GradientMaskingExtractor
from src.attacks.features_extraction.multiple_loss import MultipleLossExtractor
from src.attacks.features_extraction.noise_optim import NoiseOptimExtractor

from src.attacks.scores_computation.computer import ScoreComputer
from src.attacks.scores_computation.carlini import CarliniLossThresholdComputer
from src.attacks.scores_computation.secmi import SecMIStat, SecMINN
from src.attacks.scores_computation.combination_attack import CombinationAttackComputer
from src.attacks.scores_computation.autoencoder_rec import (
    AutoEncReconLossThresholdComputer,
)
from src.attacks.scores_computation.ssl_attack import SSLAttackComputer

from src.attacks.scores_computation.pia import PIAComputer, PIANComputer
from src.attacks.scores_computation.gradient_masking import GradientMaskingComputer
from src.attacks.scores_computation.multiple_loss import MultipleLossComputer
from src.attacks.scores_computation.noise_optim import NoiseOptimComputer

from typing import Dict


feature_extractors: Dict[str, FeatureExtractor] = {
    "carlini_lt": CarliniLossThresholdExtractor,
    "secmi_nn": SecMIExtractor,
    "secmi_stat": SecMIExtractor,
    "combination_attack": CombinationAttackExtractor,
    "autoencoder_rec": AutoEncReconLossThresholdExtractor,
    "ssl_attack": SSLAttackExtractor,
    "pia": PIAExtractor,
    "pian": PIAExtractor,
    "gradient_masking": GradientMaskingExtractor,
    "multiple_loss": MultipleLossExtractor,
    "noise_optim": NoiseOptimExtractor,
}

score_computers: Dict[str, ScoreComputer] = {
    "carlini_lt": CarliniLossThresholdComputer,
    "secmi_nn": SecMINN,
    "secmi_stat": SecMIStat,
    "combination_attack": CombinationAttackComputer,
    "autoencoder_rec": AutoEncReconLossThresholdComputer,
    "ssl_attack": SSLAttackComputer,
    "pia": PIAComputer,
    "pian": PIANComputer,
    "gradient_masking": GradientMaskingComputer,
    "multiple_loss": MultipleLossComputer,
    "noise_optim": NoiseOptimComputer,
}

from src.attacks.utils import load_data, get_datasets_clf
