"""Train the SleepPPG-Net baseline.

    python src/model_trainers/train_sleep_ppg_net.py --config configs/sleepppgnet.yaml

The architecture is fixed, so the config only controls the data, the optimiser and
where the run is written.
"""

from model_trainers.trainer import build_arg_parser, load_config, run_training
from models.model_type import ModelType
from models.sleep_ppg_net import SleepPPGNet


def main():
    parser = build_arg_parser("Train the SleepPPG-Net baseline")
    args = parser.parse_args()
    config = load_config(args, parser)
    run_training(args, config, SleepPPGNet, ModelType.PPG_ONLY)


if __name__ == "__main__":
    main()
