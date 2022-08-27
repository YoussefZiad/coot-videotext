"""
Train captioning with MART.

Originally published by https://github.com/jayleicn/recurrent-transformer under MIT license
Reworked by https://github.com/gingsi/coot-videotext under Apache 2 license
"""

import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader

from coot.configs_retrieval import ExperimentTypesConst
from mart import arguments_mart
from mart.configs_mart import MartConfig as Config
from mart.model import create_mart_model
from mart.recursive_caption_dataset import create_mart_datasets_and_loaders, prepare_batch_inputs
from mart.trainer_caption import MartTrainer
from nntrainer import arguments, utils
from nntrainer.utils_torch import set_seed
from nntrainer.utils_yaml import load_yaml_config_file
from CustomVideoDataset import CustomVideoDataset


EXP_TYPE = ExperimentTypesConst.CAPTION


def main():
    # ---------- Setup script arguments. ----------
    parser = utils.ArgParser(description=__doc__)
    arguments.add_default_args(parser)  # logging level etc.
    arguments.add_exp_identifier_args(parser)  # arguments to identify the experiment to run
    arguments.add_trainer_args(parser, dataset_path=False)  # general trainer arguments
    parser.add_argument("--preload", action="store_true", help="Preload everything.")  # feature preloading
    arguments_mart.add_mart_args(parser)  # some more paths for mart
    parser.add_argument("--load_model", type=str, default=None, help="Load model from file.")
    parser.add_argument("--print_model", action="store_true", help=f"Print model")
    args = parser.parse_args()

    # load repository config yaml file to dict
    exp_group, exp_name, config_file = arguments.setup_experiment_identifier_from_args(args, EXP_TYPE)
    config = load_yaml_config_file(config_file)

    # update experiment config given the script arguments
    config = arguments.update_config_from_args(config, args)
    config = arguments_mart.update_mart_config_from_args(config, args)

    # read experiment config dict
    cfg = Config(config)
    if args.print_config:
        print(cfg)

    # set seed
    verb = "Set seed"
    if cfg.random_seed is None:
        cfg.random_seed = np.random.randint(0, 2 ** 15, dtype=np.int32)
        verb = "Randomly generated seed"
    print(f"{verb} {cfg.random_seed} deterministic {cfg.cudnn_deterministic} "
          f"benchmark {cfg.cudnn_benchmark}")
    set_seed(cfg.random_seed, cudnn_deterministic=cfg.cudnn_deterministic, cudnn_benchmark=cfg.cudnn_benchmark)

    # create dataset
    train_set, val_set, train_loader, val_loader = create_mart_datasets_and_loaders(
        cfg, args.coot_feat_dir, args.annotations_dir, args.video_feature_dir)

    run_name = f"{args.run_name}{1}"

    # create model from config
    model = create_mart_model(cfg, len(train_set.word2idx), cache_dir=args.cache_dir)

    # print model for debug if requested
    if args.print_model:
        print(model)

    # always load best epoch during validation
    load_best = args.load_best or args.validate

    model.eval()

    trainer = MartTrainer(
        cfg, model, exp_group, exp_name, run_name, len(train_loader), log_dir=args.log_dir,
        log_level=args.log_level, logger=None, print_graph=args.print_graph, reset=args.reset, load_best=load_best,
        load_epoch=args.load_epoch, load_model=args.load_model, inference_only=args.validate,
        annotations_dir=args.annotations_dir)

    # "v___c8enCfzqw": {"duration": 172.8, "timestamps": [[0, 51.84], [24.19, 118.37], [72.58, 157.25], [132.19, 172.8]],
    #                   "sentences": [
    #                       "A lady in black jacket is posing for the camera, then a man with medium length hair is shown, the man touched her hair to check on it, then the girl blow dry her hair.",
    #                       " The lady sat at the center of the studio with her wet hair, she put a white cream on her hand then rub it on her hair.",
    #                       " The girl blow dry her hair with white blower, sectioned her hair, brushed her hair with roller brush while blow drying it at the same time, she roll the brush downwards and upwards.",
    #                       " She styled her hair by brushing and combing it to give more volume."]}

    filename = "provided_embeddings/yc2_100m_coot_val.h5"

    emb = h5py.File(filename)

    coot_feat = emb['vid_emb'][:3][:]

    all_inputs = train_set.clip_sentence_to_feature("__c8enCfzqw", [24.19, 118.37],  " The lady sat at the center of the studio "
                                                                            "with her wet hair, she put a white cream "
                                                                            "on her hand then rub it on her hair.",
                                           coot_feat, 1)

    batch = [[{'name':[all_inputs[0]['name']], 'input_tokens':[all_inputs[0]['input_tokens']],
              'input_ids':torch.tensor(np.array([all_inputs[0]['input_ids']])), 'input_labels':torch.tensor(np.array([all_inputs[0]['input_labels']])),
              'input_mask':torch.tensor(np.array([all_inputs[0]['input_mask']])), 'token_type_ids':torch.tensor(np.array([all_inputs[0]['token_type_ids']])),
              'video_feature':torch.tensor(np.array([all_inputs[0]['video_feature']]))}], [2,2], [{'name': "v___c8enCfzqw",
                     'timestamp' :[[0, 51.84], [24.19, 118.37],
                                  [72.58, 157.25], [132.19, 172.8]],
                     'gt_sentence':[
                        "A lady in black jacket is posing for the camera, then a man with medium length hair is shown, the man touched her hair to check on it, then the girl blow dry her hair.",
                        " The lady sat at the center of the studio with her wet hair, she put a white cream on her hand then rub it on her hair.",
                        " The girl blow dry her hair with white blower, sectioned her hair, brushed her hair with roller brush while blow drying it at the same time, she roll the brush downwards and upwards.",
                        " She styled her hair by brushing and combing it to give more volume."]}]]

    trainer.generate_text_external_source(batch, val_loader)


if __name__ == "__main__":
    main()
