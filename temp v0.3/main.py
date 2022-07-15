import argparse
from config import main_config

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('--mode', default=main_config['mode'], type=str, help=None)
    args, unknown = parser.parse_known_args()

    if args.mode == 'playground':
        print('mode = playground')
        from playground import enterplayground
        enterplayground(parser)
    elif args.mode == 'training':
        from src.training import training_entry
        training_entry(parser)
    elif args.mode == 'finetuning':
        from src.finetuning import finetune_entry
        finetune_entry(parser)

    # elif args.mode == 'prep_reval':
    #     from src.eval import eval_entry
    #     eval_entry(parser)
    else:
        raise NotImplementedError('invalud mode')