from .config import *

def enterplayground(parser):
    print('enterplayground()')

    parser.add_argument('--submode', default=speed_testing_config['submode'], type=str, help=None)
    args, unknown = parser.parse_known_args()

    if args.submode in ['dataonlyspeedtest','partialspeedtest','partialspeedtest_ViT']:
        from .speed_testing import speedtest
        speedtest(parser)
    else:
        raise NotImplementedError()
