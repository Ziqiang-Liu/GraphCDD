import argparse


def parameter_parser():

    parser = argparse.ArgumentParser(description="Run GACCDA.")

    parser.add_argument("--dataset-path",
                        nargs="?",
                        default="./data",
                        help="Training datasets.")

    parser.add_argument("--epoch",
                        type=int,
                        default=200,
                        help="Number of training epochs. Default is 651.")

    parser.add_argument("--gcn-layers",
                        type=int,
                        default=2,
                        help="Number of Graph Convolutional Layers. Default is 2.")

    parser.add_argument("--out-channels",
                        type=int,
                        default=256,
                        help="out-channels of cnn. Default is 128.")

    parser.add_argument("--circRNA-number",
                        type=int,
                        default=959,
                        help="circRNA number. Default is 585.")

    parser.add_argument("--fcir",
                        type=int,
                        default=128,
                        help="circRNA feature dimensions. Default is 256.")

    parser.add_argument("--disease-number",
                        type=int,
                        default=12,
                        help="disease number. Default is 88.")

    parser.add_argument("--fdis",
                        type=int,
                        default=128,
                        help="disease number. Default is 256.")
    parser.add_argument("--drug-number",
                        type=int,
                        default=18,
                        help="disease number. Default is 88.")

    parser.add_argument("--fdrug",
                        type=int,
                        default=128,
                        help="disease number. Default is 256.")




    parser.add_argument("--validation",
                        type=int,
                        default=5,
                        help="5 cross-validation.")


    return parser.parse_args()