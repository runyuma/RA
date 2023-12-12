import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Training script.')

    # Add arguments
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--iters', type=int, default=10000, help='Number of iterations.')
    parser.add_argument('--save', type=str2bool, default=True, help='Whether to save the model.')
    parser.add_argument('--render', type=str2bool, default=False, help='Whether to render the environment.')

    # Parse arguments
    args = parser.parse_args()

    # Use the arguments (here we're just printing them)
    print(f"Seed: {args.seed}")
    print(f"Iterations: {args.iters}")
    print(f"Save: {args.save}")
    print(f"Render: {args.render}")
if __name__ == "__main__":
    main()