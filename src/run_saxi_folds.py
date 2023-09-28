import argparse
import subprocess

def display_options(parser):
    print("Available options:")
    for action in parser._actions:
        if isinstance(action, argparse._ArgumentGroup):
            continue
        metavar = action.dest.upper()
        help_text = action.help
        default = action.default
        print(f"--{action.dest}: {help_text} (default: {default})")

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Description of your script.')

    # Add the command-line options
    parser.add_argument('--csv', required=True, help='Path to the CSV file')
    parser.add_argument('--folds', type=int, default=None, help='Number of folds')
    parser.add_argument('--valid_split', type=float, default=None, help='Validation split ratio')
    parser.add_argument('--group_by', type=str, default=None, help='Column to group by')
    parser.add_argument('--nn', type=str, default=None, help='Neural network type')
    parser.add_argument('--surf_column', type=str, default=None, help='Surface column name')
    parser.add_argument('--class_column', type=str, default=None, help='Class column name')
    parser.add_argument('--compute_scale_factor', type=int, default=None, help='Compute scale factor')
    parser.add_argument('--mount_point', type=str, default=None, help='Mount point')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--base_encoder', type=str, default=None, help='Base encoder type')
    parser.add_argument('--base_encoder_params', type=str, default=None, help='Base encoder parameters')
    parser.add_argument('--hidden_dim', type=int, default=None, help='Hidden dimension')
    parser.add_argument('--radius', type=float, default=None, help='Radius')
    parser.add_argument('--subdivision_level', type=int, default=None, help='Subdivision level')
    parser.add_argument('--image_size', type=int, default=None, help='Image size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--patience', type=int, default=None, help='Patience')
    parser.add_argument('--log_every_n_steps', type=int, default=None, help='Log every n steps')
    parser.add_argument('--tb_dir', type=str, default=None, help='Tensorboard directory')
    parser.add_argument('--tb_name', type=str, default=None, help='Tensorboard name')
    parser.add_argument('--neptune_project', type=str, default=None, help='Neptune project')
    parser.add_argument('--neptune_tags', type=str, default=None, help='Neptune tags')
    parser.add_argument('--target_layer', type=str, default=None, help='Target layer')
    parser.add_argument('--fps', type=int, default=None, help='Frames per second')
    parser.add_argument('--out', type=str, default='output_dir/', help='Output directory')

    # Display available options
    display_options(parser)

    # Prompt the user to enter the path to the CSV file
    csv_path = input("\nEnter the CSV file path: ")

    # Print a message to instruct the user to enter the option values
    print("\nEnter option values (Press Enter to skip an option):")

    # Collect option values from the user
    for arg in parser._actions:
        if isinstance(arg, argparse._ArgumentGroup):
            continue
        if arg.dest == 'csv':
            continue
        value = input(f"{arg.dest} [{arg.default}]: ").strip()
        if value:
            setattr(args, arg.dest, value)

    # Build the command string based on the provided options and user input
    cmd = f"python src/saxi_folds.py --csv {csv_path} "

    # Append other options if they are provided
    for arg in parser._actions:
        if isinstance(arg, argparse._ArgumentGroup):
            continue
        if arg.dest == 'csv':
            continue
        value = getattr(args, arg.dest)
        if value is not None:
            cmd += f"--{arg.dest} {value} "

    # Run the command using subprocess
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    else:
        print("Command executed successfully.")

if __name__ == "__main__":
    main()

