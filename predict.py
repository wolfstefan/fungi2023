import os
import os.path as osp
import subprocess
import sys
from pathlib import Path


# custom script arguments
CONFIG_PATH = 'models/swin_large_b12x6-fp16_fungi+val_res_384_cb_epochs_6.py'
CHECKPOINT_PATH = "models/swin_large_b12x6-fp16_fungi+val_res_384_cb_epochs_6_20230524-9582690d.pth"
SCORE_THRESHOLD = 0.2


def run_inference(input_csv, output_csv, data_root_path):
    """Load model and dataloader and run inference."""

    if not data_root_path.endswith('/'):
        data_root_path += '/'
    data_cfg_opts = [
        f'test_dataloader.dataset.data_root=',
        f'test_dataloader.dataset.ann_file={input_csv}',
        f'test_dataloader.dataset.data_prefix={data_root_path}']

    inference = subprocess.Popen([
        'python', '-m',
        'tools.test_generate_result_pre-consensus',
        CONFIG_PATH, CHECKPOINT_PATH,
        output_csv,
        '--threshold', str(SCORE_THRESHOLD),
        '--no-scores',
        '--cfg-options'] + data_cfg_opts)
    return_code = inference.wait()
    if return_code != 0:
        print(f'Inference crashed with exit code {return_code}')
        sys.exit(return_code)
    print(f'Written {output_csv}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        help="Path to a file with observation ids and image paths.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data-root-path",
        help="Path to a directory where images are stored.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-file",
        help="Path to a file where predict script will store predictions.",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    output_csv = os.path.basename(args.output_file)
    if not output_csv.endswith(".csv"):
        output_csv = output_csv + ".csv"
    run_inference(
        input_csv=args.input_file,
        output_csv=output_csv,
        data_root_path=args.data_root_path,
    )
