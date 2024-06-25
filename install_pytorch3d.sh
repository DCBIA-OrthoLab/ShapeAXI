#!/bin/bash

# Command to determine PyTorch version string
PYTORCH_VERSION=$(python -c "import sys; import torch; pyt_version_str=torch.__version__.split('+')[0].replace('.', ''); version_str=''.join([f'py3{sys.version_info.minor}_cu', torch.version.cuda.replace('.', ''), f'_pyt{pyt_version_str}']); print(version_str)")


# Command to install PyTorch3D
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/${PYTORCH_VERSION}/download.html