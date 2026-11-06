"""Separate archive reproduction from stable full-precision evaluation."""
from contextlib import contextmanager
POLICIES={'encoder':False,'grasp':False,'tabr':False,'no_retrieval':False}
ARCHIVE_POLICIES=dict(POLICIES,encoder=True)
@contextmanager
def source_math(torch,source,archive=False):
    enabled=(ARCHIVE_POLICIES if archive else POLICIES)[source]
    old=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32=enabled
        torch.backends.cudnn.allow_tf32=enabled
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32=old[0]
        torch.backends.cudnn.allow_tf32=old[1]
