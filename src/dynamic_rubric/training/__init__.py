"""Static-only training integration surfaces."""
from .verl_dataset import (
    build_rar_verl_rows,
    build_verl_rows,
    write_rar_verl_parquets,
    write_verl_parquets,
)

__all__ = [
    "build_rar_verl_rows",
    "build_verl_rows",
    "write_rar_verl_parquets",
    "write_verl_parquets",
]
