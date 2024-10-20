import shutil
from pathlib import Path
from subprocess import check_call

from setuptools import setup

SERVICE_NAME = "yamusic-fs.service"


def install() -> None:
    install_system_service_path = Path.home().joinpath(
        f".config/systemd/user/{SERVICE_NAME}"
    )
    install_system_service_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        Path(f"contrib/{SERVICE_NAME}"),
        install_system_service_path,
    )
    check_call(["systemctl", "daemon-reload", "--user"])
    check_call(["systemctl", "enable", SERVICE_NAME, "--user"])


if __name__ == "__main__":
    setup()
    install()
