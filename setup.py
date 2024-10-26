import os
import shutil
from pathlib import Path
from subprocess import check_call, check_output

from setuptools import setup

SERVICE_NAME = "yamusic-fs.service"

IS_CI = os.getenv("CI") is not None


def get_tag() -> str:
    if IS_CI:
        if not Path("TAG").exists():
            return ""
        tag = Path("TAG").read_text().strip()
    else:
        tag = (
            check_output(["git", "tag", "-l", "--contains", "HEAD"])
            .strip()
            .decode()
        )
    return tag


def get_revisions() -> str:
    if IS_CI:
        if not Path("REVISIONS").exists():
            return ""
        version = Path("REVISIONS").read_text().strip()
    else:
        version = check_output(["git", "describe", "--tags"]).strip().decode()
    return version


def pre_setup() -> None:
    if Path("VERSION").exists():
        return
    tag = get_tag()
    if tag:
        version = tag.replace("v", "")
    else:
        revisions = get_revisions()
        if revisions:
            tag, count_commit, _ = revisions.split("-")
            major, minor, revision = tag.split(".")
            version = f"{major}.{minor}.dev{count_commit}"
        else:
            version = "0.0.1"
    Path("VERSION").write_text(version)


def install() -> None:
    if IS_CI:
        return
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
    pre_setup()
    setup()
    install()
