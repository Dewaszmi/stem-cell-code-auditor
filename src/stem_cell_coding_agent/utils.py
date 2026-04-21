from pathlib import Path

# Source - https://stackoverflow.com/a/73564246
# Posted by miigotu, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-21, License - CC BY-SA 4.0


# Helper to check whether code is ran inside a Docker container
def is_docker():
    cgroup = Path("/proc/self/cgroup")
    return Path("/.dockerenv").is_file() or (cgroup.is_file() and "docker" in cgroup.read_text())
