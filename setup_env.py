#!/usr/bin/env python3
"""
Deep Learning Specialization - Interactive Environment Setup
------------------------------------------------------------
This script helps you:
  1. Handle Git LFS (download large datasets / models)
  2. Extract all zipped datasets and models
  3. Create a virtual environment (venv or Conda)
  4. Install all required packages so every notebook runs without errors

Stable environment (do not change):
  - Python 3.12.x
  - packages from requirements.txt

Works on Windows, macOS, and Linux.
Supports both standard venv and Conda/Miniconda/Anaconda.
"""

import os
import sys
import subprocess
import platform
import shutil
import zipfile
import venv
from pathlib import Path


# Fixed stable Python version for this repository (do not change)
REQUIRED_PYTHON = (3, 12)
REQUIRED_PYTHON_STR = "3.12"


# ---------------------------- Helpers ---------------------------- #

def print_header():
    print("\n" + "=" * 70)
    print("  Deep Learning Specialization - Environment Setup")
    print("  DeepLearning.AI (Andrew Ng) · Coursera")
    print("=" * 70)
    print(f"  Stable environment: Python {REQUIRED_PYTHON_STR} + requirements.txt")
    print("  Also handles Git LFS + automatic zip extraction")
    print("=" * 70 + "\n")


def print_step(step: str):
    print(f"\n>>> {step}")


def ask_yes_no(question: str, default: bool = True) -> bool:
    """Ask a yes/no question and return True/False."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        answer = input(question + suffix).strip().lower()
        if answer == "":
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")


def run_command(cmd: list, check: bool = True, shell: bool = False, capture: bool = False) -> bool:
    """Run a command and return True if successful."""
    try:
        if capture:
            result = subprocess.run(cmd, check=check, shell=shell, capture_output=True, text=True)
            return result.returncode == 0
        else:
            subprocess.run(cmd, check=check, shell=shell)
            return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        print(f"Return code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n[ERROR] Command not found: {cmd[0] if isinstance(cmd, list) else cmd}")
        return False


def command_exists(cmd: str) -> bool:
    """Check if a command is available in PATH."""
    return shutil.which(cmd) is not None


def check_python_version():
    """Warn if the current Python is not 3.12.x."""
    current = sys.version_info[:2]
    if current != REQUIRED_PYTHON:
        print(f"[WARNING] You are running Python {sys.version.split()[0]}")
        print(f"          This repository is tested and stable ONLY with Python {REQUIRED_PYTHON_STR}.x")
        print(f"          It is strongly recommended to use Python {REQUIRED_PYTHON_STR}.")
        if not ask_yes_no("Continue anyway?", default=False):
            print("Exiting. Please run this script with Python 3.12.")
            sys.exit(1)
        print()


def get_python_executable(venv_path: Path) -> Path:
    """Return the path to the Python executable inside a venv."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def get_conda_python(env_name: str) -> str:
    """Return the python executable path for a conda environment."""
    try:
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c",
             "import sys; print(sys.executable)"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        conda_base = os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_EXE")
        if conda_base:
            if platform.system() == "Windows":
                return str(Path(conda_base).parent.parent / "envs" / env_name / "python.exe")
            return str(Path(conda_base).parent.parent / "envs" / env_name / "bin" / "python")
        return "python"


# ---------------------------- Git LFS ---------------------------- #

def handle_git_lfs(project_root: Path):
    """Ensure Git LFS is installed and pull all LFS objects."""
    print_step("Checking Git LFS (required for datasets, models, videos, zips)...")

    if not command_exists("git"):
        print("[WARNING] git is not available. Skipping Git LFS steps.")
        print("          Make sure you cloned the repository correctly.")
        return

    if not command_exists("git-lfs"):
        print("\n[IMPORTANT] Git LFS is NOT installed on your system.")
        print("Many large files in this repository (datasets .h5, models,")
        print(".zip archives, .mp4 videos, .csv, .npy, etc.) are stored with Git LFS.")
        print("Without Git LFS you will only get small pointer files and notebooks will fail.\n")
        print("Install Git LFS first:")
        print("  • Windows / macOS:  https://git-lfs.com  or  https://github.com/git-lfs/git-lfs#installing")
        print("  • Ubuntu/Debian:    sudo apt install git-lfs")
        print("  • Fedora:           sudo dnf install git-lfs")
        print("  • macOS (Homebrew): brew install git-lfs")
        print("\nAfter installing, run these commands in the repo root:")
        print("  git lfs install")
        print("  git lfs pull")
        print("Then re-run this setup script.\n")
        if not ask_yes_no("Continue without Git LFS? (not recommended)", default=False):
            sys.exit(1)
        return

    # Initialize LFS for this user if needed
    print("Git LFS found. Ensuring it is initialized...")
    run_command(["git", "lfs", "install"], check=False)

    # Pull all LFS objects
    print("Downloading all Git LFS files (this may take a while and use significant disk space)...")
    print("Files include: .h5 datasets, .zip archives, .mp4 videos, large .csv, .npy, etc.\n")

    success = run_command(["git", "lfs", "pull"], check=False)
    if success:
        print("[SUCCESS] Git LFS files downloaded.")
    else:
        print("[WARNING] git lfs pull reported issues.")
        print("You can try running it manually later:  git lfs pull")
        if not ask_yes_no("Continue anyway?", default=True):
            sys.exit(1)


# ---------------------------- Zip Extraction ---------------------------- #

def find_and_extract_zips(project_root: Path, delete_after: bool = True):
    """Find all .zip files, extract them next to the zip, then optionally delete the zip."""
    print_step("Looking for .zip archives to extract...")

    zip_files = list(project_root.rglob("*.zip"))
    # Filter out anything inside virtual environments or hidden folders
    zip_files = [
        z for z in zip_files
        if "venv" not in z.parts
        and ".git" not in z.parts
        and not any(p.startswith(".") for p in z.parts if p != z.name)
    ]

    if not zip_files:
        print("No .zip files found. (They may already have been extracted or LFS not pulled yet.)")
        return

    print(f"Found {len(zip_files)} zip archive(s):\n")
    for z in zip_files:
        rel = z.relative_to(project_root)
        print(f"  • {rel}")

    if not ask_yes_no("\nExtract all of them now?", default=True):
        print("Skipping zip extraction.")
        return

    extracted = 0
    failed = 0
    for zpath in zip_files:
        rel = zpath.relative_to(project_root)
        extract_dir = zpath.parent
        print(f"\nExtracting: {rel}")
        print(f"  → into:   {extract_dir.relative_to(project_root)}")

        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                # Safety: avoid extracting absolute paths or path traversal
                for member in zf.namelist():
                    member_path = Path(member)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        print(f"  [SKIP] Unsafe path in archive: {member}")
                        continue
                zf.extractall(path=extract_dir)
            print("  [OK] Extracted successfully.")
            extracted += 1

            if delete_after:
                try:
                    zpath.unlink()
                    print("  [OK] Zip file deleted after extraction.")
                except Exception as e:
                    print(f"  [WARNING] Could not delete zip: {e}")
        except zipfile.BadZipFile:
            print("  [ERROR] Not a valid zip file (maybe still an LFS pointer?).")
            print("          Run 'git lfs pull' first and try again.")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] Failed to extract: {e}")
            failed += 1

    print(f"\nExtraction finished: {extracted} succeeded, {failed} failed.")
    if failed > 0:
        print("Tip: Make sure Git LFS files were fully downloaded (git lfs pull).")


# ---------------------------- Setup Modes ---------------------------- #

def setup_venv(project_root: Path):
    """Create and configure a standard Python venv."""
    venv_path = project_root / "venv"

    if venv_path.exists():
        print(f"\nA virtual environment already exists at: {venv_path}")
        if ask_yes_no("Do you want to recreate it? (This will delete the existing one)", default=False):
            print_step("Removing existing virtual environment...")
            shutil.rmtree(venv_path)
        else:
            print("Using the existing virtual environment.")

    if not venv_path.exists():
        print_step("Creating virtual environment (venv)...")
        try:
            venv.create(venv_path, with_pip=True)
            print("Virtual environment created successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to create virtual environment: {e}")
            sys.exit(1)

    python_exe = str(get_python_executable(venv_path))

    print("\n" + "-" * 55)
    print("To activate this environment later, run:")
    if platform.system() == "Windows":
        print(f"  {venv_path}\\Scripts\\activate")
    else:
        print(f"  source {venv_path}/bin/activate")
    print("-" * 55)

    return python_exe, "venv", str(venv_path)


def setup_conda(project_root: Path):
    """Create and configure a Conda environment locked to Python 3.12."""
    if not command_exists("conda"):
        print("\n[ERROR] Conda was not found on your system.")
        print("Please install Miniconda or Anaconda first:")
        print("  https://docs.conda.io/en/latest/miniconda.html")
        print("Then restart your terminal and run this script again.")
        sys.exit(1)

    print("\nConda detected.")
    print(f"Note: Environment is locked to Python {REQUIRED_PYTHON_STR} (stable tested version).")

    default_name = "dl-specialization"
    env_name = input(f"Enter a name for the Conda environment [{default_name}]: ").strip() or default_name

    # Check if environment already exists
    result = subprocess.run(
        ["conda", "env", "list"], capture_output=True, text=True
    )
    env_exists = env_name in result.stdout

    if env_exists:
        print(f"\nA Conda environment named '{env_name}' already exists.")
        if ask_yes_no("Do you want to remove and recreate it?", default=False):
            print_step(f"Removing existing Conda environment '{env_name}'...")
            run_command(["conda", "env", "remove", "-n", env_name, "-y"])
            env_exists = False
        else:
            print(f"Using existing Conda environment '{env_name}'.")

    if not env_exists:
        print_step(f"Creating Conda environment '{env_name}' with Python {REQUIRED_PYTHON_STR}...")
        success = run_command([
            "conda", "create", "-n", env_name, f"python={REQUIRED_PYTHON_STR}", "-y"
        ])
        if not success:
            print("[ERROR] Failed to create Conda environment.")
            print(f"Make sure Python {REQUIRED_PYTHON_STR} is available in your Conda channels.")
            sys.exit(1)
        print(f"Conda environment '{env_name}' created successfully.")

    python_exe = get_conda_python(env_name)

    print("\n" + "-" * 55)
    print("To activate this environment later, run:")
    print(f"  conda activate {env_name}")
    print("-" * 55)

    return python_exe, "conda", env_name


def setup_current_env():
    """Use the currently running Python environment."""
    print("\nUsing the current Python environment (no new environment will be created).")
    return sys.executable, "current", None


# ---------------------------- Main Logic ---------------------------- #

def main():
    print_header()
    check_python_version()

    project_root = Path(__file__).resolve().parent
    requirements_file = project_root / "requirements.txt"

    if not requirements_file.exists():
        print("[ERROR] requirements.txt not found in the project root.")
        print("Make sure you are running this script from the repository root.")
        sys.exit(1)

    print(f"Detected OS      : {platform.system()} {platform.release()}")
    print(f"Python version  : {sys.version.split()[0]}")
    print(f"Project location: {project_root}")

    conda_available = command_exists("conda")
    print(f"Conda available : {'Yes' if conda_available else 'No'}")
    print()

    # -------------------- Git LFS -------------------- #
    handle_git_lfs(project_root)

    # -------------------- Zip extraction -------------------- #
    delete_zips = ask_yes_no(
        "\nAfter extracting zip files, delete the original .zip archives to save space?",
        default=True
    )
    find_and_extract_zips(project_root, delete_after=delete_zips)

    # -------------------- Choose environment type -------------------- #
    print("\nHow would you like to set up the Python environment?")
    print("  1. Create a new virtual environment (venv)     [recommended for most users]")
    if conda_available:
        print("  2. Create a new Conda environment             [recommended if you use Anaconda/Miniconda]")
        print("  3. Use the current Python environment         [no isolation]")
        print("  4. Exit")
        max_choice = 4
    else:
        print("  2. Use the current Python environment         [no isolation]")
        print("  3. Exit")
        print("\n  (Conda not detected. Install Miniconda/Anaconda to enable Conda option)")
        max_choice = 3

    choice = input(f"\nChoose an option [1-{max_choice}] (default: 1): ").strip() or "1"

    if (conda_available and choice == "4") or (not conda_available and choice == "3"):
        print("Exiting. No changes made.")
        sys.exit(0)

    if choice == "1":
        python_exe, env_type, env_id = setup_venv(project_root)
    elif choice == "2" and conda_available:
        python_exe, env_type, env_id = setup_conda(project_root)
    elif (choice == "2" and not conda_available) or (choice == "3" and conda_available):
        python_exe, env_type, env_id = setup_current_env()
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    # -------------------- Upgrade pip -------------------- #
    if ask_yes_no("\nDo you want to upgrade pip to the latest version?", default=True):
        print_step("Upgrading pip...")
        if env_type == "conda":
            cmd = ["conda", "run", "-n", env_id, "python", "-m", "pip", "install", "--upgrade", "pip"]
        else:
            cmd = [python_exe, "-m", "pip", "install", "--upgrade", "pip"]
        if not run_command(cmd):
            print("[WARNING] Failed to upgrade pip. Continuing anyway...")

    # -------------------- Install dependencies -------------------- #
    print("\nInstallation options:")
    print("  1. Full install (recommended) - installs everything from requirements.txt")
    print("  2. Skip installation")

    install_choice = input("\nChoose an option [1/2] (default: 1): ").strip() or "1"

    if install_choice == "1":
        print_step("Installing all dependencies from requirements.txt...")
        print("This may take several minutes depending on your internet speed.\n")
        print("Note: requirements.txt pins torch with CUDA 12.4. If you do not have a")
        print("compatible NVIDIA GPU, the install may fail. You can later install the")
        print("CPU-only build with:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n")

        if env_type == "conda":
            cmd = ["conda", "run", "-n", env_id, "python", "-m", "pip", "install", "-r", str(requirements_file)]
        else:
            cmd = [python_exe, "-m", "pip", "install", "-r", str(requirements_file)]

        success = run_command(cmd)

        if success:
            print("\n[SUCCESS] All packages installed successfully!")
        else:
            print("\n[ERROR] Some packages failed to install.")
            print("You can try running the command manually after activating the environment:")
            if env_type == "conda":
                print(f"  conda activate {env_id}")
            elif env_type == "venv":
                if platform.system() == "Windows":
                    print(f"  {env_id}\\Scripts\\activate")
                else:
                    print(f"  source {env_id}/bin/activate")
            print("  pip install -r requirements.txt")
            print("\nIf the failure was related to torch+CUDA, try the CPU wheels instead.")
            sys.exit(1)
    else:
        print("Skipping package installation.")

    # -------------------- Verify installation -------------------- #
    if ask_yes_no("\nDo you want to verify the core packages?", default=True):
        print_step("Verifying core packages...")
        verify_code = (
            "import numpy, pandas, sklearn, tensorflow, matplotlib, h5py; "
            "print('All core packages imported successfully!')"
        )

        if env_type == "conda":
            cmd = ["conda", "run", "-n", env_id, "python", "-c", verify_code]
        else:
            cmd = [python_exe, "-c", verify_code]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print("[WARNING] Some core packages could not be imported.")
            if result.stderr:
                print(result.stderr)

        # Optional torch check
        torch_code = (
            "import torch; "
            "print(f'PyTorch {torch.__version__} — CUDA available: {torch.cuda.is_available()}')"
        )
        if env_type == "conda":
            cmd = ["conda", "run", "-n", env_id, "python", "-c", torch_code]
        else:
            cmd = [python_exe, "-c", torch_code]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print("[INFO] PyTorch not available or failed to import "
                  "(this is OK if you only need TensorFlow).")

    # -------------------- Launch Jupyter -------------------- #
    if ask_yes_no("\nDo you want to launch Jupyter Lab now?", default=False):
        print_step("Launching Jupyter Lab...")
        print("Press Ctrl+C in the terminal to stop the server later.\n")

        try:
            if env_type == "conda":
                subprocess.run(["conda", "run", "-n", env_id, "jupyter", "lab"], check=False)
            else:
                subprocess.run([python_exe, "-m", "jupyter", "lab"], check=False)
        except KeyboardInterrupt:
            print("\nJupyter stopped.")
        except Exception as e:
            print(f"[WARNING] Could not launch Jupyter: {e}")
            print("You can start it manually after activating the environment:")
            print("  jupyter lab   or   jupyter notebook")

    # -------------------- Final message -------------------- #
    print("\n" + "=" * 70)
    print("  Setup finished!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Activate the environment (see instructions printed earlier).")
    print("  2. Open the course folders (C1 … C5) and run the notebooks.")
    print("  3. If any dataset/model is missing, re-run:  git lfs pull")
    print("  4. If a zip was not extracted, re-run this script or unzip manually.")
    print("\nHappy learning!\n")


if __name__ == "__main__":
    main()
