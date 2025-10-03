#!/usr/bin/env python3
"""
GEOG 288KC Documentation Build Script
This script builds the Quarto website and prepares it for GitHub Pages deployment
Supports incremental builds by default (only changed files) with --full option for complete rebuild
The --full flag also clears Quarto cache and freeze files to ensure a completely fresh build
"""

import os
import sys
import subprocess
import shutil
import argparse
import time
import glob
from pathlib import Path
import json

# Directories/files excluded by Quarto (_quarto.yml render settings)
# Keep this in sync with _quarto.yml 'render' excludes
EXCLUDED_DIR_SNIPPETS = [
    "nbs/",
    "installation/",
    "example_course/",
    "LLMs-from-scratch/",
]

def is_excluded_path(path_str: str) -> bool:
    """Return True if the given path string should be excluded from rendering."""
    normalized = path_str.replace("\\", "/")
    return any(snippet in normalized for snippet in EXCLUDED_DIR_SNIPPETS)

def is_buildable_file(path_str: str) -> bool:
    """Return True if file has a buildable extension and is not in an excluded path."""
    if not (path_str.endswith(".qmd") or path_str.endswith(".ipynb")):
        return False
    return not is_excluded_path(path_str)

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔨 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description} failed")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        sys.exit(1)

def get_changed_files():
    """Get list of .qmd and .ipynb files that have changed since the last commit."""
    try:
        # Check if we're in a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"], 
            check=True, capture_output=True, text=True
        )
        
        # Get files changed since the last commit
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"], 
            check=True, capture_output=True, text=True
        )
        
        changed_files = []
        for file_path in result.stdout.strip().split('\n'):
            if file_path and is_buildable_file(file_path):
                # Check if file still exists (wasn't deleted)
                if Path(file_path).exists():
                    changed_files.append(file_path)
        
        # Also check for staged changes
        result = subprocess.run(
            ["git", "diff", "--name-only", "--cached"], 
            check=True, capture_output=True, text=True
        )
        
        for file_path in result.stdout.strip().split('\n'):
            if file_path and is_buildable_file(file_path):
                if Path(file_path).exists() and file_path not in changed_files:
                    changed_files.append(file_path)
        
        # If no changes since last commit, check untracked files
        if not changed_files:
            result = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"], 
                check=True, capture_output=True, text=True
            )
            
            for file_path in result.stdout.strip().split('\n'):
                if file_path and is_buildable_file(file_path):
                    if Path(file_path).exists():
                        changed_files.append(file_path)
        
        return changed_files
        
    except subprocess.CalledProcessError:
        # Not a git repository or no commits yet - return all files
        print("⚠️  Not in a git repository or no previous commits found")
        print("   Falling back to full build")
        return None
    except Exception as e:
        print(f"⚠️  Error detecting changed files: {e}")
        print("   Falling back to full build")
        return None

def get_all_buildable_files():
    """Get all .qmd and .ipynb files that should be built according to _quarto.yml."""
    all_files = []
    
    # Get all .qmd files from current directory (respecting _quarto.yml excludes)
    current_path = Path(".")
    for qmd_file in current_path.rglob("*.qmd"):
        path_str = str(qmd_file)
        if is_excluded_path(path_str):
            continue
        all_files.append(path_str)
    
    # Get all .ipynb files from current directory (respecting _quarto.yml excludes)
    for ipynb_file in current_path.rglob("*.ipynb"):
        path_str = str(ipynb_file)
        if is_excluded_path(path_str):
            continue
        all_files.append(path_str)
    
    return all_files

def activate_conda_environment():
    """Activate the geoAI conda environment."""
    print("🐍 Activating geoAI environment...")
    
    # Get conda base path
    try:
        result = subprocess.run(["conda", "info", "--base"], 
                              check=True, capture_output=True, text=True)
        conda_base = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: Conda not found. Please install conda/miniconda/mambaforge")
        sys.exit(1)
    
    # Set up environment variables for conda
    conda_sh = os.path.join(conda_base, "etc", "profile.d", "conda.sh")
    if not os.path.exists(conda_sh):
        print(f"❌ Error: Conda script not found at {conda_sh}")
        sys.exit(1)
    
    # Update PATH to include the geoAI environment
    try:
        result = subprocess.run(["conda", "info", "--envs"], 
                              check=True, capture_output=True, text=True)
        
        # Find geoAI environment path
        env_path = None
        for line in result.stdout.split('\n'):
            if 'geoAI' in line:
                parts = line.split()
                if len(parts) >= 2:
                    env_path = parts[-1]  # Last part is the path
                    break
        
        if not env_path:
            print("❌ Error: geoAI environment not found")
            print("Please create the environment first:")
            print("   conda env list")
            sys.exit(1)
        
        # Add environment bin to PATH and guide Quarto to use this env's Python/Jupyter
        env_bin = os.path.join(env_path, "bin")
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{env_bin}:{current_path}"
        os.environ["CONDA_DEFAULT_ENV"] = "geoAI"
        os.environ["CONDA_PREFIX"] = env_path
        # Ensure Quarto uses the same interpreter/kernel discovery as this conda env
        os.environ["QUARTO_PYTHON"] = os.path.join(env_bin, "python")
        os.environ["QUARTO_JUPYTER"] = os.path.join(env_bin, "jupyter")
        # Also expose the repo root on PYTHONPATH so source imports resolve as a fallback
        repo_root = str(Path(__file__).resolve().parents[1])
        existing_pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f"{repo_root}:{existing_pp}" if existing_pp else repo_root
        
        print(f"   ✅ Environment activated: geoAI")
        
    except subprocess.CalledProcessError:
        print("❌ Error: Failed to get conda environment information")
        sys.exit(1)

def check_prerequisites():
    """Check if required tools are available."""
    print("🔍 Checking prerequisites...")
    
    # Check if quarto is installed
    try:
        subprocess.run(["quarto", "--version"], check=True, capture_output=True)
        print("   ✅ Quarto found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: Quarto is not installed or not in PATH")
        print("Please install Quarto from https://quarto.org/docs/get-started/")
        sys.exit(1)
    
    # Check if we're in the right directory
    if not Path("_quarto.yml").exists():
        print("❌ Error: _quarto.yml not found. Are you in the book directory?")
        sys.exit(1)
    print("   ✅ _quarto.yml found")

def check_jupyter_kernel(kernel_name: str = "geoai"):
    """Ensure the requested Jupyter kernel is available to Quarto/Jupyter.

    Exits with guidance if the kernel is not found.
    """
    try:
        result = subprocess.run(
            ["jupyter", "kernelspec", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout)
        kernels = data.get("kernelspecs", {})
        if kernel_name not in kernels:
            print(f"❌ Error: Jupyter kernel '{kernel_name}' not found.\n   Known kernels: {', '.join(sorted(kernels.keys())) or '(none)'}")
            print("   Tip: Register the kernel for this environment:")
            print(f"      python -m ipykernel install --user --name {kernel_name} --display-name \"{kernel_name}\"")
            print("   Or run: make kernelspec")
            sys.exit(1)
        else:
            print(f"   ✅ Jupyter kernel found: {kernel_name}")
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        print("⚠️  Warning: Could not verify Jupyter kernels. Proceeding, but Quarto may fail if the kernel is missing.")

def install_jupyter_kernel(kernel_name: str = "geoai", display_name: str | None = None, force: bool = True):
    """Install or update the Jupyter kernelspec to point at the current Python.

    This ensures the named kernel uses this env's interpreter, avoiding drift
    when Quarto spawns kernels by name.
    """
    display = display_name or kernel_name
    args = [
        "python", "-m", "ipykernel", "install",
        "--user", "--name", kernel_name, "--display-name", display,
    ]
    if force:
        args.append("--force")
    try:
        subprocess.run(args, check=True, capture_output=True, text=True)
        print(f"   ✅ Jupyter kernel installed/updated: {kernel_name}")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Failed to (re)install kernel '{kernel_name}': {e}")

def verify_kernel_binding(kernel_name: str, expected_python: str):
    """Inspect kernelspec argv and warn if it doesn't match expected interpreter."""
    try:
        result = subprocess.run(
            ["jupyter", "kernelspec", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout)
        ks = data.get("kernelspecs", {}).get(kernel_name)
        if not ks:
            return
        resource_dir = ks.get("resource_dir")
        kernel_json = Path(resource_dir) / "kernel.json"
        if not kernel_json.exists():
            return
        spec = json.loads(kernel_json.read_text())
        argv = spec.get("argv", [])
        actual_python = argv[0] if argv else ""
        if Path(actual_python).resolve() != Path(expected_python).resolve():
            print("⚠️  Warning: Kernelspec python does not match active env")
            print(f"   kernelspec: {actual_python}")
            print(f"   expected:   {expected_python}")
            print("   Tip: run 'make kernelspec' to rebind the kernel to this env.")
        else:
            print("   ✅ Kernelspec is bound to this environment's python")
    except Exception:
        # Non-fatal; used only for diagnostics
        pass

def ensure_editable_install():
    """Ensure the geogfm package is installed editable in the active env.

    Attempts an import; if it fails, runs `pip install -e ..` from the book dir.
    """
    try:
        subprocess.run(["python", "-c", "import geogfm"], check=True, capture_output=True)
        print("   ✅ geogfm importable in this environment")
        return
    except subprocess.CalledProcessError:
        print("🔧 Installing geogfm in editable mode (pip -e ..)...")
        try:
            # We are in book/; install package from repo root
            subprocess.run(["python", "-m", "pip", "install", "-e", ".."], check=True)
            print("   ✅ geogfm installed (editable)")
        except subprocess.CalledProcessError as e:
            print("❌ Error: Failed to install geogfm (editable)")
            print(f"   {e}")
            sys.exit(1)

def clean_docs():
    """Clean the docs directory and any stray HTML files."""
    print("🧹 Cleaning previous build...")
    
    # Remove any stray HTML files in current directory
    for html_file in Path(".").glob("*.html"):
        html_file.unlink()
    print("   Removed any stray HTML files from current directory")
    
    # Clean the docs directory
    docs_path = Path("../docs")
    if docs_path.exists():
        shutil.rmtree(docs_path)
        print("   Previous build files removed from ../docs/")
    
    # Ensure docs directory exists
    docs_path.mkdir(exist_ok=True)
    print("   Created ../docs/ directory")

def clear_quarto_cache():
    """Clear Quarto freeze files and caches for a fresh full build."""
    print("🗑️  Clearing Quarto cache and freeze files...")
    
    total_removed = 0
    cache_items_removed = []
    
    # Clear _freeze directory (contains cached execution results)
    freeze_path = Path("_freeze")
    if freeze_path.exists() and freeze_path.is_dir():
        shutil.rmtree(freeze_path)
        total_removed += 1
        cache_items_removed.append("_freeze/ (cached execution results)")
        print("   Removed _freeze/ directory")
    
    # Clear .quarto directory (contains Quarto cache)
    quarto_cache_path = Path(".quarto")
    if quarto_cache_path.exists() and quarto_cache_path.is_dir():
        shutil.rmtree(quarto_cache_path)
        total_removed += 1
        cache_items_removed.append(".quarto/ (Quarto cache)")
        print("   Removed .quarto/ directory")
    
    # Clear any .quarto directories in subdirectories
    for quarto_dir in Path(".").rglob(".quarto"):
        if quarto_dir.is_dir():
            # Skip if it's in excluded directories
            if ("nbs/" in str(quarto_dir) or "installation/" in str(quarto_dir)
                or "example_course/" in str(quarto_dir) or "LLMs-from-scratch/" in str(quarto_dir)):
                continue
            shutil.rmtree(quarto_dir)
            total_removed += 1
            cache_items_removed.append(f"{quarto_dir} (subdirectory cache)")
            print(f"   Removed {quarto_dir}")
    
    # Clear Jupyter checkpoints
    for checkpoint_dir in Path(".").rglob(".ipynb_checkpoints"):
        if checkpoint_dir.is_dir():
            # Skip if it's in excluded directories
            if ("nbs/" in str(checkpoint_dir) or "installation/" in str(checkpoint_dir)
                or "example_course/" in str(checkpoint_dir) or "LLMs-from-scratch/" in str(checkpoint_dir)):
                continue
            shutil.rmtree(checkpoint_dir)
            total_removed += 1
            cache_items_removed.append(f"{checkpoint_dir} (Jupyter checkpoints)")
            print(f"   Removed {checkpoint_dir}")
    
    if total_removed > 0:
        print(f"   ✅ Cleared {total_removed} cache/freeze items:")
        for item in cache_items_removed:
            print(f"      - {item}")
    else:
        print("   ✅ No cache or freeze files found to clear")
    
    print("   🔄 Fresh build environment prepared")

def clean_intermediate_files():
    """Remove all HTML and other intermediate files from chapters directories."""
    print("🧹 Cleaning intermediate files from chapters...")
    
    # Extensions to clean up
    extensions_to_remove = [
        "*.html",           # Rendered HTML files
        "*_files/",         # Quarto output directories
        "*.ipynb_checkpoints/",  # Jupyter checkpoints
        "*/.quarto/",       # Quarto cache directories
        "*.aux",            # LaTeX auxiliary files
        "*.log",            # Log files
        "*.out",            # LaTeX output files
        "*.toc",            # Table of contents files
        "*.nav",            # LaTeX navigation files
        "*.snm",            # LaTeX slide navigation files
        "*.fls",            # LaTeX file list
        "*.fdb_latexmk",    # LaTeX make files
        "*.synctex.gz",     # SyncTeX files
    ]
    
    chapters_path = Path("chapters")
    if not chapters_path.exists():
        print("   No chapters directory found")
        return
    
    total_removed = 0
    
    for extension in extensions_to_remove:
        if extension.endswith("/"):
            # Remove directories
            for item in chapters_path.rglob(extension):
                if item.is_dir():
                    shutil.rmtree(item)
                    total_removed += 1
                    print(f"   Removed directory: {item}")
        else:
            # Remove files
            for item in chapters_path.rglob(extension):
                if item.is_file():
                    item.unlink()
                    total_removed += 1
                    print(f"   Removed file: {item}")
    
    print(f"   ✅ Cleaned {total_removed} intermediate files/directories")

def build_site(files_to_build=None, full_build=False):
    """Build the Quarto website with detailed progress tracking."""
    import time
    
    # Determine which files to build
    if full_build or not files_to_build:
        print("📊 Analyzing files to build...")
        all_files = get_all_buildable_files()
        files_to_process = all_files
        build_type = "full build"
    else:
        files_to_process = files_to_build if files_to_build else []
        build_type = "incremental build"
    
    if not files_to_process:
        print("✅ No files to build - all files are up to date!")
        return
    
    total_files = len(files_to_process)
    print(f"🚀 Starting {build_type} ({total_files} file{' ' if total_files == 1 else 's'} to process)")
    print("")
    
    # Track timing
    overall_start_time = time.time()
    file_times = []
    
    # Build each file with detailed progress
    for i, file_path in enumerate(files_to_process, 1):
        file_start_time = time.time()
        
        # Calculate progress
        progress_pct = (i - 1) / total_files * 100
        remaining_files = total_files - (i - 1)
        
        # Estimate time remaining based on average file time
        if file_times:
            avg_time_per_file = sum(file_times) / len(file_times)
            estimated_remaining = avg_time_per_file * remaining_files
            eta_str = f" (ETA: {estimated_remaining:.1f}s)"
        else:
            eta_str = ""
        
        # Show progress header
        elapsed = time.time() - overall_start_time
        print(f"[{i:2d}/{total_files}] ({progress_pct:5.1f}%) Building: {file_path}")
        print(f"        ⏱️  Elapsed: {elapsed:.1f}s{eta_str}")
        
        # Build the file with a simpler progress indicator
        print(f"        🔨 Rendering... ", end="", flush=True)
        
        try:
            # Run the command directly (we're already in the book directory)
            result = subprocess.run(
                f"quarto render '{file_path}'", 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True
            )
            
            file_elapsed = time.time() - file_start_time
            file_times.append(file_elapsed)
            
            print(f"✅ Done ({file_elapsed:.1f}s)")
            
        except subprocess.CalledProcessError as e:
            file_elapsed = time.time() - file_start_time
            print(f"❌ Failed ({file_elapsed:.1f}s)")
            print(f"        Error: {e.stderr}")
            print(f"        Command: quarto render '{file_path}'")
            sys.exit(1)
        
        # Add spacing between files (except for the last one)
        if i < total_files:
            print("")
    
    # Final summary
    total_elapsed = time.time() - overall_start_time
    avg_time = total_elapsed / total_files if total_files > 0 else 0
    
    print("")
    print("📊 Build completed!")
    print(f"   ⏱️  Total time: {total_elapsed:.1f}s")
    print(f"   📈 Average per file: {avg_time:.1f}s")
    print(f"   📁 Files processed: {total_files}")
    
    if file_times:
        fastest = min(file_times)
        slowest = max(file_times)
        print(f"   ⚡ Fastest file: {fastest:.1f}s")
        print(f"   🐌 Slowest file: {slowest:.1f}s")

def verify_build():
    """Verify the build was successful."""
    docs_path = Path("../docs")
    
    if not docs_path.exists() or not any(docs_path.iterdir()):
        print("❌ Error: ../docs directory is empty or doesn't exist after build")
        sys.exit(1)
    
    print("✅ Build completed successfully!")
    print("📁 Documentation built in ../docs/ directory")
    
    # Count files and get size
    file_count = len(list(docs_path.rglob("*")))
    size_result = subprocess.run(["du", "-sh", "../docs"], capture_output=True, text=True)
    size = size_result.stdout.split()[0] if size_result.returncode == 0 else "unknown"
    
    print("📊 Build summary:")
    print(f"   - Files in ../docs/: {file_count}")
    print(f"   - Size: {size}")
    
    # Check critical files
    index_html = docs_path / "index.html"
    nojekyll = docs_path / ".nojekyll"
    
    if index_html.exists():
        print("   - ✅ index.html found")
    else:
        print("   - ⚠️  index.html not found")
    
    if nojekyll.exists():
        print("   - ✅ .nojekyll found (GitHub Pages compatibility)")
    else:
        print("   - ⚠️  .nojekyll not found")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build GEOG 288KC documentation with incremental build support"
    )
    parser.add_argument(
        "--serve", "-s",
        action="store_true",
        help="Start local server after building"
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Force full rebuild of all files and clear Quarto cache/freeze files (default: incremental build)"
    )
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Clean intermediate files (HTML, _files/, etc.) from chapters directories"
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Render foundational pages in order to tangle the library, then exit"
    )
    parser.add_argument(
        "--bootstrap-pages",
        nargs="*",
        help="Optional explicit list of .qmd pages to render in order (relative to book dir)"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Render only the specified file(s) (paths relative to the book directory)"
    )
    return parser.parse_args()

def serve_locally():
    """Start local server to preview the site."""
    print("🌐 Starting local server...")
    print("   Press Ctrl+C to stop the server when done")
    print("   Your site will open in your default browser")
    print("")
    
    try:
        # Run quarto preview directly (we're already in the book directory)
        subprocess.run(["quarto", "preview"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

def main():
    """Main build process."""
    args = parse_arguments()
    
    # Handle clean-only operation
    if args.clean and not any([args.full, args.serve]):
        # Check if this is a clean-only operation (no other build flags)
        print("🧹 Cleaning intermediate files...")
        
        # We still need to check prerequisites for conda environment
        activate_conda_environment()
        check_prerequisites()
        
        clean_intermediate_files()
        print("✅ Intermediate files cleaned successfully!")
        return

    # Handle bootstrap-only operation
    if args.bootstrap and not any([args.full, args.serve, args.clean]):
        activate_conda_environment()
        check_prerequisites()
        check_jupyter_kernel("geoai")
        # Refresh kernelspec to bind name -> current interpreter path
        install_jupyter_kernel("geoai")
        verify_kernel_binding("geoai", os.environ.get("QUARTO_PYTHON", sys.executable))
        ensure_editable_install()
        # Import local bootstrap to avoid global dependency
        from tools.bootstrap_lib import bootstrap
        bootstrap(Path("."), pages=args.bootstrap_pages)
        return
    
    if args.full:
        print("🚀 Starting GEOG 288KC documentation build (FULL BUILD)...")
    else:
        print("🚀 Starting GEOG 288KC documentation build (incremental)...")
    
    try:
        activate_conda_environment()
        check_prerequisites()
        check_jupyter_kernel("geoai")
        install_jupyter_kernel("geoai")
        verify_kernel_binding("geoai", os.environ.get("QUARTO_PYTHON", sys.executable))
        ensure_editable_install()
        
        # Clear Quarto caches and freeze files for full builds
        if args.full:
            clear_quarto_cache()
        
        files_to_build = None
        # Highest precedence: explicit list provided via --only
        if args.only:
            files_to_build = args.only
        elif not args.full:
            # Determine which files need to be built
            changed_files = get_changed_files()
            if changed_files is not None:
                files_to_build = changed_files
                if not files_to_build:
                    print("✅ No files have changed since last commit - skipping build")
                    print("   Use --full flag to force a complete rebuild or --only to render specific file(s)")
                    if args.clean:
                        clean_intermediate_files()
                    return
        
        if args.clean:
            clean_intermediate_files()
        
        # Only clean the site output directory for full builds; keep it for incremental or --only renders
        if args.full:
            clean_docs()
        # Always bootstrap foundational pages first on full builds to avoid ordering issues
        if args.full:
            from tools.bootstrap_lib import bootstrap
            bootstrap(Path("."), pages=args.bootstrap_pages)
        build_site(files_to_build, args.full)
        verify_build()
        
        print("")
        if args.full or (files_to_build is None):
            print("🎉 Your site is ready for deployment!")
        else:
            print(f"🎉 Successfully built {len(files_to_build)} changed file(s)!")
        
        if args.serve:
            serve_locally()
        else:
            print("📝 Next steps:")
            print("   1. Review the built site in the ../docs/ folder")
            print("   2. Commit and push changes to your repository")
            print("   3. Enable GitHub Pages from the docs/ folder in repository settings")
            print("")
            print("🌐 To preview locally, you can run:")
            print("   quarto preview")
            print("   Or use: make preview")
            if not args.full:
                print("")
                print("💡 Tip: Use make docs-full for complete rebuild when needed")
        
    except KeyboardInterrupt:
        print("\n❌ Build cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 