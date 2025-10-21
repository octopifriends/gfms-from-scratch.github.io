#!/usr/bin/env python3
"""
GDAL/PROJ Diagnostic Tool for HPC Systems

This script helps diagnose and fix GDAL/PROJ configuration issues,
particularly the "proj.db not found" error common on HPC systems.

Usage:
    python diagnose_gdal_hpc.py
    
Or with environment setup:
    python diagnose_gdal_hpc.py --setup
"""

import os
import sys
from pathlib import Path
import argparse


def find_conda_prefix():
    """Find the conda environment prefix."""
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        return Path(conda_prefix)
    
    # Try alternative methods
    if sys.prefix:
        return Path(sys.prefix)
    
    return None


def find_proj_db(search_paths=None):
    """Find proj.db file in common locations."""
    if search_paths is None:
        conda_prefix = find_conda_prefix()
        search_paths = [
            os.environ.get('PROJ_LIB'),
            os.environ.get('PROJ_DATA'),
        ]
        
        if conda_prefix:
            search_paths.extend([
                conda_prefix / 'share' / 'proj',
                conda_prefix / 'Library' / 'share' / 'proj',
            ])
        
        search_paths.extend([
            Path('/usr/share/proj'),
            Path('/usr/local/share/proj'),
            Path.home() / 'mambaforge' / 'share' / 'proj',
            Path.home() / 'miniconda3' / 'share' / 'proj',
            Path.home() / 'anaconda3' / 'share' / 'proj',
        ])
    
    found_locations = []
    for path in search_paths:
        if path is None:
            continue
        path = Path(path)
        if path.is_dir():
            proj_db = path / 'proj.db'
            if proj_db.is_file():
                found_locations.append(path)
    
    return found_locations


def check_gdal_import():
    """Check if GDAL can be imported and get version info."""
    try:
        from osgeo import gdal, osr
        gdal_version = gdal.__version__
        return True, gdal_version, None
    except ImportError as e:
        return False, None, str(e)


def check_proj_functionality():
    """Test PROJ coordinate transformation."""
    try:
        from osgeo import osr
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        return True, "PROJ coordinate transformation successful"
    except Exception as e:
        return False, f"PROJ test failed: {str(e)}"


def check_pyproj():
    """Check pyproj configuration."""
    try:
        import pyproj
        return True, {
            'version': pyproj.proj_version_str,
            'data_dir': pyproj.datadir.get_data_dir(),
        }
    except Exception as e:
        return False, str(e)


def generate_env_setup(proj_lib_path, gdal_data_path=None):
    """Generate environment setup commands."""
    setup_commands = []
    
    if proj_lib_path:
        setup_commands.append(f"export PROJ_LIB={proj_lib_path}")
        setup_commands.append(f"export PROJ_DATA={proj_lib_path}")
    
    if gdal_data_path:
        setup_commands.append(f"export GDAL_DATA={gdal_data_path}")
    elif proj_lib_path:
        # Try to infer GDAL_DATA from PROJ_LIB
        potential_gdal = proj_lib_path.parent / 'gdal'
        if potential_gdal.is_dir():
            setup_commands.append(f"export GDAL_DATA={potential_gdal}")
    
    return setup_commands


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose and fix GDAL/PROJ configuration issues on HPC'
    )
    parser.add_argument(
        '--setup', 
        action='store_true',
        help='Apply environment variable fixes automatically'
    )
    parser.add_argument(
        '--export-script',
        type=str,
        help='Export environment setup to a shell script'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("GDAL/PROJ Diagnostic Tool for HPC Systems")
    print("=" * 70)
    print()
    
    # Check conda environment
    print("1. Checking Conda Environment...")
    conda_prefix = find_conda_prefix()
    if conda_prefix:
        print(f"   ✅ Conda environment found: {conda_prefix}")
        print(f"   Active environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    else:
        print("   ⚠️  Conda environment not detected")
    print()
    
    # Check GDAL import
    print("2. Checking GDAL Installation...")
    gdal_ok, gdal_version, gdal_error = check_gdal_import()
    if gdal_ok:
        print(f"   ✅ GDAL imported successfully: version {gdal_version}")
    else:
        print(f"   ❌ GDAL import failed: {gdal_error}")
        print("   → Install GDAL: conda install -c conda-forge gdal")
        return 1
    print()
    
    # Check current environment variables
    print("3. Checking Environment Variables...")
    proj_lib = os.environ.get('PROJ_LIB')
    proj_data = os.environ.get('PROJ_DATA')
    gdal_data = os.environ.get('GDAL_DATA')
    
    if proj_lib:
        print(f"   PROJ_LIB: {proj_lib}")
    else:
        print("   ⚠️  PROJ_LIB not set")
    
    if proj_data:
        print(f"   PROJ_DATA: {proj_data}")
    else:
        print("   ⚠️  PROJ_DATA not set")
    
    if gdal_data:
        print(f"   GDAL_DATA: {gdal_data}")
    else:
        print("   ⚠️  GDAL_DATA not set")
    print()
    
    # Search for proj.db
    print("4. Searching for proj.db...")
    proj_locations = find_proj_db()
    if proj_locations:
        print(f"   ✅ Found proj.db in {len(proj_locations)} location(s):")
        for i, loc in enumerate(proj_locations, 1):
            proj_db_file = loc / 'proj.db'
            size = proj_db_file.stat().st_size / (1024*1024)  # MB
            print(f"      {i}. {loc}")
            print(f"         Size: {size:.2f} MB")
    else:
        print("   ❌ proj.db not found in standard locations")
        print("   → Reinstall: conda install -c conda-forge proj-data")
        return 1
    print()
    
    # Check pyproj
    print("5. Checking PyProj Configuration...")
    pyproj_ok, pyproj_info = check_pyproj()
    if pyproj_ok:
        print(f"   ✅ PyProj version: {pyproj_info['version']}")
        print(f"   Data directory: {pyproj_info['data_dir']}")
        pyproj_db = Path(pyproj_info['data_dir']) / 'proj.db'
        if pyproj_db.exists():
            print(f"   ✅ proj.db accessible via pyproj")
        else:
            print(f"   ⚠️  proj.db not found in pyproj data directory")
    else:
        print(f"   ❌ PyProj check failed: {pyproj_info}")
    print()
    
    # Test PROJ functionality
    print("6. Testing PROJ Functionality...")
    proj_ok, proj_msg = check_proj_functionality()
    if proj_ok:
        print(f"   ✅ {proj_msg}")
    else:
        print(f"   ❌ {proj_msg}")
    print()
    
    # Generate recommendations
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    if proj_locations:
        recommended_proj = proj_locations[0]
        print("Add these lines to your shell script or ~/.bashrc:")
        print()
        
        setup_commands = generate_env_setup(recommended_proj)
        for cmd in setup_commands:
            print(f"   {cmd}")
        print()
        
        if args.export_script:
            script_path = Path(args.export_script)
            with open(script_path, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write("# GDAL/PROJ environment configuration\n")
                f.write("# Generated by diagnose_gdal_hpc.py\n\n")
                for cmd in setup_commands:
                    f.write(f"{cmd}\n")
            script_path.chmod(0o755)
            print(f"   ✅ Setup script exported to: {script_path}")
            print(f"   Usage: source {script_path}")
            print()
        
        if args.setup:
            print("Applying environment variables to current session...")
            for cmd in setup_commands:
                var, value = cmd.replace('export ', '').split('=', 1)
                os.environ[var] = value
                print(f"   ✅ Set {var}={value}")
            print()
            
            # Re-test
            proj_ok, proj_msg = check_proj_functionality()
            if proj_ok:
                print(f"   ✅ PROJ now working correctly!")
            else:
                print(f"   ⚠️  PROJ still has issues: {proj_msg}")
    
    print()
    print("For HPC batch jobs (SLURM), add to your job script:")
    print()
    print("   #!/bin/bash")
    print("   #SBATCH --job-name=geoai")
    print("   ")
    print("   # Activate environment")
    print("   conda activate geoAI")
    print("   ")
    print("   # Set GDAL/PROJ paths")
    if proj_locations:
        print(f"   export PROJ_LIB={proj_locations[0]}")
        print(f"   export PROJ_DATA={proj_locations[0]}")
    print("   export GDAL_DATA=$CONDA_PREFIX/share/gdal")
    print("   ")
    print("   # Run your script")
    print("   python your_script.py")
    print()
    
    print("=" * 70)
    
    if proj_ok and proj_locations:
        print("✅ GDAL/PROJ configuration appears to be working!")
        return 0
    else:
        print("⚠️  Some issues detected. Follow recommendations above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())


