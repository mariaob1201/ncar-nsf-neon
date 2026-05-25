#!/usr/bin/env python3
"""
CTSM Parameter Perturbation Workflow
Non-invasive approach using CTSM's built-in parameter modification system
"""

import yaml
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import netCDF4 as nc
import numpy as np


class CTSMExperimentManager:
    """Manages CTSM parameter perturbation experiments"""

    def __init__(self, base_dir="/experiments"):
        self.base_dir = Path(base_dir)
        self.param_sets_dir = self.base_dir / "parameter_sets"
        self.results_dir = self.base_dir / "results"
        self.config_file = self.base_dir / "experiment_config.yaml"

        # Create directories if they don't exist
        self.param_sets_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_experiment_config(self):
        """Load experiment configuration from YAML"""
        with open(self.config_file) as f:
            return yaml.safe_load(f)

    def create_parameter_file(self, base_params, modifications, output_file):
        """
        Modify a parameter file (NetCDF format)

        Args:
            base_params: Path to original parameter file
            modifications: Dict of parameter changes
            output_file: Where to save modified file
        """
        # Copy base file
        shutil.copy(base_params, output_file)

        # Open and modify
        with nc.Dataset(output_file, 'r+') as ds:
            for param_name, new_value in modifications.items():
                if param_name in ds.variables:
                    # Handle different parameter types
                    if isinstance(new_value, dict):
                        # For indexed parameters (e.g., per PFT)
                        var = ds.variables[param_name]
                        for idx, val in new_value.items():
                            var[int(idx)] = val
                    else:
                        # For scalar or uniform changes
                        ds.variables[param_name][:] = new_value

                    print(f"  Modified {param_name} = {new_value}")

    def create_user_namelist(self, modifications, output_file):
        """
        Create user_nl_clm file with namelist modifications

        Args:
            modifications: Dict of namelist variable changes
            output_file: Path to user_nl_clm file
        """
        with open(output_file, 'w') as f:
            f.write("! User namelist modifications for parameter perturbation\n")
            f.write(f"! Generated: {datetime.now()}\n\n")

            for var_name, value in modifications.items():
                if isinstance(value, str):
                    f.write(f"{var_name} = '{value}'\n")
                elif isinstance(value, bool):
                    f.write(f"{var_name} = .{str(value).lower()}.\n")
                else:
                    f.write(f"{var_name} = {value}\n")

    def setup_experiment(self, exp_name, exp_config):
        """
        Set up a single experiment

        Args:
            exp_name: Name of the experiment
            exp_config: Configuration dict for this experiment
        """
        print(f"\n{'=' * 60}")
        print(f"Setting up experiment: {exp_name}")
        print(f"{'=' * 60}")

        # Create experiment directories
        exp_param_dir = self.param_sets_dir / exp_name
        exp_result_dir = self.results_dir / exp_name
        exp_param_dir.mkdir(exist_ok=True)
        exp_result_dir.mkdir(exist_ok=True)

        # Store metadata
        metadata = {
            "name": exp_name,
            "created": datetime.now().isoformat(),
            "description": exp_config.get("description", ""),
            "modifications": exp_config.get("modifications", {}),
            "base_case": exp_config.get("base_case", "NEON_case")
        }

        with open(exp_result_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Process parameter modifications
        if "parameter_file_mods" in exp_config:
            print("\nCreating modified parameter file...")
            base_param_file = exp_config.get("base_param_file",
                                             "/opt/ncar/cesm2/inputdata/lnd/clm2/paramdata/clm5_params.c171117.nc")
            output_param_file = exp_param_dir / "modified_params.nc"

            self.create_parameter_file(
                base_param_file,
                exp_config["parameter_file_mods"],
                output_param_file
            )

        # Process namelist modifications
        if "namelist_mods" in exp_config:
            print("\nCreating user namelist...")
            user_nl_file = exp_param_dir / "user_nl_clm"
            self.create_user_namelist(
                exp_config["namelist_mods"],
                user_nl_file
            )

        # Generate run script
        self.generate_run_script(exp_name, exp_config, exp_result_dir)

        print(f"\n✓ Experiment {exp_name} setup complete!")
        print(f"  Parameters: {exp_param_dir}")
        print(f"  Results: {exp_result_dir}")

    def generate_run_script(self, exp_name, exp_config, result_dir):
        """Generate a shell script to run the experiment"""

        script_content = f"""#!/bin/bash
# Run script for experiment: {exp_name}
# Generated: {datetime.now()}

# Set up environment
cd /home/user/CLM-NEON

# Clone base case
CASE_NAME="{exp_name}"
BASE_CASE="{exp_config.get('base_case', 'NEON.ABBY')}"

echo "Cloning case from $BASE_CASE..."
./create_case --clone $BASE_CASE --case $CASE_NAME --cime-output-root {result_dir}

cd $CASE_NAME

# Copy modified parameter file if it exists
if [ -f "{self.param_sets_dir / exp_name / 'modified_params.nc'}" ]; then
    echo "Copying modified parameter file..."
    cp {self.param_sets_dir / exp_name / 'modified_params.nc'} .
    # Add to user_nl_clm
    echo "paramfile = '$(pwd)/modified_params.nc'" >> user_nl_clm
fi

# Copy user namelist modifications if they exist
if [ -f "{self.param_sets_dir / exp_name / 'user_nl_clm'}" ]; then
    echo "Applying namelist modifications..."
    cat {self.param_sets_dir / exp_name / 'user_nl_clm'} >> user_nl_clm
fi

# Build and run
echo "Building case..."
./case.build

echo "Running simulation..."
./case.submit

echo "Experiment {exp_name} submitted!"
"""

        script_path = result_dir / "run_experiment.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        script_path.chmod(0o755)
        print(f"  Run script: {script_path}")

    def run_all_experiments(self):
        """Set up all experiments from config file"""
        config = self.load_experiment_config()

        print(f"\nProcessing {len(config['experiments'])} experiments...")

        for exp_name, exp_config in config['experiments'].items():
            self.setup_experiment(exp_name, exp_config)

        print("\n" + "=" * 60)
        print("All experiments configured!")
        print("=" * 60)
        print("\nTo run experiments, execute the run_experiment.sh script")
        print("in each experiment's results directory.")


# Example configuration file structure
EXAMPLE_CONFIG = """
experiments:
  sensitivity_leaf_cn:
    description: "Test sensitivity to leaf C:N ratio"
    base_case: "NEON.ABBY"
    parameter_file_mods:
      leafcn:
        # Modify leaf C:N for different PFTs
        1: 35.0    # needleleaf evergreen temperate tree
        2: 30.0    # needleleaf evergreen boreal tree
    namelist_mods:
      hist_fincl1: "'GPP','NPP','NEE','LEAFC'"
      hist_nhtfrq: -24  # daily output

  increased_vcmax:
    description: "Increase maximum photosynthetic capacity"
    base_case: "NEON.ABBY"
    parameter_file_mods:
      vcmx25:
        1: 70.0    # increase from default ~50
        2: 65.0
    namelist_mods:
      hist_fincl1: "'GPP','FPSN','Vcmx25'"

  modified_soil_params:
    description: "Test different soil hydraulic properties"
    base_case: "NEON.ABBY"
    namelist_mods:
      # These would go in the namelist
      organic_frac: 0.15
      baseflow_scalar: 0.002
"""

if __name__ == "__main__":
    # Initialize manager
    manager = CTSMExperimentManager()

    # Create example config if it doesn't exist
    if not manager.config_file.exists():
        print("Creating example configuration file...")
        with open(manager.config_file, 'w') as f:
            f.write(EXAMPLE_CONFIG)
        print(f"Edit {manager.config_file} to define your experiments")
    else:
        # Run all experiments
        manager.run_all_experiments()