---
title: "HPC Setup Notes"
subtitle: "GRIT HPC Notes for GEOG-288KC"
jupyter: geoai
format:
  html:
    code-fold: false
---

# GRIT HPC Setup Guide for GEOG-288KC

This guide provides step-by-step instructions for setting up access to the GRIT HPC system and configuring Jupyter Notebook with the geoAI kernel.

## Prerequisites

- GRIT user account credentials
- SSH client installed on your local machine
- Web browser with JavaScript enabled

## Setup Instructions

### Step 1: Create a GRIT User Account

Contact your course administrator to request a GRIT user account if you don't already have one.

### Step 2: Configure SSH Access and Kernel Symbolic Link

1. Open your terminal/command prompt
2. Connect to the GRIT HPC system via SSH:

   ```bash
   ssh <net_id>@hpc.grit.ucsb.edu
   ```

3. Once logged in, create a symbolic link to the geoAI kernel:

   ```bash
   ln -s /home/g288kc/.local/share/jupyter/kernels/geoai ~/.local/share/jupyter/kernels/
   ```

4. Exit the SSH session:

   ```bash
   exit
   ```

### Step 3: Access the GRIT HPC Web Portal

1. Open your web browser and navigate to: <https://hpc.grit.ucsb.edu>
2. Log in using your GRIT credentials

   ![GRIT Login Page](https://gist.github.com/user-attachments/assets/5c749926-9a50-463d-82c0-3bff1f05a529)

### Step 4: Launch Jupyter Notebook

1. Once logged in, click on **"My interactive sessions"** in the menu
2. Under the **"Servers"** section, select **"Jupyter Notebook"**

   ![Interactive Sessions Menu](https://gist.github.com/user-attachments/assets/ffb16d7f-e77c-4c7f-a1e1-7262578e58b7)

### Step 5: Configure Jupyter Notebook Instance

Create a new Jupyter Notebook instance with the following parameters:

| Parameter | Value |
|-----------|-------|
| **Username** | Your GRIT username |
| **Partition** | `grit_nodes` |
| **Job Duration** | `168` (maximum allowed, in hours) |
| **CPUs** | `4` (adjust based on needs) |
| **RAM** | `16` (GB, adjust based on needs) |
| **Job Name** | Optional - leave empty or provide descriptive name |

Click **"Launch"** to start your Jupyter Notebook instance.

### Step 6: Configure the geoAI Kernel

To enable AI features in your Jupyter Notebook:

1. Once your Jupyter Notebook is running, go to the **"Kernel"** menu
2. Select **"Change Kernel..."**

   ![Kernel Menu](https://gist.github.com/user-attachments/assets/9e2e32a6-7e4d-4b28-9520-c244b11c4656)

3. From the list of available kernels, select **"geoAI Course"**

   ![Kernel Selection](https://gist.github.com/user-attachments/assets/d9d5cf04-2e70-4ecf-bc44-0f2e17a2c8a3)

## Important Notes

- **Session Duration**: The maximum job duration is 168 hours (7 days). Plan your work accordingly.
- **Resource Allocation**: The CPU and RAM values provided are recommendations. Adjust based on your computational needs.
- **Kernel Link**: The symbolic link created in Step 2 is essential for accessing the geoAI kernel. Without it, the kernel won't appear in your Jupyter options.

## Troubleshooting

### Cannot see geoAI kernel

- Verify the symbolic link was created correctly by running:

  ```bash
  ls -la ~/.local/share/jupyter/kernels/
  ```

- Restart your Jupyter Notebook session

### SSH Connection Failed

- Verify you're using the correct hostname: `hpc.grit.ucsb.edu`
- Check that your GRIT account is active
- Ensure you're connected to the internet

### Jupyter Session Won't Start

- Check if you have any existing sessions that need to be terminated
- Verify you've selected the correct partition (`grit_nodes`)
- Try reducing the requested resources (CPUs/RAM)

## Support

For additional help:

- Contact your course instructor
- Reach out to GRIT HPC support
- Consult the [UCSB HPC documentation](https://csc.cnsi.ucsb.edu/)

---

*Last updated: September 2025*
*Based on instructions from [tjaartvdwalt/grit_user_setup.md](https://gist.github.com/tjaartvdwalt/e4080d91aa38d0758d8ef813f75d64ac)*
