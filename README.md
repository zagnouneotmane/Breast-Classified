# Whole-slide CNN Training Pipeline

This guide will help you set up an EC2 instance with TensorFlow GPU support, installing project dependencies, and running your data loading or DVC pipeline for your whole-slide CNN training project.

## Step 1: Launch an EC2 Instance

1. **Log in to AWS Management Console**:
   - Navigate to the [AWS Management Console](https://aws.amazon.com/console/).
   - Sign in with your AWS credentials.

2. **Launch a new EC2 instance**:
   - Go to the EC2 Dashboard.
   - Click "Launch Instance."
   - Choose an Amazon Machine Image (AMI). For TensorFlow with GPU, select an Amazon Deep Learning AMI or a standard Ubuntu AMI.
   - Choose an instance type. Select a GPU instance type such as `p2.xlarge`, `p3.2xlarge`, or `g4dn.xlarge`.
   - Configure security group: Add rules to allow SSH (port 22) from your IP address.
   - Review and launch the instance. Ensure you have a key pair for SSH access.

## Step 2: Connect to the EC2 Instance

1. **Obtain the Public DNS of your instance**:
   - In the EC2 Dashboard, find your running instance and note its Public DNS.

2. **Connect via SSH**:
   - Open a terminal on your local machine.
   - Connect using the SSH command: 
     ```bash
     ssh -i /path/to/your-key-pair.pem ubuntu@your-instance-public-dns
     ```

## Step 3: Install NVIDIA Drivers and CUDA Toolkit

1. **Update the package lists**:
   ```bash
   sudo apt-get update
   ```

2. **Install necessary packages**:
   ```bash
   sudo apt-get install -y build-essential
   ```

3. **Install NVIDIA drivers**:
   ```bash
   wget https://us.download.nvidia.com/XFree86/Linux-x86_64/460.32.03/NVIDIA-Linux-x86_64-460.32.03.run
   sudo bash NVIDIA-Linux-x86_64-460.32.03.run
   ```

4. **Install CUDA Toolkit**:
   - Download and install CUDA from [NVIDIA's CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
   sudo sh cuda_11.0.2_450.51.05_linux.run
   ```

5. **Add CUDA to your PATH and LD_LIBRARY_PATH**:
   ```bash
   echo 'export PATH=/usr/local/cuda-11.0/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

## Step 4: Install cuDNN

1. **Download cuDNN**:
   - Go to the [cuDNN download page](https://developer.nvidia.com/cudnn).
   - Download the version that matches your CUDA version.

2. **Install cuDNN**:
   ```bash
   tar -xzvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
   sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
   sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
   sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
   ```

## Step 5: Clone Your Git Repository

1. **Install Git**:
   ```bash
   sudo apt-get install -y git
   ```

2. **Clone your repository**:
   ```bash
   git clone https://github.com/git-repository.git
   cd git-repository
   ```

## Step 6: Install TensorFlow with GPU Support

1. **Install virtual environment package**:
   ```bash
   sudo apt install python3-venv
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv tensorflow-gpu
   source ~/tensorflow-gpu/bin/activate
   ```

3. **Install TensorFlow GPU**:
   ```bash
   pip install --upgrade pip
   pip install tensorflow-gpu
   ```
4. **Install OpenSlide Dependencies**
   ```bash
   sudo apt-get update
   sudo apt-get install openslide-tools
   sudo apt-get install python3-openslide
   sudo apt-get install libopenslide0
   ```

## Step 7: Install Project Dependencies

1. **Navigate to your project directory** (if not already there):
   ```bash
   cd your-repository
   ```

2. **Install the requirements**:
   ```bash
   pip install -r requirements.txt
   ```

## Step 8: Run Data Loading Script or Reproduce DVC Pipeline

1. **Run the data loading script**:
   ```bash
   python3 src/stages/data_load.py --config params.yaml
   ```

   **OR**

2. **Reproduce the DVC pipeline**:
   ```bash
   dvc repro
   ```
