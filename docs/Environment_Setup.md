# **Environment Setup**

This guide provides step-by-step instructions to set up your development environment using the Anaconda distribution and Spyder IDE. 

However, you are free to use any IDE or environment that suits your preferences, such as **VS Code**, **PyCharm**, or **Jupyter Notebook**, as long as you ensure compatibility with Python 3.10 or higher and install the required dependencies.

---

## **Installing Anaconda**
If you do not have Anaconda installed, follow these steps to set it up:
1. **Download Anaconda:**  
   Visit the [Anaconda Downloads Page](https://www.anaconda.com/products/distribution) and download the installer for your operating system.
   
2. **Install Anaconda:**  
   - Run the installer and follow the instructions.  
   - During installation, ensure that the option to add Anaconda to your system’s PATH is selected (recommended).

3. **Verify Installation:**  
   Open the **Anaconda Prompt** and type:
   conda --version

If installed correctly, this will display the installed version of conda.

## **Installing the Spyder IDE**
Spyder is a scientific Python development environment that integrates seamlessly with Anaconda. Follow these steps to install and configure Spyder (5.15>):

1. **Install Spyder:**  
   Open the Anaconda Prompt and run the following command:
   ```plaintext
   conda install spyder=5.15 -c conda-forge -y
   ```
   
2. **Verify Installation**  
   After installation, type the following command to ensure Spyder is installed correctly:
   ```plaintext
   spyder --version
   ```
                                                                
3. **Launch Spyder**  
   To launch the Spyder IDE from the Anaconda Prompt, simply type:
      ```plaintext
   spyder
   ```

Alternatively, you can open Spyder from the Anaconda Navigator:

- Open Anaconda Navigator.
- Locate Spyder in the list of applications and click Launch.

## **Key Considerations with Anaconda**
1. **Order Matters:**  
   Always install packages available through `conda` first before using `pip`.

2. **Environment Management:**  
   - Anaconda uses `conda` as its package manager, which can conflict with `pip`.  
   - Use `conda` whenever possible to maintain compatibility with the Anaconda ecosystem.

3. **Mixing `pip` and `conda`:**  
   - You can mix `pip` and `conda` packages, but proceed with caution to avoid dependency issues.  
   - Reserve `pip` for packages unavailable in the `conda` repositories (e.g., `alpaca-py`, `alpaca-trade-api`).
   
---

## **Steps to Install `requirements.txt` with Anaconda**

### **1. Create a New Environment**
Creating an isolated environment ensures compatibility and prevents package conflicts.  
Run the following commands to create and activate a new environment:

```
Replace 'env_name' with your preferred environment name
conda create -n env_name python=3.10 -y
conda activate env_name

```

### **2. Install Conda-Compatible Packages**
Install packages available through conda first for better compatibility:
```
conda install numpy pandas matplotlib ta-lib tqdm -c conda-forge -y
```

### **3. Install Remaining Packages with pip**
Use pip to install packages not available through conda:
```
pip install alpaca-py alpaca-trade-api aiohttp websockets websocket-client pydantic pyttsx3 typing-extensions packaging comtypes
```

### **4. Verify Installation**
Check that all packages are correctly installed by running:
```
pip freeze

```
This version ensures all the steps are properly formatted and easy to follow!

See the requirements.txt file for all required dependencies.

