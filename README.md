#

<div align="center">
<h1>Try Harder: Hard Sample Generation and Learning for
Clothes-Changing Person Re-ID</h1>


</div>

This is the official implementation of the paper "Try Harder: Hard Sample Generation and Learning for Clothes-Changing Person Re-ID" (ACM MM 2025).

---
![teaser2](/assets/frame-whitebg.png)&nbsp;

---

For convenience, we provide pre-generated hard samples for LTCC and PRCC datasets, as well as the trained SOTA model weights. You can download them from:[Download Link](https://pan.baidu.com/s/1TEGRhTwEKuIFGi6ECBEsaw?pwd=a1mr) 

## Hard Sample Generation

### Semantic-preserving Hard Sample Generation(SHSG)

#### Environment Setup

```bash
conda env create -f environment_shsg.yml
conda activate idm
```

#### Data Generation

Before running the person filter script, modify the directory paths in the `__main__` section according to your data structure:

```python
if __name__ == "__main__":
    # Source image directory and target directory
    person_dir = "/path/to/your/dataset/train"  # Replace with your dataset path
    output_dir = "/path/to/your/HQ_FIlter_Path/output"  # Replace with your output path
    
    # Execute filtering
    filter_person_images_per5(person_dir, output_dir)
```

To filter high-quality images from your dataset, you can use the provided filtering script:

```bash
cd ./SHSG-IDM-VTON
python ./Person_HQ_Filter_LTCC.py
python ./Person_HQ_Filter_PRCC.py
```

Then, you need to download the required model weights for IDM-VTON. Please refer to the official repository for detailed instructions:

[IDM-VTON GitHub Repository](https://github.com/yisol/IDM-VTON)

After downloading the weights and confirming that gradio_demo can run normally locally, you need to place the `./SHSG-IDM-VTON/change-cloth-script.py` file into the `./SHSG-IDM-VTON/IDM-VTON-main/gradio_demo` directory. And modify the directory paths in the `main()` function according to your data structure:

```python
def main():
    dataset_name = "ltcc"  # or "prcc"

    if dataset_name == "ltcc":
        person_folder = "/path/to/your/person/images"  # Replace with your person images path
        cloth_folder = "/path/to/your/cloth/images"   # Replace with your cloth images path
        output_folder = "/path/to/your/output/folder" # Replace with your output path
    elif dataset_name == "prcc":
        person_folder = "/path/to/your/prcc/person/images"
        cloth_folder = "/path/to/your/prcc/cloth/images"
        output_folder = "/path/to/your/prcc/output/folder"
    else:
        raise ValueError("dataset name must be set!")
```

Then, run the script to generate hard samples:

```bash
cd ./SHSG-IDM-VTON/IDM-VTON-main/
python change-cloth-script.py
```

---



### Coarse-grained Hard Positive Sample Generation (CHPSG)

In this part, we use the inpaint-diffusion model to generate coarse-grained hard positive samples.

First, we use the [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) method to extract the cloth masks for each training image.

After get the cloth masks for each training image, we can generate the coarse-grained hard positive samples as follows:

```bash
cd ./CHPSG
python inpaint_batch.py
```

Now, we can train the more precise model with various baseline method based on the original training dataset and our coarse-grained hard positive samples. Note: our coarse-grained hard samples don't have cloth labels, so some baseline methods can not be seleted to train.


## Hard Sample Learning

### Environment Setup

Create conda environment

```bash
conda env create -f environment_hsl.yml
```

Install pytorch

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

Install apex

```bash
cd ./apex
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./
```

### Troubleshooting

If you encounter any issues with the environment setup, please refer to the official GitHub repositories of the following methods for additional installation guidance:

- [CAL](https://github.com/raoyongming/CAL)
- [AIM](https://github.com/BoomShakaY/AIM-CCReID)
- [FIRe](https://github.com/QizaoWang/FIRe-CCReID)
- [CSSC](https://github.com/QizaoWang/CSSC-CCReID)

Star ⭐ us if you like it!
