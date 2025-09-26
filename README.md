# FIRE 🔥

**Fidelity Image Resolution Enhancement** (FIRE) is a cutting-edge tool designed to enhance image resolution with unparalleled precision and fidelity. Whether you're working with low-resolution images, restoring old photographs, or improving the quality of visual data, FIRE leverages advanced algorithms to deliver stunning results.

With a focus on speed, accuracy, and ease of use, FIRE is perfect for developers, researchers, and creatives looking to elevate their image processing workflows. Dive in and experience the future of image resolution enhancement today!

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/2treesan/FIRE
cd FIRE
```

2. Create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Run the main script with your configurations:

```bash
python main.py \
    loader.hr_dir="<path_to_hr>" \
    model="<model_name>" \
    trainer.epochs=100 \
    # Add other configurations as needed
```

- Replace `<path_to_hr>` with the path to your high-resolution images.
- Replace `<model_name>` with the model defined in the `src/model` directory.

## Available Models

- **EDSRLite**: A lightweight version of the Enhanced Deep Super-Resolution network, optimized for faster performance while maintaining high-quality results.
