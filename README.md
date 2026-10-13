# Actin-stress-fibers-generator
## A Graphic card with at least 8GB VRAM is recommended.
## Environment:
      with Python 3.9 and CUDA 11.7:
            pytorch 1.11.0,
            torchvision 0.12.0,
## unzip 'validation/example/example.7z' could acquire validation data.
    Please unzip the items to the folder 'validation/example/'
    
## unzip 'model/generator.7z.001~generator.7z.022' could acquire pre-trained weight for the network.
    Please unzip the items to the folder 'model/'
 
## Usage:
    run 'run_SFs_generator.py'
    change average_num to 1 if you do not need the average distribution of stress fibers
    allowing for modify your ownn colormap in visualizing stress fibers
    
## Dataset for training
    Unzip file 'dataset.7z'
