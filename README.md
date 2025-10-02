AAPN-Tiny: A Compact Edge-Deployable Adaptive Attention Pyramid Architecture for Multi-Class Fault Diagnosis in Solar Photovoltaic Modules
=
Manuscript ID: IEEE LAT AM T Submission ID: Paper ID: 9716


Authors: <br>
•	Rayappa David Amar Raj  <br>
•	Rama Muni Reddy Yanamala  <br>
•	Archana Pallakonda <br>
•	Kanasottu Anil Naik <br>

Included Scripts
=
This repository contains all scripts required to reproduce the simulation and numerical results presented in the article. <br>

| Source Code	| Related Figure/Table |	Description
| aappsolar2c.ipynb | Figs. 4,5,6 shows the architecture and results are shown in Figs. 8, 9, 10, 11 |	Interactive python notebook having python code developed for the AAPN model |
architecture. This code classifies 2-class solar faults
aappsolar8c.ipynb
Fig. 4,5,6 shows the architecture and results are shown in Figs. 8, 9, 10, 11	Interactive python notebook having python code developed for the AAPN model architecture. This code classifies 8-class solar faults
aapp11csolar.ipynb
Fig. 4,5,6 shows the architecture and results are shown in Figs. 8, 9, 10, 11	Interactive python notebook having python code developed for the AAPN model architecture. This code classifies 11-class solar faults
aappsolar12.ipynb
Fig. 4,5,6 shows the architecture and results are shown in Figs. 8, 9, 10, 11	Interactive python notebook having python code developed for the AAPN model architecture. This code classifies 12-class solar faults
AAPP_det_12c_x1.ipynb
Table-111	Ablation study: Source code having 2 enhanced blocks and 2 dense layers in SEA block
AAPP_det_12c_x2.ipynb
Table-111	Ablation study: Source code having 1 enhanced blocks and 2 dense layers in SEA block
AAPP_det_12c_xy1.ipynb
Table-111	Ablation study: Source code having 3 enhanced blocks and 1 dense layers in SEA block
AAPP_det_12c.ipynb
Table-111	Ablation study: Source code having 3 enhanced blocks and 2 dense layers in SEA block
AAPP_det_12c_xy2.ipynb
Table-111	Ablation study: Source code having 3 enhanced blocks and 3 dense layers in SEA block


Requirements:
=
•	Python / Jupyter notebook <br>
•	Corel dev Edge TPU board <br>
•	gitbash terminal <br>


Hardware Setup and Implementation steps:
=
To successfully run code on the Edge TPU Corel Dev Board, you need to follow several steps. Here’s a comprehensive guide to get your code up and running:

Set Up and Boot the Board 
• Gather requirements: 
• Ensure you have a host computer running Linux, Mac, or Windows 10. 
• Install Python 3 on your host computer. 
• Prepare a microSD card with at least 8 GB capacity. 
• A USB-C power supply (2-3 A / 5 V) and USB-C to USB-A cable. 
• Wi-Fi or Ethernet connection. 
• Flash the board: 
• Download and unzip the SD card image (enterprise-eagle-flashcard-20211117215217.zip). 
• Use balenaEtcher to flash the image to the microSD card. 
• Set the boot mode switches to SD card to boot from the microSD card. 
• Power the board and let it flash the system image to the eMMC memory. The process takes 5-10 minutes. 
• Once done, change the boot mode switches to eMMC mode and boot up the board. 
Access the Dev Board’s shell: 
    • Install the Mendel Development Tool (MDT) on your host computer: 
    • python3 -m pip install --user mendel-development-tool 
    • If necessary, add ~/.local/bin to your PATH: 
    • echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bash_profile 
    • source ~/.bash_profile 
    • Windows users: Set up an alias for MDT in Git Bash: 
    • echo "alias mdt='winpty mdt'" >> ~/.bash_profile 
    • source ~/.bash_profile 
    • Connect the Dev Board to your computer using a USB-C cable. 
    • Run mdt devices to ensure MDT can detect the board. 
    • If it’s ready, you should see the board's hostname and IP address, such as: 
    • orange-horse (192.168.100.2) 
    • Now, run mdt shell to access the board’s shell.

Connect the Board to the Internet 
    • You need the board online for software updates, model downloads, etc. 
    • Use either Wi-Fi or Ethernet for internet access. 
    • For Wi-Fi, use nmtui to select and activate a network.   
    • Alternatively, connect to Wi-Fi using the command: 
    • nmcli dev wifi connect <NETWORK_NAME> password ifname wlan0

Prepare the Environment 
    • Update the board’s software: 
    • Run the following commands to ensure your board is up to date: 
    • sudo apt-get update 
    • sudo apt-get dist-upgrade

Transfer Files to the Board 
    • Using mdt push: To transfer files to the board, use mdt push. Here’s an example of how to push files to the board: 
    • mdt push <local_file_path> <remote_file_path> For example, to push a Python script: mdt push ./your_script.py /home/board_user/ This will copy the file from your local machine to the Dev Board.
    Open Git Bash and MDT Shell 
        • Git Bash (on Windows): 
        • Git Bash allows you to run shell commands with a Unix-like environment. 
        • You can open it by searching for "Git Bash" in the Start Menu after you’ve installed Git for Windows. 
        • MDT Shell: 
        • Use mdt shell to open the shell of the Dev Board, allowing you to run commands directly on the board.

Run Code on Edge TPU 
    • Once you have your files on the board, you can start running the code. 
    • Make sure any dependencies (like scikit-learn, numpy, etc.) are installed on the board. 
    • sudo apt-get install python3-numpy python3-sklearn 
    • Once everything is set up, navigate to your file directory on the Dev Board: 
    • cd /home/board_user/ • Run your Python script: 
    • python3 your_script.py 
    • The Edge TPU will accelerate the inference process, improving the efficiency of your model.

Access the Board’s Files (Optional) 
    • Git Access: If you want to manage files using Git, you can clone repositories directly onto the Dev Board by running: 
    • git clone <repository_url>

References and Documentation 
=
    • Official Coral Dev Board Documentation: 
    • Coral Dev Board Get Started Guide 
    • https://coral.ai/docs/dev-board/get-started/#update-mendel 
    • Mendel Development Tool Documentation: 
    • MDT Documentation

Description of proposed work
=
The proposed work outlines a deep learning architecture called the Adaptive Attention Pyramid Network (AAPN), which combines several key components to enhance feature extraction, model regularization, and classification accuracy. Below is a detailed breakdown of the work:

Key Components:
=

Squeeze-and-Excitation (SE) Block:
=

    The SE block is a feature recalibration mechanism that adapts to the importance of different channels in the feature maps.
    It first performs a Global Average Pooling (GAP) to extract a global context of the input feature map, followed by two fully connected layers: one with a reduced number of filters (filters // ratio, typically reducing to 1/8th of the total number of channels) and the second that restores the channel dimension.
    The output of these layers is passed through a sigmoid activation function to create a set of scaling factors, which are then multiplied element-wise with the original feature map using the Multiply layer. This allows the model to adjust the importance of each channel dynamically.

Enhanced Pyramid Block:
=

    This  block focuses on enhancing feature extraction by incorporating Depthwise Separable Convolutions (DSC) along with the SE block:
    Depthwise Separable Convolution: A more efficient convolution operation that reduces computational complexity while maintaining performance. It consists of two steps: depthwise convolution (applied to each input channel separately) and pointwise convolution (1x1 convolution to combine the depthwise results).
    After the depthwise convolution, a 1x1 convolution layer is applied to increase the number of filters and introduce more non-linearities.
    The SE block follows to adaptively recalibrate the importance of channels.
    The output is then normalized with BatchNormalization and passed through a MaxPooling layer to reduce spatial dimensions.

Adaptive Attention Pyramid Network (AAPN):
=

    The AAPN is a stacked architecture that integrates multiple Enhanced Pyramid Blocks. It progressively increases the number of filters in each pyramid block (64, 128, 256) to capture increasingly complex features at different scales.
    The architecture starts with an initial convolutional layer followed by three stacked pyramid blocks, each refining the feature representations through depthwise separable convolutions, SE blocks, and pooling operations.
    After the pyramid blocks, Global Average Pooling is applied to compress the spatial dimensions of the feature map into a single vector. This is followed by a fully connected layer with 128 units, incorporating Dropout (0.4) to prevent overfitting.
    The final output layer is a softmax classifier that outputs probabilities across the specified number of classes (num_classes = 12).

Model Compilation:
=

    The model is compiled using the Adam optimizer with a learning rate of 0.001, and the Categorical Crossentropy loss function with label smoothing (0.1) to improve generalization.
Accuracy is tracked as the evaluation metric.

Summary of the Architecture:
=

    Input Shape: The model accepts input images of shape (40, 24, 1) (height, width, and 1 channel for grayscale images).

Network Design:
=

    A sequence of convolutions, depthwise separable convolutions, and SE blocks forms the core of the feature extraction.
    The model progressively learns features at different scales, making it adaptive to varying spatial patterns in the input data.
    Regularization: Dropout (0.4) and label smoothing (0.1) are used for regularization, aiming to reduce overfitting and improve the model’s generalization ability.
    Output: The final output is a vector of class probabilities, suitable for multi-class classification tasks.

Proposed Work:
=

    The proposed work aims to leverage a combination of advanced deep learning techniques—such as depthwise separable convolutions, Squeeze-and-Excitation blocks, and pyramid structures—to build a more efficient, scalable, and accurate model for image classification tasks. The architecture is designed to balance between performance and computational efficiency, making it ideal for resource-constrained environments, while still achieving high classification accuracy through adaptive feature extraction.
