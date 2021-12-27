# PS-VAE for NeuroCAAS

The [Neuroscience Cloud Analysis As a Service (NeuroCAAS)](http://www.neurocaas.org/) (Abe et al. 2020) implementation of the PS-VAE can be found at http://www.neurocaas.com/analysis/11 (as an extension of BehaveNet). NeuroCAAS replaces the need for expensive computing infrastructure and technical expertise with inexpensive, pay-as-you-go cloud computing and a simple drag-and-drop interface. To fit the PS-VAE, the user simply needs to upload a video, a corresponding labels file, and configuration files specifying desired model parameters. Then, the NeuroCAAS analysis will automatically perform the hyperparameter search as described above, parallelized across multiple GPUs. The output of this process is a downloadable collection of diagnostic plots and videos, as well as the models themselves.

This repository contains the source code copied into the custom [Amazon Machine Image (AMI)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html) for the PS-VAE analysis on NeuroCAAS. It follows the guidelines in the [NeuroCAAS Developer Guide] (https://github.com/cunningham-lab/neurocaas/blob/master/docs/devguide.md) with a few exceptions as noted below:

The BehaveNet analysis requires multiple dataset files with flexible extensions and names. In addition, it has certain environment setup requirements that require user input and cannot be done in an automated NeuroCAAS analysis. However, the current NeuroCAAS drag-n-drop interface only allows for a single dataset file for each analysis (selecting multiple dataset files launches a separate analysis instance for each file). 

Since all uploaded dataset files are available in the S3 bucket, this implementation uses the config file to provide all filenames required for the analysis and analysis options, and uploads those files as dataset files. In the internal bash script, it calls a separate script to parse the config file, download all specified dataset files, and run any desired optional commands. Since the config file specifies dataset filenames, selection of a dataset file in the NeuroCAAS web interface does not matter.

Repository Structure:

examples/ - example JSON files required for NeuroCAAS analysis usage, including PS-VAE architecture specifications, compute resources, hyperparameter settings, etc.

gamma/ - implements the hyperparameter search and plotting functions over alpha, beta, and gamma weights for the PS-VAE loss components

nogamma/ - implements the hyperparameter search and plotting functions over alpha and beta, as described in the most updated BehaveNet documentation

instance/ - scripts and other files to automate the BehaveNet analysis on NeuroCAAS

AMI Setup Notes:

Requires a .behavenet containing identical directories.json files in the /root dir and in the /home/ubuntu dir to work properly, since the process is run as root but all analysis takes place as the "ubuntu" user.

NeuroCAAS Usage Notes:

When running an analysis on the NeuroCAAS website, only check ONE input box - it doesn't matter which one, the config.json will inform the EC2 instance of which files to download - checking multiple boxes will launch multiple instances running the exact same analysis (for parallelized jobs using different data files, contact neurocaas@gmail.com).

For further documentation on PS-VAE, refer to https://behavenet.readthedocs.io/en/develop/index.html.
