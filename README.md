# BehaveNet for NeuroCAAS

This repository contains files that are copied over to the custom [Amazon Machine Image (AMI)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html) used for the BehaveNet analysis on [NeuroCAAS](http://www.neurocaas.org/). 

AMI Setup Notes:

Requires a .behavenet containing identical directories.json files in the /root dir and in the /home/ubuntu dir to work properly, since the process is run as root but all analysis takes place as the "ubuntu" user.

NeuroCAAS Usage Notes:

When running an analysis on the NeuroCAAS website, only check ONE input box - it doesn't matter which one, the config.json will inform the EC2 instance of which files to download - checking multiple boxes will launch multiple instances running the exact same analysis (for parallelized jobs using different data files, contact neurocaas@gmail.com).

Repository Structure:

examples/ - example JSON files required for NeuroCAAS analysis usage

gamma/ - standard PS-VAE hyperparameter search over alpha, beta, and gamma

nogamma/ - modules to run the PS-VAE hyperparameter search without the gamma hyperparameter, uses the nogamma branch of behavenet

instance/ - scripts and other files to automate the BehaveNet analysis on NeuroCAAS


For further documentation, refer to https://behavenet.readthedocs.io/en/develop/index.html and https://github.com/themattinthehatt/behavenet.
