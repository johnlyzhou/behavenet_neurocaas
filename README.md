# BehaveNet for Neurocaas

EC2 Instance Notes:

Needs a .behavenet containing directories.json in the /root dir and in the /home/ubuntu dir to work properly, since process is run as /root but all analysis takes place in /home/ubuntu. The analysis only directly references the directories.json in /root/.behavenet, all other jsons are fine in /home/ubuntu/.behavenet.

Usage Notes:

When running an analysis on the NeuroCAAS website, only check ONE input box - it doesn't matter which one, the config.json will inform the EC2 instance of which files to download.

Repository Structure:

examples/ - example JSON files required for NeuroCAAS analysis usage

gamma/ - standard PS-VAE hyperparameter search over alpha, beta, and gamma

nogamma/ - modules to run the PS-VAE hyperparameter search without the gamma hyperparameter, uses the nogamma branch of behavenet

instance/ - scripts and other files to automate the BehaveNet analysis on NeuroCAAS


For further documentation, refer to https://behavenet.readthedocs.io/en/develop/index.html and https://github.com/themattinthehatt/behavenet.
