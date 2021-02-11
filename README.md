# behavenet-neurocaas

Needs a .behavenet containing directories.json in the /root dir and in the /home/ubuntu dir to work properly, since process is run as /root but all analysis takes place in /home/ubuntu. The analysis only directly references the directories.json in /root/.behavenet, all other jsons are fine in /home/ubuntu/.behavenet

Do NOT check any input boxes except for data.hdf5 - the config.json will inform the EC2 instance of which to download
