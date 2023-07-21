#!/home/pcgta/mambaforge/envs/ovseg/bin/python

from modules.server import VoxSegServer

if __name__ == "__main__":
    server = VoxSegServer(batch_size=None)
