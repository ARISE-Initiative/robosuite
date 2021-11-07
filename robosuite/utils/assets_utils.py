"""
Utility functions for downloading asset files
"""

import os
import sys
import requests
import pathlib

def download_assets():
    """
    Download robosuite assets
    """
    d = pathlib.Path(__file__).parent.resolve()

    assets_path = os.path.join(d, "..", "models" , "assets")
    assets_url = "http://utexas.box.com/shared/static/g961uwr5zvh7ybbvie5k18ywmjs19xn8.gz"
    assets_tmp_path = "/tmp/robosuite_assets_v1.tar.gz"

    if not os.path.exists(assets_path):
        print("Installing the robosuite assets")
        print("Creating assets path")
        os.makedirs(assets_path)

        print("Downloading assets from", assets_url)

        assets_obj = requests.get(assets_url)
        with open(assets_tmp_path, 'wb') as local_file:
            local_file.write(assets_obj.content)

        print("Decompressing assets to", assets_path)
        os.system("tar -zxf {} --directory {}".format(assets_tmp_path, os.path.dirname(assets_path)))


if __name__ == "__main__":
    download_assets()
