"""
Utility functions for downloading asset files
"""

import os
import sys
#import robosuite
import pathlib

def download_assets():
    """
    Download robosuite assets
    """
    d = pathlib.Path(__file__).parent.resolve()

    assets_path = os.path.join(d, "..", "models" , "assets")
    assets_url = "http://utexas.box.com/shared/static/g961uwr5zvh7ybbvie5k18ywmjs19xn8.gz"

    if not os.path.exists(assets_path):
        print("Installing the robosuite assets")
        print("Creating assets path")
        os.makedirs(assets_path)

        print("Downloading assets from", assets_url)
        os.system(
            "wget -c --retry-connrefused --tries=5 --timeout=5 "
            "{} -O /tmp/robosuite_assets_v1.tar.gz".format(assets_url)
        )

    print("Decompressing assets to", assets_path)
    os.system("tar -zxf /tmp/robosuite_assets_v1.tar.gz --directory {}".format(os.path.dirname(assets_path)))


if __name__ == "__main__":
    download_assets()
