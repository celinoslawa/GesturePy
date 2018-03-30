#!/bin/bash
sudo apt-get update
sudo apt-get upgrade
sudo apt install python3-opencv=3.1.*
sudo apt-get install -y python3-pip=9.0.*
sudo apt-get install python-tk=2.7.* python3-tk=3.6.* python-imaging-tk=4.1.*
pip3 install Pillow


python3 cameracapture.py
