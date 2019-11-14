Timbre Recognition
---
Repository for timbre recogntion net. Written in Python3.7 and TensorFlow2.0. Other
Dependencies can be found in requirements3.7.txt. Note that the version of tensorflow
used is dependant on hardware (GPU or not). 

The project uses an Inception-ResNet-v2 base architecture modififed for audio data
to embed timbre instances into a vector space using triplet loss. 

##### In this directory (/):
If a flie or directory isn't in this repository it's because it's ignored.
- **timbre_recognition/**: main code directory that contains the timbre recognition library
- **proposal/**: written proposal in latex
- **thesis/**: written thesis in latex
- **references/**: references for thesis
- **requirements3.7.txt**: viretualenv dependancies for the project
