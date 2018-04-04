# Image Text To Speech

Image Text To Speech attempts to implement character detection in complex indoor/outdoor scene images, and classification of detected characters. 

Work in progress.

## Getting Started

To extract characters from an image run binary_conversion.py with an image "Input.png" in the working directory.

To train the neural network on EMNIST letters dataset for 100 epochs run "emnist.py train"

To classify characters extracted from binary_conversion.py run "emnist.py classify"

### Dependencies

```
OpenCV
numpy
pytorch
torchvision
skimage
PIL
```
## Contributing

## Acknowledgments

* Image Binzarization loosely follows "Text Detection in Indoor/Outdoor Scene Images - Gatos, Pratikakis, Kepene, Perantonis 
