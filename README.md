# CNNs-on-CHB-MIT
The project is about applying CNNs to EEG data from CHB-MIT to predict seizure. It's a group project assigned at UNIVERSITA' DI CAMERINO for computer science bachelor.
The objective of the project was to try to replicate the result obtained in the paper:
[Truong, Nhan Duy, et al. "Convolutional neural networks for seizure prediction using intracranial and scalp electroencephalogram." Neural Networks 105 (2018): 104-111.](https://www.sciencedirect.com/science/article/pii/S0893608018301485)

The algorithm consist to create spectograms of the data and than use them with a CNN model to predict seizure.

More information are in [presentazione.pdf](presentazione.pdf) and [relazione.pdf](relazione.pdf). The two file are respectively the presentation and the relation of the work in italian language.

## Getting Started

### Prerequisites
In the project anaconda was used to managed the packages. Packages required:

* keras 2.2.2
* python 3.6.6
* tensorflow 1.10.0
* matplotlib
* numpy
* pyedflib
* scipy

For the evaluation of the network, training and testing, the GPU is used to have a fast evaluation. By using the CPU the training time is a lot more slowly than using GPU. Packages required for GPU:
* tensorflow-GPU

For the using of the GPU this link was very useful to install all the driver for Ubuntu 18.04 LTS https://medium.com/@naomi.fridman/install-conda-tensorflow-gpu-and-keras-on-ubuntu-18-04-1b403e740e25 (Note that the GPU used was GTX 850M so I can't ensure that the guide linked will work for different hardware).

### Installing

Download or clone the repository on your computer and set the parameters:
* [PARAMETERS_DATA_EDITING.txt](PARAMETERS_DATA_EDITING.txt): contain the parameters for the creation of the spectograms:
  - **pathDataSet**: path of the folder containing the dataset;
  - **FirstPartPathOutput**: path of the folder where spectograms will be saved;
 
* [PARAMETERS_CNN.txt](PARAMETERS_CNN.txt): contain the parameters for the use of CNN:
  - **PathSpectogramFolder**: Path of the folder containing the spectograms;
  - **OutputPath**: file where to save the results;
  - **OutputPathModels**: where to save the CNN models.
 
## Recovering data
The dataset is downloadable from this site: [https://physionet.org/pn6/chbmit/](https://physionet.org/pn6/chbmit/). To get all the data it's suggested to use this command:
```
wget -r --no-parent https://physionet.org/pn6/chbmit/
```
In the code only patients 1, 2, 5, 19, 21, 23 are used, the others are discarded for problems in the data.
**NOTE**: For the patient 19 replace the summary file(chb19-summary.txt) with the one in this repository inside the folder summaryChanged.

## Running

After setted all the parameters run the code.
```
python DataserToSpectogram.py #Creation of the spectograms
python CNN.py #Creation of the CNN and evaluation of the model on the spectograms
python TestThreshold.py #Search the best thresold for each patient
```

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.
 

## Authors

* [**Simone Morettini**](https://github.com/MesSem)
* [**Alessandra Renieri**](https://github.com/a311987)

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the [LICENSE.md](LICENSE.md) file for details
<!---
## Acknowledgments

* ______
--->
