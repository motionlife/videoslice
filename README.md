# Video Slicing and Captioning Model
model for generateing multiple short videos from a long video based on semantic context

## Installation

Use the package manager pip or conda to install your tensorflow 2.4 GPU environment and other dependencies.

```bash
conda create -n tf python=3.9 cudatoolkit=11.1 cudnn=8.1 tensoflow=2.4 matplotlib 
pip install tensorflow-datasets tensoflow-addons
```
Download corresponding datasets if you want to train your own model weights, pretrained weights download link:
<a id="raw-url" href="https://drive.google.com/drive/u/0/shared-with-me">Download Weights</a>


## Usage

You can run the demo.ipynb in Google Colab or run the script below.

```bash
# train the model from scratch
python train.py --arguments

# specify customized parameters
run.sh --options --img_dir="your/test/image/directory" --weights="weights/file/path"
```

### example output
TBD

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
