# Rice Interface

Exhibition interface for the rice project

project by [@cqx931](https://github.com/cqx931/)

## Installation

```sh
pip install -r requirements.txt
```

## Run

```sh
python app
```

## Parameters

- `d`: debug

## Architecture
```
raspberry pi cam -> mjpg streamer -> python opencv analyzer
                                         | [socket communication]
                                  -> web interface
```

## Related Repo
- [mjpg-streamer](https://github.com/cqx931/mjpg-streamer)
- [masterProject](https://github.com/cqx931/masterProject)

## Colab Experiments
- [rice](https://colab.research.google.com/drive/1altnQe2wf7Ele74IKrhHKjxIN2zCHZoU#scrollTo=BG08-HZ5vANw)
- [ricev2](https://colab.research.google.com/drive/1Ay9ZPEoNlbPBK2T2aXBxXJvVtoiKFKdG)
- [rice_tsne](https://colab.research.google.com/drive/1dMf2GaFHH_nvtReOu49NmBqYSepYniiX?usp=sharing#scrollTo=rW8AwjDf8hh_)

## Setup Info

Streaming Resolution: 1920 x 1080  
Screen Resolution: 1440 x 2560

Streaming Web Interface:
http://192.168.1.22:8080/index.html

TP LINK Router IP: 
http://192.168.1.253/

```
ssid: rice  
password: masterProject  
raspberry pi static ip: 192.168.1.22  
router: http://192.168.1.253/  
nameserver: 192.168.1.253
```
