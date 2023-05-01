# riceInterface
Exhibition interface for the rice project

### Architecture
```
raspberry pi cam -> mjpg streamer -> python opencv analyzer
                                         | [socket communication]
                                  -> web interface
```

### Setup Info
Streaming Resolution: 1920 x 1080  
Screen Resolution: 1440 x 2560

Streaming Web Interface:
http://192.168.1.22:8080/index.html

TP LINK Router IP:
http://192.168.1.253/  
ssid: rice  
password: masterProject  
static ip 192.168.1.22  
router: http://192.168.1.253/  
nameserver 192.168.1.253  
