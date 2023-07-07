# HAR-with-SNN
Human activity recognition with spiking neural network

## Abstract
While neural network models have been extensively compressed to match the stringent edge requirements, spiking neural networks and event-based sensing are recently emerging as promising solutions to further improve performance due to their inherent energy efficiency and capacity to process spatiotemporal data in very low latency. This work aims to evaluate the effectiveness of spiking neural networks on neuromorphic processors in human activity recognition for wearable applications. The case of workout recognition with wrist-worn wearable motion sensors is used as a study. A multi-threshold delta modulation approach is utilized for encoding the input sensor data into spike trains to move the pipeline into the event-based approach. The spikes trains are then fed to a spiking neural network with direct-event training, and the trained model is deployed on the research neuromorphic platform from Intel, Loihi,  to evaluate energy and latency efficiency. Test results show that the spike-based workouts recognition system can achieve a comparable accuracy (87.5\%) comparable to the popular milliwatt RISC-V bases multi-core processor GAP8 with a traditional neural network ( 88.1\%) while achieving two times better energy-delay product (0.66 \si{\micro\joule\second} vs. 1.32 \si{\micro\joule\second}). 

## Dataset, Training and deployment
To train the spiking RecGym dataset, please first download the spiking dataset as indicated, and install the SLAYER framework (https://github.com/bamsumit/slayerPytorch).
For deployment of the trained SNN on Loihi, please visit the Intel Neuromorphic Research Community and ask for access to the server.

## citation
To be publihsed
