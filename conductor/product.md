# Initial Concept

Create new project in golang that will implement hotword detection, training from wav samples, and use as little dependencies as it can. It can be rnn or cnn deep network.

# Product Guide - Hotword Detection in Go

## Purpose
A lightweight, low-dependency hotword detection system written in Go, specifically designed for hobbyists building DIY smart home or embedded voice assistants on small embedded Linux systems like Raspberry Pi.

## Target Users
- Hobbyists and makers building voice-controlled devices.
- Developers needing a portable, Go-native audio triggering solution.

## Goals
- **Minimal Dependencies:** Native Go implementation to ensure high portability and simple "go build" deployment.
- **Accuracy & Reliability:** High detection performance with low false-positive rates, even in noisy environments.
- **Resource Efficiency:** Low latency and minimal memory footprint, optimized for real-time performance on resource-constrained hardware.

## Key Features
- **Custom Training Pipeline:** Train on custom hotwords using a directory-based WAV dataset (e.g., 'hotword/' and 'background/' folders).
- **Flexible Architecture:** Support for CNN (Convolutional Neural Network) and RNN/LSTM (Recurrent Neural Network) deep learning models.
- **Real-time Inference:** Efficient processing of live audio streams with built-in Voice Activity Detection (VAD) to minimize CPU load during silence.
- **Robustness:** Built-in data augmentation (noise injection) to improve model performance with limited training samples.
- **Deployment:** Exportable model files for easy transfer to target devices.

## Target Hardware
- Small embedded Linux systems (e.g., Raspberry Pi, Raspberry Pi Zero).
