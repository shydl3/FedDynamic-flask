# FedDynamic

## Overview
Course project for CS230 implementing a federated learning system with dynamic client participation.

## System Architecture
- **Server**: Central aggregation point
- **Clients**: 3 distributed nodes sharing the same codebase but with different instance files
  - `client1.py`
  - `client2.py`
  - `client3.py`

## Installation

```bash
# Install required dependencies
pip install flwr torch torchvision
```bash

## Running the Project

# server
Start the central server:

```bash
python server.py
```bash

## Clients
Run each client in separate terminals:

```bash
# Terminal 1
python client1.py

# Terminal 2
python client2.py

# Terminal 3
python client3.py
```bash

## Project Structure
server.py: Implements the federated server logic
client1.py, client2.py, client3.py: Client implementations with identical code but different instance files

