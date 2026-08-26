# ARTEMIS Lunar Plasma Analysis

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)

Imperial College London MSci Physics project.

**Working title:**
*Plasma physics controlling the Moon's interaction with the solar wind near crustal magnetic field anomalies.*

## Overview
This project investigates how the solar wind interacts with lunar crustal magnetic anomalies using data from NASA's ARTEMIS mission.

This repository contains tools for:
- Downloading ARTEMIS spacecraft data
- Processing magnetic field and plasma measurements
- Investigating lunar crustal magnetic anomalies
- Searching for signatures of mini-magnetospheres
- Exploring potential magnetic reconnection events

Data processing is performed using PySPEDAS.

Authors:
- Yulun Wu
- Margot Lippold

Supervisor:
- Jonathan Eastwood

## Installation

Clone the repo:

```bash
git clone https://github.com/your_username/artemis-lunar-plasma.git
cd artemis-lunar-plasma
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install PySPEDAS:

```bash
pip install pyspedas
```
---

## Current Progress

- [x] Repository created
- [x] GitHub setup
- [x] Virtual environment configured
- [x] Install PySPEDAS
- [x] Download first FGM dataset
- [ ] Plot IMF orientation
- [ ] Identify lunar crossings
- [ ] Search for magnetic pile-up events
- [ ] Investigate reconnection candidates

## Useful Resources

### Project Notes
- OneNote: https://d.docs.live.net/847A08EEB93FD415/Documentos/SPC-Eastwood-2/

### Literature
- Zotero library: https://www.zotero.org/groups/6643888/artemis-lunar-plasma

## License

For academic and research use.
