# JP_Morgan_Projects
# Natural Gas Pricing and Storage Contract Valuation

## Project Overview

This repository contains two integrated Python modules for:
1. **Natural Gas Price Forecasting** (P1)
2. **Storage Contract Valuation** (P2)

Developed for energy trading desks to evaluate natural gas storage arbitrage opportunities by analyzing historical prices and projecting future contract values.

## Repository Structure

JP_Morgan_Projects/
├── P1/
│ ├── NatGasAnalyzer.py # Price forecasting model
│ ├── Nat_Gas.csv # Sample price data
│ └── tests/ # Unit tests planned
├── P2/
│ ├── ContractValue.py # Storage contract valuation
│ ├── Nat_Gas.csv # Sample price data
│ └── tests/ # Unit tests planned
└── README.md # This document


## Task 1: Natural Gas Price Forecasting (P1)

### Features
- Historical price analysis and visualization
- Seasonal decomposition (trend, seasonality, residuals)
- Polynomial regression model with monthly seasonality
- Future price forecasting (12+ months)

## Task 2: Storage Contract Valuation (P2)

### Features
- Multi-period injection/withdrawal scheduling
- Comprehensive cost accounting:
- Storage fees
- Injection/withdrawal costs
- Transportation fees
- Capacity constraint validation
- Professional valuation reporting