# Dataset Documentation

## Overview
- **File**: `P2.csv`
- **Origin**: Corporate Bankruptcy / Financial Distress Dataset (Taiwan Economic Journal, 1999–2009).
- **Records (Rows)**: 6,819 companies.
- **Attributes (Columns)**: 95 financial ratio features + 1 binary target.
- **Target Variable**: `Bankrupt?`
  - `0`: Solvent / Non-bankrupt (6,599 instances, ~96.77%)
  - `1`: Bankrupt (220 instances, ~3.23%)

## Key Feature Categories
1. **Profitability Margins**: ROA(A), ROA(B), ROA(C), Operating Gross Margin, Realized Sales Gross Margin, Net Profit Margin.
2. **Cash Flow & Liquidity**: Cash Flow to Total Assets, Quick Ratio, Current Ratio, Cash Flow to Sales, Cash Turnover Rate.
3. **Solvency & Capital Structure**: Debt Ratio %, Net Worth to Assets, Borrowing Dependency, Liability to Equity, Total Debt to Net Worth.
4. **Operating Efficiency & Turnover**: Inventory Turnover, Accounts Receivable Turnover, Total Asset Turnover, Fixed Assets Turnover.
5. **Growth Indicators**: Total Asset Growth Rate, Net Value Growth Rate, Operating Profit Growth Rate.

## Notes
- All monetary metrics are normalized and expressed as standardized ratios.
- The dataset is cross-sectional and lacks temporal/timestamp dimensions.
