# Petroleum Production Forecasting & Economics  
Decline Curve Analysis (DCA) + Economic Evaluation + Sensitivity + Power BI

This project applies **Arps Decline Curve Analysis (DCA)** on real production data from the **Volve Field (Equinor Open Data)** to forecast oil recovery and evaluate project economics.  
It integrates reservoir engineering methods with petroleum economics and visualization tools (Excel, Python, Power BI).

##  Project Highlights
- Fit **Arps decline model** to Volve well data (NO 15/9-F-12 H).  
- Forecast oil production until **economic limit = 50 bbl/day**.  
- Calculate EUR (Estimated Ultimate Recovery).  
- Build cashflow model with CAPEX, OPEX, oil price, discount rate.  
- Compute key economic indicators: NPV, IRR, Payback, PI, Breakeven price.  
- Run sensitivity analysis: NPV vs Oil Price, OPEX, Discount Rate.  
- Present results in an **interactive Power BI dashboard** and **PowerPoint presentation**.

##  Repository Contents
- DCA econo.py → Python script for Arps decline fitting & forecasting.  
- Dca Of well.xlsx → Cleaned well production dataset.  
- Dca economics excel.xlsx → Economics workbook with cashflows.  
- DCA_Economics_Project_Praveen.pptx → Final presentation with charts & results.  
- .png → Plots (forecast, cumulative, sensitivities).  
- README.md → This file.  

## Methods & Formulas
### Decline Curve Analysis (Arps, 1945)
- (q_i): Initial rate (bbl/day)  
- (D_i): Nominal decline rate (1/year)  
- (b): Decline exponent (0 = exponential, 1 = harmonic)  

### Economics
- Revenue = Oil (bbl) × Price ($/bbl)  
- OPEX = Variable ($/bbl) + Fixed ($/yr ÷ 12)  
- Net CF = Revenue − OPEX − CAPEX (at t=0)  
- Discounted CF = Net CF ÷ (1+r)^(t/12)  
- NPV = Σ Discounted CF  
- IRR = discount rate where NPV=0  
- PI = (NPV + CAPEX) ÷ CAPEX  
- Payback = time when cumulative CF > 0  

##  Key Results (Base Case)
- **Well:** NO 15/9-F-12 H (Volve field)  
- **Fitted Arps parameters:**  
  - qi ≈ 32,028 bbl/day  
  - Di ≈ 0.345 year⁻¹  
  - b ≈ 0 (exponential)  
- **Forecast Horizon:** ~18.75 years  
- **EUR:** ~33.9 million bbl  

**Economics (CAPEX $100M, OPEX $10/bbl + $1M/yr, Oil $70/bbl, Discount 10%):**
- NPV ≈ **$1.49 Billion**  
- IRR ≈ **20,774%**  
- Payback ≈ **0.08 years (~1 month)**  
- PI ≈ **15.9**  
- Breakeven Oil Price ≈ **$14.1/bbl**

##  Sensitivity Analysis
- **NPV vs Oil Price:** $40 → ~$0.12B, $100 → ~$3.9B.  
- **NPV vs OPEX:** $5/bbl → high NPV, $20/bbl → much lower.  
- **NPV vs Discount Rate:** 5% → very high NPV, 20% → reduced but still positive.  

##  How to Run
Calculate this Excel file
then python file for plots and approximation
Run DCA + economics:
python "DCA econo.py"

Outputs:
Excel with forecasts & economics.
PNG plots.
PPTX presentation.
Power BI input tables.
