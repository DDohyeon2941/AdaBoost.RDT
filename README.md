# **AdaBoost.RDT: AdaBoost Integrated With Residual-Based Decision Tree for Demand Prediction of Bike Sharing Systems Under Extreme Demands**
 
## **Overview**
AdaBoost.RDT is a novel boosting algorithm that integrates **Adaptive Boosting (AdaBoost) with a Residual-Based Decision Tree (RDT)** to improve demand prediction under extreme fluctuations. Traditional boosting methods tend to **underestimate extreme demand**, leading to performance degradation in high-demand scenarios. AdaBoost.RDT mitigates this by introducing a **residual-based decision tree** to refine predictions, ensuring robustness against extreme values while maintaining accuracy for normal demand.

## **Data**
AdaBoost.RDT was evaluated on **real-world bike-sharing datasets** from Seoul and Daejeon. These datasets contain:
- **Hourly rental demand per station**
- **Weather conditions** (temperature, precipitation, wind speed, etc.)
- **Temporal features** (weekday/weekend, time of day, seasonality)
- **Demographic and infrastructure data** (population density, proximity to public transport, etc.)

The datasets include both **normal demand events** and **extreme demand events**, making them ideal for testing the algorithm’s ability to both extreme and normal events.

## **Methodology**
AdaBoost.RDT consists of the following key components:
- **Adjusted Model Tree (A-MT)**: A modified model tree that selectively applies linear regression to improve predictions for extreme and normal demand events while preventing overfitting to noise (first-stage).
- **Residual-Based Decision Tree (R-DT)**: A second-stage decision tree that predicts residuals from AdaBoost’s base model to refine extreme demand predictions.
- **Adaptive Quantile Adjustment**: Adjusts final predictions based on residuals to prevent systematic underestimation of extreme values.
- **Noise Robustness**: Implements a selective filtering mechanism to prevent overfitting to noise while capturing important demand fluctuations.

The algorithm is compared against standard boosting methods, including:
- **AdaBoost**
- **Gradient Boosting Machine (GBM)**
- **Noise-robust boosting variants** such as Huber Loss Boosting (MBoost) and L1 Loss Boosting (LADBoost).

## **Results**
Experimental results demonstrate that AdaBoost.RDT:
- **Reduces underestimation errors** for extreme demand cases.
- **Maintains accuracy for normal demand events**, unlike traditional methods that overfit to extreme values.
- **Improves overall robustness**, outperforming AdaBoost, GBM, and other noise-robust boosting models in extreme event prediction.

## **Contributing**
We welcome contributions to this project. Please submit pull requests or open issues for any bugs or enhancements.

## **Citation**
For more details, please refer to the full paper on [IEEE](https://ieeexplore.ieee.org/document/10705154).
