## Medical_Device_Failure
###### Disclaimer: `Model Architecture`, `Problem Statement` and `Data Glossary` have been adjusted to protect company's IP. This actual project was carried out in July 2020
##### Problem Statement
Non-routine quality events, like material failures, cost the medical device industry between US $2 - 5 billion per year on average.Materials-related issues are responsible for >30% of medical device failures, In 2017, about 35% of medical device failures reported to the Food and Drug Administration (FDA) were materials-related, another 5% was due to electronic failures. A medical device material failure makes them prone to errors and this can be traced back to the materials used to develop the medical device.Many medical devices can be affected by a material failure, and failure can during the course of a medical device lifecycle.

In addition to financial losses, material failures can lead to:
- Litigation and judgments
- Product recalls
- Failed regulatory requirements
- Warranty claims
- Lost inventory
- Lost future sales
- Reputational damage
  
Material failures are costly at any point in the product life cycle, but the consequences get more and more costly the further along the product is. Early design changes cost less than redesigning a product in the late development stage, but both of those are far less costly than a post-market failure.

###### Objective

- Exploratory Data Analysis : Run a descriptive analysis to gain preliminary insights in medical device failure

- Statistical Analysis: Hypothesis test - Answer the question is there statistical difference in the internal velocity of the 4 Gear Types
  
- Machine Learning: For this medical device, develop a ML model to identify (1) potential failures;(2) key attributes driving these failures;(3) optimize the model performance

##### Data Glossary
The data contains the different features of this medical device:

 - Device_Failure : No failure (N), Failure (Yes) [Target Variable]

- Gear_Type : There are 4 Gear Types T, X, Y,and Z [Categorical]

- Velocity: Internal velocity of rotor measure in rpm [Continuous]

- Torque(Nm) : device spindle measured in Nm [Continuous]

- Wear_Measure: Wear measure in micrometers [Continuous]

- Sensor_1: Proprietary Resitive Sensor (upper) [Continuous]

- Sensor_2 : Proprietary Resitive Sensor (Lower) [Continuous]

#####
For each objective a `.py` and a corresponding `.ipynb` is provided for code development
- `Medical_Device_Failure_Exploratory_Data_Analysis.py` and `Medical_Device_Failure_Exploratory_Data_Analysis.ipynb`
- `Medical_Device_Failure_Statistical_Analysis.py` and `Medical_Device_Failure_Statistical_Analysis.ipynb`
- `Medical_Device_Failure_MachineLearning.py` and `Medical_Device_Failure_MachineLearning.ipynb`
