# **Analysis of Autism Longitudinal Data Using Mixed Linear Models**

## **Abstract**
This project analyzes longitudinal data from a University of Michigan study of 155 children followed at approximately ages 2, 3, 5, 9, and 13. The primary outcome is the Vineland Socialization Age Equivalent (VSAE), a parent‑reported measure of social development. Using mixed linear models with random intercepts and slopes, the analysis evaluates how age, language proficiency at age 2, diagnostic category (autism vs. PDD), race, and their interactions predict socialization trajectories over time. Missingness was substantial—only 41 of 155 children completed all five visits—which influenced model precision. Despite this limitation, age, language, diagnosis, and race (and their interactions with age) emerged as significant predictors of VSAE. 

---

## **Dataset Overview**
The dataset includes:
- **155 children**, each assigned to *autism* or *PDD* at age 2  
- Up to **5 longitudinal visits** (ages ~2, 3, 5, 9, 13)  
- Key variables:  
  - VSAE (socialization score)  
  - Age  
  - Language proficiency (low/medium/high)  
  - Diagnosis (autism/PDD)  
  - Race (white/non‑white)  
  - Gender  

Only **26.45%** of children completed all five visits, and missingness varied by visit. 

---

## **Research Questions**
1. Does language proficiency at age 2 predict VSAE trajectories?  
2. Do autism vs. PDD diagnoses predict different developmental trajectories?  
3. Does race predict VSAE over time?  
4. Is gender predictive of VSAE?  
5. Are any additional variables predictive of VSAE?  


---

## **Methods**
### **Modeling Approach**
A **mixed linear effects model** with random intercepts and random slopes was selected because:
- The dataset contains **unbalanced longitudinal data** with substantial missingness, assumed missing at random.  
- Mixed models handle MAR data more effectively than GEE.  
- GEE with an exponential correlation structure was not feasible in standard R/SAS implementations.  


### **Model Building**
- Individual predictors and interaction terms were tested using ANOVA.  
- Significant predictors were retained in the final model.  
- Final predictors included:  
  - Age  
  - Language  
  - Diagnosis  
  - Race  
  - Age × Language  
  - Age × Diagnosis  
  - Age × Race  


---

## **Key Findings**
### **Significant Predictors**
- **Language proficiency**: Medium and high groups showed different trajectories.  
- **Diagnosis**: PDD vs. autism produced significantly different slopes.  
- **Race**: White vs. non‑white showed significant differences in both intercept and slope.  
- **Age interactions**: All three (language, diagnosis, race) interacted significantly with age.  


### **Non‑significant Predictors**
- **Gender** was not predictive of VSAE.  


---

## **Final Model Summary**
### **Fixed Effects (selected)**
- Intercept: **5.521**  
- Age: **1.368**  
- Race (White): **–4.645** (p = 0.02)  
- Age × Language (High): **3.824** (p < 0.0001)  
- Age × Diagnosis (PDD): **1.7** (p = 0.021)  
- Age × Race (White): **1.809** (p = 0.013)  


### **Random Effects**
- Subject‑specific intercept: **8.361**  
- Subject‑specific age slope: **3.819**  


---

## **Interpretation**
- Children with higher language proficiency at age 2 show **steeper increases** in VSAE over time.  
- Children diagnosed with PDD show **more rapid improvement** than those with autism.  
- Race differences persist both at baseline and in growth rate.  
- Gender does not meaningfully influence socialization development.  


---

## **Limitations**
- High missingness (only 41 children with complete data) likely increased model error.  
- Missingness patterns were not fully modeled, limiting causal interpretation.  


---

## **Reproducibility**
This repository includes:
- Data cleaning scripts  
- Mixed model code (R or SAS)  
- Visualization scripts  
- Model output tables  

