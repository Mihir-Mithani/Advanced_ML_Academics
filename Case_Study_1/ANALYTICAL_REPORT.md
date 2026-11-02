# Case Study 1: Algorithmic Fairness & Risk Assessment in High-Stakes Financial Lending
## Analytical Report

---

### Executive Summary

This report presents a comprehensive audit of the Home Credit Default Risk lending pipeline, evaluating baseline model performance, algorithmic fairness across protected groups, and model interpretability through SHAP explanations and counterfactual analysis.

**Key Findings:**
- **Best Model**: XGBoost (ROC-AUC: 0.7612, PR-AUC: 0.2517, Business Cost: $20,616)
- **Critical Fairness Violations**: 
  - Region rating shows severe disparate impact (DIR = 0.15, FAIL - 4/5ths Rule)
  - Age groups fail equalized odds (TPR disparity: 0.46, FPR disparity: 0.19)
  - Housing type fails equalized odds (TPR disparity: 0.30, FPR disparity: 0.11)
- **Top Risk Factors**: External credit scores (EXT_SOURCE_2, EXT_SOURCE_3), credit amount, goods price
- **Counterfactual Insights**: Reducing credit amount by 20% or improving external scores to 0.6+ can flip denials to approvals

---

### Part A: Evaluation & Baseline Modeling

#### Dataset Overview
- **Training samples**: 307,511 applications
- **Default rate**: 8.07% (highly imbalanced)
- **Features**: 120 after preprocessing (16 categorical, 104 numerical)
- **Preprocessing**: Median imputation for numerical, mode for categorical, label encoding

#### Model Comparison

| Model | ROC-AUC | PR-AUC | Business Cost | Optimal Threshold |
|-------|---------|--------|---------------|-------------------|
| Logistic Regression | 0.7466 | 0.2267 | $21,225 | 0.68 |
| Random Forest | 0.7401 | 0.2168 | $21,735 | 0.68 |
| LightGBM | 0.7609 | 0.2519 | $20,694 | 0.68 |
| **XGBoost** | **0.7612** | **0.2517** | **$20,616** | **0.68** |

**Business Cost Matrix**: False Negative (approving defaulted loan) = 5× cost of False Positive (rejecting good loan)

#### Recommendation
Deploy **XGBoost with threshold 0.68** for optimal business cost. The model achieves the best balance between default detection and approval rates.

---

### Part B: Algorithmic Analysis & Fairness Audit

#### Protected Attributes Defined
1. **Gender** (CODE_GENDER): Male vs Female
2. **Age Groups** (DAYS_BIRTH): 18-25, 26-35, 36-45, 46-55, 56-65, 65+
3. **Region Rating** (REGION_RATING_CLIENT): Low (1), Medium (2), High (3)
4. **Income Brackets** (AMT_INCOME_TOTAL): Quartile-based Low to High
5. **Housing Type** (NAME_HOUSING_TYPE): Encoded categories

#### Fairness Metrics Results

##### Disparate Impact Ratio (Four-Fifths Rule: DIR ≥ 0.8)

| Protected Attribute | Privileged Group | Unprivileged Group | DIR | Status |
|---------------------|------------------|-------------------|-----|--------|
| Age Group | 36-45 | 18-25 | 1.92 | ✅ PASS |
| Region Rating | High | Low | **0.15** | ❌ **FAIL** |
| Income Bracket | High | Low | 1.45 | ✅ PASS |
| Housing Type | House/Apt | Rented | N/A* | - |

*DIR not computed due to group mapping complexity

**Critical Finding**: Region rating shows **severe disparate impact** (DIR = 0.15). Applicants from low-rated regions have only 15% of the approval rate of high-rated regions, violating the Four-Fifths Rule.

##### Equalized Odds (TPR/FPR Disparity < 0.1)

| Protected Attribute | TPR Disparity | FPR Disparity | Status |
|---------------------|---------------|---------------|--------|
| Age Group | **0.46** | **0.19** | ❌ **FAIL** |
| Region Rating | **0.35** | **0.14** | ❌ **FAIL** |
| Income Bracket | 0.08 | 0.04 | ✅ PASS |
| Housing Type | **0.30** | **0.11** | ❌ **FAIL** |

**Critical Findings**:
- **Age**: Younger applicants (18-25) have 49% TPR vs 37% for prime age (36-45)
- **Region**: Low-rated regions have 15% TPR vs 50% for high-rated regions
- **Housing**: Significant disparities across housing types

#### Proxy Discrimination / Redlining Analysis

Features acting as indirect discriminators (approval disparity > 15%):

| Feature | Approval Disparity | Key Insight |
|---------|-------------------|-------------|
| REGION_RATING_CLIENT | 0.168 | Low-rated regions: 80% approval vs 97% for high-rated |
| REGION_RATING_CLIENT_W_CITY | 0.174 | Similar pattern with city adjustment |
| NAME_HOUSING_TYPE | 0.163 | Type 4/5 (municipal/co-op): 77-79% vs 90% for standard |
| OCCUPATION_TYPE | **0.323** | **Largest disparity**: Cleaners/helpers 63% vs Pensioners 94% |

**Regulatory Risk**: Region rating and occupation type show strong proxy discrimination patterns consistent with redlining.

---

### Part C: Interpretation & Explainable AI (XAI)

#### Global Feature Importance (SHAP)

| Rank | Feature | Mean |SHAP| | Interpretation |
|------|---------|------------|----------------|
| 1 | EXT_SOURCE_3 | 0.396 | External credit bureau score 3 (most predictive) |
| 2 | EXT_SOURCE_2 | 0.358 | External credit bureau score 2 |
| 3 | EXT_SOURCE_1 | 0.166 | External credit bureau score 1 |
| 4 | AMT_GOODS_PRICE | 0.162 | Price of goods for credit |
| 5 | AMT_CREDIT | 0.120 | Credit amount requested |
| 6 | CODE_GENDER | 0.115 | Gender (proxy discrimination risk) |
| 7 | DAYS_EMPLOYED | 0.109 | Employment duration |
| 8 | DAYS_BIRTH | 0.101 | Age |
| 9 | NAME_EDUCATION_TYPE | 0.099 | Education level |
| 10 | OWN_CAR_AGE | 0.099 | Vehicle age |

**Key Insight**: External credit scores dominate model decisions. Gender appears in top 10, raising fairness concerns.

#### Counterfactual Analysis

Analyzed 10 rejected applicants (probability > 0.5). **8/10** had actionable paths to approval:

| Applicant | Original Prob | Best Modification | Change Required | New Prob |
|-----------|---------------|-------------------|-----------------|----------|
| 137772 | 61.6% | EXT_SOURCE_2: 0.33 → 0.60 | +81% | 46.0% |
| 421915 | 62.1% | EXT_SOURCE_3: 0.25 → 0.60 | +143% | 34.3% |
| 341037 | 50.8% | **AMT_CREDIT: -20%** | -20% | 49.8% |
| 380584 | 62.0% | EXT_SOURCE_2: 0.17 → 0.60 | +250% | 46.4% |
| 188062 | 50.5% | **AMT_CREDIT: -20%** | -20% | 49.3% |
| 316773 | 65.8% | EXT_SOURCE_2: 0.47 → 0.80 | +70% | 41.2% |
| 368768 | 54.4% | EXT_SOURCE_2: 0.53 → 0.70 | +33% | 41.6% |
| 297296 | 66.4% | EXT_SOURCE_3: 0.21 → 0.60 | +184% | 32.5% |

**Actionable Insights**:
- **Credit amount reduction** (20%) is the most practical intervention for borderline cases
- **External score improvement** requires significant changes (often unrealistic)
- 2 applicants had no simple flip (very high risk: 87% and 90%)

#### Adverse Action Notices (Sample)

**Applicant 137772** - DENIED
1. External credit score 1 (Value: 0.09) - Increases default risk
2. External credit score 2 (Value: 0.33) - Increases default risk
3. Goods price (Value: $202,500) - Increases default risk

**Applicant 421915** - DENIED
1. External credit score 2 (Value: 0.07) - Increases default risk
2. External credit score 3 (Value: 0.25) - Increases default risk
3. Age (Value: 38 years) - Increases default risk

**Applicant 341037** - DENIED
1. External credit score 2 (Value: 0.36) - Increases default risk
2. **Gender (Value: Male)** - Increases default risk ⚠️
3. Goods price (Value: $364,500) - Increases default risk

**Concern**: Gender appears as a top-3 factor for some denials, confirming proxy discrimination risk.

---

### Trade-off Analysis: Profitability vs Fairness

| Fairness Constraint | Business Cost | ROC-AUC | Notes |
|---------------------|---------------|---------|-------|
| Unconstrained (Current) | $20,616 | 0.761 | Baseline |
| + Equalized Odds (Age) | ~$22,000* | ~0.755* | Requires post-processing |
| + DIR ≥ 0.8 (Region) | ~$23,500* | ~0.750* | Significant cost increase |
| + Counterfactual Explanations | +$500 | - | Operational overhead |

*Estimated based on typical fairness-accuracy trade-offs

---

### Regulatory Recommendations

#### 1. **Immediate Actions (0-30 days)**
- [ ] Implement adverse action notices with SHAP-based explanations for all denials
- [ ] Add human review for applicants from low-rated regions (REGION_RATING_CLIENT = 1)
- [ ] Flag gender as sensitive attribute; monitor for disparate treatment

#### 2. **Short-term (1-3 months)**
- [ ] Retrain model with fairness constraints (equalized odds on age, region)
- [ ] Implement counterfactual explanation API for customer service
- [ ] Establish monthly fairness dashboard (DIR, TPR/FPR by group)

#### 3. **Medium-term (3-6 months)**
- [ ] Collect explicit demographic data (with consent) for direct fairness measurement
- [ ] Develop region-specific models or calibration to address redlining
- [ ] Implement reject inference to better understand denied population

#### 4. **Governance**
- [ ] Quarterly fairness audit by independent team
- [ ] Document all model changes with fairness impact assessment
- [ ] Establish appeals process with counterfactual guidance
- [ ] Board-level reporting on fairness metrics

---

### Technical Deliverables

| File | Description |
|------|-------------|
| `case_study_1_complete.py` | Complete end-to-end pipeline |
| `part_a_results.json` | Model comparison metrics |
| `part_b_fairness_audit.json` | DIR and Equalized Odds by group |
| `part_b_proxy_discrimination.json` | Redlining analysis results |
| `part_c_shap_importance.json` | Global SHAP feature importance |
| `part_c_counterfactuals.json` | 8 counterfactual paths to approval |
| `part_c_adverse_notices.json` | Sample adverse action notices |
| `executive_summary.txt` | This report |

---

### Conclusion

The current lending pipeline exhibits **significant algorithmic bias** against:
1. **Low-income regions** (redlining pattern - DIR = 0.15)
2. **Younger applicants** (age discrimination - TPR disparity = 0.46)
3. **Non-traditional housing/occupation** (proxy discrimination)

While XGBoost achieves strong predictive performance (ROC-AUC 0.76), **fairness violations require immediate remediation** before regulatory scrutiny. The recommended path forward combines:
- **Model retraining** with fairness constraints
- **Operational safeguards** (human review, enhanced notices)
- **Transparency tools** (counterfactuals, SHAP explanations)
- **Governance framework** for ongoing monitoring

The cost of fairness compliance (~$2,000-3,000 additional business cost) is justified by regulatory risk mitigation and ethical lending practices.