# DATA

<br>

---

**NOTE TO THE TEAM** If you produce new results at the sentence level, please make sure to include the `num` or `id` variable so that we can merge them to the master data. 

<br>

---
This folder contains two key files:

* File `data_master_250831.csv` is the main dataset containing all the raw data used in this project.
* File `var_list_data_master_250831.xlsx` contains a list of all the variable names and indicates their characteristics.

<br>

The data master includes the following types of variables:

* Language:
   * Native: English (EN), Spanish (ES), and Arabic (AR)
   * Machine Translation (MT) using the DEEPL tool from English to Spanish (EN_ES), English to Arabic (EN_AR)
* ConfliBERT classification:
   * Classification binary: relevant to CAMEO Quadclass (1), not relevant (0) using ConfliBERT-uncased (based on MT Summit paper)
   * Prediction binary: Prediction of ConfliBERT getting binary classification correct using ConfliBERT-uncased (based on MT Summit paper)
   

---

**If you add more variables, please document your changes in this section.**

**Make sure you merge the new variables by merging the new data using the  `num` or `id` variable.**



