#####################################################################
# IMP
# Generate data
# Javier Osorio 
# 8-01-2025
#####################################################################



# SETUP --------------------------------------------------

# Load all packages here

if (!require("pacman")) install.packages("pacman")
pacman::p_load(here, glue, openxlsx, xtable, tidyverse, readxl, dplyr, stringr, stringi, ggplot2, grid)




# GET THE DATA  --------------------------------------------------

## Predictions data ------------

# Get the preds data
preds <- read.csv('https://raw.githubusercontent.com/javierosorio/devil_in_the_details_mtsummit25/refs/heads/main/data/master_file/merged_Data_Master.csv',  header = TRUE)

# Select data
preds <- preds %>%
  drop_na() %>%
  select(c(id, en, en_Classification, en_Prediction_Score, es_Classification, es_Prediction_Score, ar_Classification, ar_Prediction_Score, es_en_DEEP_Classification, es_en_DEEP_Prediction_Score, es_en_DEEPL_Classification, es_en_DEEPL_Prediction_Score, ar_en_DEEP_Classification, ar_en_DEEP_Prediction_Score, ar_en_DEEPL_Classification, ar_en_DEEPL_Prediction_Score, en_es_DEEP_Classification, en_es_DEEP_Prediction_Score, en_es_DEEPL_Classification, en_es_DEEPL_Prediction_Score, en_ar_DEEP_Classification, en_ar_DEEP_Prediction_Score, en_ar_DEEPL_Classification, en_ar_DEEPL_Prediction_Score, en_ar_TRANSFORMERS_Classification, en_ar_TRANSFORMERS_Prediction_Score, en_es_TRANSFORMERS_Classification, en_es_TRANSFORMERS_Prediction_Score))

# Eliminate duplicates
preds <- preds %>% distinct(id, .keep_all = TRUE)

# Get variable names
#vars <- names(preds) %>% as.data.frame()
#write.xlsx(vars, "1_data/vars.xlsx")




## Text data ------------

# Get the text data
text <- read.csv('1_data/text_data_240908.csv')

# Drop NAs and duplicates
text <- text %>% 
  drop_na() %>%
  distinct(id, .keep_all = TRUE) %>%
  rename(num = X)

# Get variable names
#vars2 <- names(text) %>% as.data.frame()
#write.xlsx(vars2, "1_data/vars2.xlsx")




## Merge data ------------

# Merge 
data <- left_join(text, preds, by="id", keep = NULL)

# Fix names
data <- data %>%
  rename(en = en.x) %>%
  select(-en.y)

# Check names
names(data)

# Get variable names
#vars3 <- names(data) %>% as.data.frame()
#write.xlsx(vars3, "1_data/vars.xlsx")

# Standardize variable names 
data <- data %>%
  rename(num = num) %>%
  rename(id = id) %>%
  rename(en = en) %>%
  rename(es = es) %>%
  rename(ar = ar) %>%
  rename(es_en_deep = es_en_DEEP) %>%
  rename(es_en_deepl = es_en_DEEPL) %>%
  rename(es_en_gt = es_en_GOOGLE) %>%
  rename(es_en_opus = es_en_TRANSOFMERS) %>%
  rename(ar_en_deep = ar_en_DEEP) %>%
  rename(ar_en_deepl = ar_en_DEEPL) %>%
  rename(ar_en_gt = ar_en_GOOGLE) %>%
  rename(ar_en_opus = ar_en_TRANSFORMERS) %>%
  rename(en_es_deep = en_es_DEEP) %>%
  rename(en_es_deepl = en_es_DEEPL) %>%
  rename(en_es_gt = en_es_GOOGLE) %>%
  rename(en_es_opus = en_es_TRANSFORMERS) %>%
  rename(en_ar_deep = en_ar_DEEP) %>%
  rename(en_ar_deepl = en_ar_DEEPL) %>%
  rename(en_ar_gt = en_ar_GOOGLE) %>%
  rename(en_ar_opus = en_ar_TRANSFORMERS) %>%
  rename(matconf = MatConf) %>%
  rename(matcoop = MatCoop) %>%
  rename(verconf = VerConf) %>%
  rename(vercoop = VerCoop) %>%
  rename(relevant = Relevant) %>%
  rename(quadclass = QuadClass) %>%
  rename(en_relevant_class = en_Classification) %>%
  rename(en_relevant_pred = en_Prediction_Score) %>%
  rename(es_relevant_class = es_Classification) %>%
  rename(es_relevant_pred = es_Prediction_Score) %>%
  rename(ar_relevant_class = ar_Classification) %>%
  rename(ar_relevant_pred = ar_Prediction_Score) %>%
  rename(es_en_deep_relevant_class = es_en_DEEP_Classification) %>%
  rename(es_en_deep_relevant_pred = es_en_DEEP_Prediction_Score) %>%
  rename(es_en_deepl_relevant_class = es_en_DEEPL_Classification) %>%
  rename(es_en_deepl_relevant_pred = es_en_DEEPL_Prediction_Score) %>%
  rename(ar_en_deep_relevant_class = ar_en_DEEP_Classification) %>%
  rename(ar_en_deep_relevant_pred = ar_en_DEEP_Prediction_Score) %>%
  rename(ar_en_deepl_relevant_class = ar_en_DEEPL_Classification) %>%
  rename(ar_en_deepl_relevant_pred = ar_en_DEEPL_Prediction_Score) %>%
  rename(en_es_deep_relevant_class = en_es_DEEP_Classification) %>%
  rename(en_es_deep_relevant_pred = en_es_DEEP_Prediction_Score) %>%
  rename(en_es_deepl_relevant_class = en_es_DEEPL_Classification) %>%
  rename(en_es_deepl_relevant_pred = en_es_DEEPL_Prediction_Score) %>%
  rename(en_ar_deep_relevant_class = en_ar_DEEP_Classification) %>%
  rename(en_ar_deep_relevant_pred = en_ar_DEEP_Prediction_Score) %>%
  rename(en_ar_deepl_relevant_class = en_ar_DEEPL_Classification) %>%
  rename(en_ar_deepl_relevant_pred = en_ar_DEEPL_Prediction_Score) %>%
  rename(en_ar_opus_relevant_class = en_ar_TRANSFORMERS_Classification) %>%
  rename(en_ar_opus_relevant_pred = en_ar_TRANSFORMERS_Prediction_Score) %>%
  rename(en_es_opus_relevant_class = en_es_TRANSFORMERS_Classification) %>%
  rename(en_es_opus_relevant_pred = en_es_TRANSFORMERS_Prediction_Score) 
  
# 
data <- data %>%
  relocate(c(num, id, en, es, ar, es_en_deep, es_en_deepl, es_en_gt, es_en_opus, ar_en_deep, ar_en_deepl, ar_en_gt, ar_en_opus, 
             en_es_deep, en_es_deepl, en_es_gt, en_es_opus, en_ar_deep, en_ar_deepl, en_ar_gt, en_ar_opus, matconf, matcoop, 
             verconf, vercoop, relevant, quadclass, en_relevant_class, es_relevant_class, ar_relevant_class, en_relevant_pred, 
             es_relevant_pred, ar_relevant_pred, es_en_deep_relevant_class, es_en_deepl_relevant_class, ar_en_deep_relevant_class, 
             ar_en_deepl_relevant_class, en_es_deep_relevant_class, en_es_deepl_relevant_class, en_es_opus_relevant_class, 
             en_ar_deep_relevant_class, en_ar_deepl_relevant_class, en_ar_opus_relevant_class, es_en_deep_relevant_pred, 
             es_en_deepl_relevant_pred, ar_en_deep_relevant_pred, ar_en_deepl_relevant_pred, en_es_deep_relevant_pred, 
             en_es_deepl_relevant_pred, en_es_opus_relevant_pred, en_ar_deep_relevant_pred, en_ar_deepl_relevant_pred, 
             en_ar_opus_relevant_pred))

# Check names
names(data)



## Select data for IMP paper  ------------
data <- data %>%
  select(c(num, id, en, es, ar,                                     # Text vars
           es_en_deepl, ar_en_deepl,                                # MT with DeepL
           relevant, quadclass, matconf, matcoop, verconf, vercoop, # Manual classifications
           en_relevant_class,                                       # NST Relevant classification with ConfliBERT-uncased model
           es_en_deepl_relevant_class, ar_en_deepl_relevant_class,  # DeepL Relevant classification with ConfliBERT-uncased model
           en_relevant_pred,                                        # NST Relevant prediction with ConfliBERT-uncased model
           es_en_deepl_relevant_pred, ar_en_deepl_relevant_pred))   # DeepL Relevant prediction with ConfliBERT-uncased model


# Check nanes
names(data)


## Export clean data frame  ------------
write.table(data, file=glue('1_data/data_master_250831.tsv'), sep='\t', row.names = FALSE)
write.csv(data,   file=glue('1_data/data_master_250831.csv', row.names=FALSE))
write.xlsx(data,  file=glue('1_data/data_master_250831.xlsx'))



# Remove working data
rm(preds, text)


# End of script