#####################################################################
# IMP
# Generate data
# Javier Osorio 
# 8-13-2025
#####################################################################



# SETUP --------------------------------------------------

# Load all packages here

if (!require("pacman")) install.packages("pacman")
pacman::p_load(here, glue, openxlsx, xtable, tidyverse, readxl, dplyr, stringr, stringi, ggplot2, grid)


# Get the data
data <- read.csv('https://raw.githubusercontent.com/javierosorio/devil_in_the_details_mtsummit25/refs/heads/main/data/master_file/merged_Data_Master.csv',  header = TRUE)


# Get variables
vars <- names(data) %>% as.data.frame()

# Export var names
#write.xlsx(vars, "1_data/vars.xlsx")


# Exclude diff variables.  They are too messy and do not add much
data <- data %>%
  select(-c(contains('_diff')))

# Get variables
vars <- names(data) %>% as.data.frame()


# RENAME VARS --------------------------------------------------

data <- data %>% 
  rename(id = id) %>%
  rename(en_native = en) %>%
  rename(es_native = es) %>%
  rename(ar_native = ar) %>%
  rename(esen_deep = es_en_DEEP) %>%
  rename(esen_deepl = es_en_DEEPL) %>%
  rename(esen_gt = es_en_GOOGLE) %>%
  rename(esen_opus = es_en_TRANSOFMERS) %>%
  rename(aren_deep = ar_en_DEEP) %>%
  rename(aren_deepl = ar_en_DEEPL) %>%
  rename(aren_gt = ar_en_GOOGLE) %>%
  rename(aren_opus = ar_en_TRANSFORMERS) %>%
  rename(enes_deep = en_es_DEEP) %>%
  rename(enes_deepl = en_es_DEEPL) %>%
  rename(enes_gt = en_es_GOOGLE) %>%
  rename(enes_opus = en_es_TRANSFORMERS) %>%
  rename(enar_deep = en_ar_DEEP) %>%
  rename(enar_deepl = en_ar_DEEPL) %>%
  rename(enar_gt = en_ar_GOOGLE) %>%
  rename(enar_opus = en_ar_TRANSFORMERS) %>%
  rename(en_native_bin_class = en_Classification) %>%
  rename(es_native_bin_class = es_Classification) %>%
  rename(ar_native_bin_class = ar_Classification) %>%
  rename(esen_deep_bin_class = es_en_DEEP_Classification) %>%
  rename(esen_deepl_bin_class = es_en_DEEPL_Classification) %>%
  rename(esen_gt_bin_class = es_en_GOOGLE_Classification) %>%
  rename(esen_opus_bin_class = es_en_TRANSOFMERS_Classification) %>%
  rename(aren_deep_bin_class = ar_en_DEEP_Classification) %>%
  rename(aren_deepl_bin_class = ar_en_DEEPL_Classification) %>%
  rename(aren_gt_bin_class = ar_en_GOOGLE_Classification) %>%
  rename(aren_opus_bin_class = ar_en_TRANSFORMERS_Classification) %>%
  rename(enes_deep_bin_class = en_es_DEEP_Classification) %>%
  rename(enes_deepl_bin_class = en_es_DEEPL_Classification) %>%
  rename(enes_gt_bin_class = en_es_GOOGLE_Classification) %>%
  rename(enes_opus_bin_class = en_es_TRANSFORMERS_Classification) %>%
  rename(enar_deep_bin_class = en_ar_DEEP_Classification) %>%
  rename(enar_deepl_bin_class = en_ar_DEEPL_Classification) %>%
  rename(enar_gt_bin_class = en_ar_GOOGLE_Classification) %>%
  rename(enar_opus_bin_class = en_ar_TRANSFORMERS_Classification) %>%
  rename(en_native_bin_pred = en_Prediction_Score) %>%
  rename(es_native_bin_pred = es_Prediction_Score) %>%
  rename(ar_native_bin_pred = ar_Prediction_Score) %>%
  rename(esen_deep_bin_pred = es_en_DEEP_Prediction_Score) %>%
  rename(esen_deepl_bin_pred = es_en_DEEPL_Prediction_Score) %>%
  rename(esen_gt_bin_pred = es_en_GOOGLE_Prediction_Score) %>%
  rename(esen_opus_bin_pred = es_en_TRANSOFMERS_Prediction_Score) %>%
  rename(aren_deep_bin_pred = ar_en_DEEP_Prediction_Score) %>%
  rename(aren_deepl_bin_pred = ar_en_DEEPL_Prediction_Score) %>%
  rename(aren_gt_bin_pred = ar_en_GOOGLE_Prediction_Score) %>%
  rename(aren_opus_bin_pred = ar_en_TRANSFORMERS_Prediction_Score) %>%
  rename(enes_deep_bin_pred = en_es_DEEP_Prediction_Score) %>%
  rename(enes_deepl_bin_pred = en_es_DEEPL_Prediction_Score) %>%
  rename(enes_gt_bin_pred = en_es_GOOGLE_Prediction_Score) %>%
  rename(enes_opus_bin_pred = en_es_TRANSFORMERS_Prediction_Score) %>%
  rename(enar_deep_bin_pred = en_ar_DEEP_Prediction_Score) %>%
  rename(enar_deepl_bin_pred = en_ar_DEEPL_Prediction_Score) %>%
  rename(enar_gt_bin_pred = en_ar_GOOGLE_Prediction_Score) %>%
  rename(enar_opus_bin_pred = en_ar_TRANSFORMERS_Prediction_Score) %>%
  rename(en_native_gral_rar_nativeity = en_general_rarity) %>%
  rename(es_native_gral_rar_nativeity = es_general_rarity) %>%
  rename(ar_native_gral_rar_nativeity = ar_general_rarity) %>%
  rename(esen_deep_gral_rarity = es_en_DEEP_general_rarity) %>%
  rename(esen_deepl_gral_rarity = es_en_DEEPL_general_rarity) %>%
  rename(esen_gt_gral_rarity = es_en_GOOGLE_general_rarity) %>%
  rename(esen_opus_gral_rarity = es_en_TRANSFORMERS_general_rarity) %>%
  rename(aren_deep_gral_rarity = ar_en_DEEP_general_rarity) %>%
  rename(aren_deepl_gral_rarity = ar_en_DEEPL_general_rarity) %>%
  rename(aren_gt_gral_rarity = ar_en_GOOGLE_general_rarity) %>%
  rename(aren_opus_gral_rarity = ar_en_TRANSFORMERS_general_rarity) %>%
  rename(enes_deep_gral_rarity = en_es_DEEP_general_rarity) %>%
  rename(enes_deepl_gral_rarity = en_es_DEEPL_general_rarity) %>%
  rename(enes_gt_gral_rarity = en_es_GOOGLE_general_rarity) %>%
  rename(enes_opus_gral_rarity = en_es_TRANSFORMERS_general_rarity) %>%
  rename(enar_deep_gral_rarity = en_ar_DEEP_general_rarity) %>%
  rename(enar_deepl_gral_rarity = en_ar_DEEPL_general_rarity) %>%
  rename(enar_gt_gral_rarity = en_ar_GOOGLE_general_rarity) %>%
  rename(enar_opus_gral_rarity = en_ar_TRANSFORMERS_general_rarity) %>%
  rename(en_native_dom_rar_nativeity = en_genre_rarity) %>%
  rename(es_native_dom_rar_nativeity = es_genre_rarity) %>%
  rename(ar_native_dom_rar_nativeity = ar_genre_rarity) %>%
  rename(esen_deep_dom_rarity = es_en_DEEP_genre_rarity) %>%
  rename(esen_deepl_dom_rarity = es_en_DEEPL_genre_rarity) %>%
  rename(esen_gt_dom_rarity = es_en_GOOGLE_genre_rarity) %>%
  rename(esen_opus_dom_rarity = es_en_TRANSFORMERS_genre_rarity) %>%
  rename(aren_deep_dom_rarity = ar_en_DEEP_genre_rarity) %>%
  rename(aren_deepl_dom_rarity = ar_en_DEEPL_genre_rarity) %>%
  rename(aren_gt_dom_rarity = ar_en_GOOGLE_genre_rarity) %>%
  rename(aren_opus_dom_rarity = ar_en_TRANSFORMERS_genre_rarity) %>%
  rename(enes_deep_dom_rarity = en_es_DEEP_genre_rarity) %>%
  rename(enes_deepl_dom_rarity = en_es_DEEPL_genre_rarity) %>%
  rename(enes_gt_dom_rarity = en_es_GOOGLE_genre_rarity) %>%
  rename(enes_opus_dom_rarity = en_es_TRANSFORMERS_genre_rarity) %>%
  rename(enar_deep_dom_rarity = en_ar_DEEP_genre_rarity) %>%
  rename(enar_deepl_dom_rarity = en_ar_DEEPL_genre_rarity) %>%
  rename(enar_gt_dom_rarity = en_ar_GOOGLE_genre_rarity) %>%
  rename(enar_opus_dom_rarity = en_ar_TRANSFORMERS_genre_rarity) %>%
  rename(en_native_verbs = en_verb_counts) %>%
  rename(es_native_verbs = es_verb_counts) %>%
  rename(ar_native_verbs = ar_verb_counts) %>%
  rename(esen_deep_verbs = es_en_DEEP_verb_counts) %>%
  rename(esen_deepl_verbs = es_en_DEEPL_verb_counts) %>%
  rename(esen_gt_verbs = es_en_GOOGLE_verb_counts) %>%
  rename(esen_opus_verbs = es_en_TRANSFORMERS_verb_counts) %>%
  rename(aren_deep_verbs = ar_en_DEEP_verb_counts) %>%
  rename(aren_deepl_verbs = ar_en_DEEPL_verb_counts) %>%
  rename(aren_gt_verbs = ar_en_GOOGLE_verb_counts) %>%
  rename(aren_opus_verbs = ar_en_TRANSFORMERS_verb_counts) %>%
  rename(enes_deep_verbs = en_es_DEEP_verb_counts) %>%
  rename(enes_deepl_verbs = en_es_DEEPL_verb_counts) %>%
  rename(enes_gt_verbs = en_es_GOOGLE_verb_counts) %>%
  rename(enes_opus_verbs = en_es_TRANSFORMERS_verb_counts) %>%
  rename(enar_deep_verbs = en_ar_DEEP_verb_counts) %>%
  rename(enar_deepl_verbs = en_ar_DEEPL_verb_counts) %>%
  rename(enar_gt_verbs = en_ar_GOOGLE_verb_counts) %>%
  rename(enar_opus_verbs = en_ar_TRANSFORMERS_verb_counts) %>%
  rename(en_native_nouns = en_noun_counts) %>%
  rename(es_native_nouns = es_noun_counts) %>%
  rename(ar_native_nouns = ar_noun_counts) %>%
  rename(esen_deep_nouns = es_en_DEEP_noun_counts) %>%
  rename(esen_deepl_nouns = es_en_DEEPL_noun_counts) %>%
  rename(esen_gt_nouns = es_en_GOOGLE_noun_counts) %>%
  rename(esen_opus_nouns = es_en_TRANSFORMERS_noun_counts) %>%
  rename(aren_deep_nouns = ar_en_DEEP_noun_counts) %>%
  rename(aren_deepl_nouns = ar_en_DEEPL_noun_counts) %>%
  rename(aren_gt_nouns = ar_en_GOOGLE_noun_counts) %>%
  rename(aren_opus_nouns = ar_en_TRANSFORMERS_noun_counts) %>%
  rename(enes_deep_nouns = en_es_DEEP_noun_counts) %>%
  rename(enes_deepl_nouns = en_es_DEEPL_noun_counts) %>%
  rename(enes_gt_nouns = en_es_GOOGLE_noun_counts) %>%
  rename(enes_opus_nouns = en_es_TRANSFORMERS_noun_counts) %>%
  rename(enar_deep_nouns = en_ar_DEEP_noun_counts) %>%
  rename(enar_deepl_nouns = en_ar_DEEPL_noun_counts) %>%
  rename(enar_gt_nouns = en_ar_GOOGLE_noun_counts) %>%
  rename(enar_opus_nouns = en_ar_TRANSFORMERS_noun_counts) %>%
  rename(en_native_lemmas = en_lemma_counts) %>%
  rename(es_native_lemmas = es_lemma_counts) %>%
  rename(ar_native_lemmas = ar_lemma_counts) %>%
  rename(esen_deep_lemmas = es_en_DEEP_lemma_counts) %>%
  rename(esen_deepl_lemmas = es_en_DEEPL_lemma_counts) %>%
  rename(esen_gt_lemmas = es_en_GOOGLE_lemma_counts) %>%
  rename(esen_opus_lemmas = es_en_TRANSFORMERS_lemma_counts) %>%
  rename(aren_deep_lemmas = ar_en_DEEP_lemma_counts) %>%
  rename(aren_deepl_lemmas = ar_en_DEEPL_lemma_counts) %>%
  rename(aren_gt_lemmas = ar_en_GOOGLE_lemma_counts) %>%
  rename(aren_opus_lemmas = ar_en_TRANSFORMERS_lemma_counts) %>%
  rename(enes_deep_lemmas = en_es_DEEP_lemma_counts) %>%
  rename(enes_deepl_lemmas = en_es_DEEPL_lemma_counts) %>%
  rename(enes_gt_lemmas = en_es_GOOGLE_lemma_counts) %>%
  rename(enes_opus_lemmas = en_es_TRANSFORMERS_lemma_counts) %>%
  rename(enar_deep_lemmas = en_ar_DEEP_lemma_counts) %>%
  rename(enar_deepl_lemmas = en_ar_DEEPL_lemma_counts) %>%
  rename(enar_gt_lemmas = en_ar_GOOGLE_lemma_counts) %>%
  rename(enar_opus_lemmas = en_ar_TRANSFORMERS_lemma_counts)



names(data)
  



# RELOCATE VARS --------------------------------------------------

data <- data %>% 
  relocate(c(id, en_native, es_native, ar_native, esen_deep, esen_deepl, esen_gt, esen_opus, aren_deep, aren_deepl, aren_gt, 
             aren_opus, enes_deep, enes_deepl, enes_gt, enes_opus, enar_deep, enar_deepl, enar_gt, enar_opus, en_native_bin_class, 
             es_native_bin_class, ar_native_bin_class, esen_deep_bin_class, esen_deepl_bin_class, esen_gt_bin_class, esen_opus_bin_class, 
             aren_deep_bin_class, aren_deepl_bin_class, aren_gt_bin_class, aren_opus_bin_class, enes_deep_bin_class, enes_deepl_bin_class, 
             enes_gt_bin_class, enes_opus_bin_class, enar_deep_bin_class, enar_deepl_bin_class, enar_gt_bin_class, enar_opus_bin_class, 
             en_native_bin_pred, es_native_bin_pred, ar_native_bin_pred, esen_deep_bin_pred, esen_deepl_bin_pred, esen_gt_bin_pred, 
             esen_opus_bin_pred, aren_deep_bin_pred, aren_deepl_bin_pred, aren_gt_bin_pred, aren_opus_bin_pred, enes_deep_bin_pred,
             enes_deepl_bin_pred, enes_gt_bin_pred, enes_opus_bin_pred, enar_deep_bin_pred, enar_deepl_bin_pred, enar_gt_bin_pred, 
             enar_opus_bin_pred, en_native_gral_rar_nativeity, es_native_gral_rar_nativeity, ar_native_gral_rar_nativeity, esen_deep_gral_rarity, 
             esen_deepl_gral_rarity, esen_gt_gral_rarity, esen_opus_gral_rarity, aren_deep_gral_rarity, aren_deepl_gral_rarity, aren_gt_gral_rarity, 
             aren_opus_gral_rarity, enes_deep_gral_rarity, enes_deepl_gral_rarity, enes_gt_gral_rarity, enes_opus_gral_rarity, enar_deep_gral_rarity, 
             enar_deepl_gral_rarity, enar_gt_gral_rarity, enar_opus_gral_rarity, en_native_dom_rar_nativeity, es_native_dom_rar_nativeity, 
             ar_native_dom_rar_nativeity, esen_deep_dom_rarity, esen_deepl_dom_rarity, esen_gt_dom_rarity, esen_opus_dom_rarity, aren_deep_dom_rarity,
             aren_deepl_dom_rarity, aren_gt_dom_rarity, aren_opus_dom_rarity, enes_deep_dom_rarity, enes_deepl_dom_rarity, enes_gt_dom_rarity, 
             enes_opus_dom_rarity, enar_deep_dom_rarity, enar_deepl_dom_rarity, enar_gt_dom_rarity, enar_opus_dom_rarity, en_native_verbs, 
             es_native_verbs, ar_native_verbs, esen_deep_verbs, esen_deepl_verbs, esen_gt_verbs, esen_opus_verbs, aren_deep_verbs, aren_deepl_verbs, 
             aren_gt_verbs, aren_opus_verbs, enes_deep_verbs, enes_deepl_verbs, enes_gt_verbs, enes_opus_verbs, enar_deep_verbs, enar_deepl_verbs, 
             enar_gt_verbs, enar_opus_verbs, en_native_nouns, es_native_nouns, ar_native_nouns, esen_deep_nouns, esen_deepl_nouns, esen_gt_nouns, 
             esen_opus_nouns, aren_deep_nouns, aren_deepl_nouns, aren_gt_nouns, aren_opus_nouns, enes_deep_nouns, enes_deepl_nouns, enes_gt_nouns, 
             enes_opus_nouns, enar_deep_nouns, enar_deepl_nouns, enar_gt_nouns, enar_opus_nouns, en_native_lemmas, es_native_lemmas, ar_native_lemmas, 
             esen_deep_lemmas, esen_deepl_lemmas, esen_gt_lemmas, esen_opus_lemmas, aren_deep_lemmas, aren_deepl_lemmas, aren_gt_lemmas, 
             aren_opus_lemmas, enes_deep_lemmas, enes_deepl_lemmas, enes_gt_lemmas, enes_opus_lemmas, enar_deep_lemmas, enar_deepl_lemmas, 
             enar_gt_lemmas, enar_opus_lemmas))

names(data)



# EXPORT DATA --------------------------------------------------

# Export to CSV
write.csv(data, "1_data/data_master_250813.csv")





# End of script