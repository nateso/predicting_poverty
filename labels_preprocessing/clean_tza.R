library(haven)  
library(gtools)
library(reshape2)
library(ggplot2)
library(dplyr)
library(tidyr)
library(forcats)
library(magrittr)
library(geosphere)
library(labelled)

rm(list = ls())

#*******************************************************************************
#### Tanzania ####
#*******************************************************************************
tz1_a <- read_dta("../../Data/lsms/Tanzania/TZA_2008/SEC_A_T.dta")
tz1_cons <- read_dta("../../Data/lsms/Tanzania/TZA_2008/TZY1.HH.Consumption.dta")
tz1_geos <- read_dta("../../Data/lsms/Tanzania/TZA_2008/HH.Geovariables_Y1.dta")
tz1_house <- read_dta("../../Data/lsms/Tanzania/TZA_2008/SEC_H1_J_K2_O2_P1_Q1_S1.dta")
tz1_ass <- read_dta("../../Data/lsms/Tanzania/TZA_2008/SEC_N.dta")

tz2_a <- read_dta("../../Data/lsms/Tanzania/TZA_2010/HH_SEC_A.dta")
tz2_cons <- read_dta("../../Data/lsms/Tanzania/TZA_2010/TZY2.HH.Consumption.dta")
tz2_geos <- read_dta("../../Data/lsms/Tanzania/TZA_2010/HH.Geovariables_Y2.dta")
tz2_house <- read_dta("../../Data/lsms/Tanzania/TZA_2010/HH_SEC_J1.dta")
tz2_ass <- read_dta("../../Data/lsms/Tanzania/TZA_2010/HH_SEC_N.dta")

tz3_a <- read_dta("../../Data/lsms/Tanzania/TZA_2012/HH_SEC_A.dta")
tz3_cons <- read_dta("../../Data/lsms/Tanzania/TZA_2012/ConsumptionNPS3.dta")
tz3_geos <- read_dta("../../Data/lsms/Tanzania/TZA_2012/HouseholdGeovars_Y3.dta")
tz3_house <- read_dta("../../Data/lsms/Tanzania/TZA_2012/HH_SEC_I.dta")
tz3_ass <- read_dta("../../Data/lsms/Tanzania/TZA_2012/HH_SEC_M.dta")

tz4_a <- read_dta("../../Data/lsms/Tanzania/TZA_2014_ext_panel/hh_sec_a.dta")
tz4_house <- read_dta("../../Data/lsms/Tanzania/TZA_2014_ext_panel/hh_sec_i.DTA")
tz4_ass <- read_dta("../../Data/lsms/Tanzania/TZA_2014_ext_panel/hh_sec_m.DTA")
tz4_cons <- read_dta("../../Data/lsms/Tanzania/TZA_2014_ext_panel/consumptionnps4_extended_panel.dta")

tz5_a <- read_dta("../../Data/lsms/Tanzania/TZA_2019_ext_panel/HH_SEC_A.dta")
tz5_house <- read_dta("../../Data/lsms/Tanzania/TZA_2019_ext_panel/HH_SEC_I.dta")
tz5_ass <- read_dta("../../Data/lsms/Tanzania/TZA_2019_ext_panel/HH_SEC_M.dta")
tz5_cons <- read_dta("../../Data/lsms/Tanzania/TZA_2019_ext_panel/consumptionsdd.dta")

panel_key <- read_dta("../../Data/lsms/Tanzania/TZA_2012/NPSY3.PANEL.KEY.dta")
#*******************************************************************************
#### id variables ####
#*******************************************************************************
# get the panel key for the first three rounds
panel_key %<>% select(y1_hhid, y2_hhid, y3_hhid) %>% 
  distinct()

y2_y1_ids <- panel_key %>% select(y1_hhid,y2_hhid) %>% 
  distinct() %>% filter(y2_hhid != "") %>% filter(y1_hhid !="")

w1_ids <- tz1_a %>% select(hhid)

w2_ids <- tz2_a %>% filter(hh_a11 != 3) %>% # 1 same hh same location, 2 same hh, different location, 3 split-off household
  left_join(y2_y1_ids, by = 'y2_hhid') %>% 
  filter(y1_hhid %in% w1_ids$hhid) %>% select(y1_hhid, y2_hhid) %>% 
  filter(y2_hhid %in% tz2_cons$y2_hhid[is.na(tz2_cons$expmR) == F]) %>% 
  left_join(tz2_geos %>% select(y2_hhid,dist_Y1Y2), by = 'y2_hhid') %>% 
  mutate(dist_Y1Y2 = ifelse(is.na(dist_Y1Y2),0,dist_Y1Y2)) %>% 
  filter(dist_Y1Y2 < 10) %>% # remove those households that moved outside of ea
  distinct(y1_hhid, .keep_all = T) %>%  # some split-offs are still in the data (like 10 just take the first hhs)
  select(-dist_Y1Y2)

w3_ids <- tz3_a %>% filter(hh_a10 == 1) %>% filter(hh_a11 != 3) %>%  # remove split-offs and households that moved outside of ea
  rename(y2_hhid = hh_a09) %>% filter(y2_hhid %in% w2_ids$y2_hhid) %>% 
  left_join(y2_y1_ids, by = 'y2_hhid') %>% select(y1_hhid,y2_hhid,y3_hhid) %>% 
  filter(y3_hhid %in% tz3_cons$y3_hhid) %>% 
  left_join(tz3_geos %>% select(y3_hhid,dist_Y1Y3), by = 'y3_hhid') %>% 
  mutate(dist_Y1Y3 = ifelse(is.na(dist_Y1Y3),0,dist_Y1Y3)) %>% 
  filter(dist_Y1Y3 < 10) %>% 
  distinct(y1_hhid, .keep_all = T) %>% select(-dist_Y1Y3)

#### the extended panel...  
w4_ids <- tz4_a %>% filter(hh_a10 == 1) %>% filter(hh_a11 %in% c(1,2)) %>% # no geovars at household level... thus restrict to households in same location as before.
  rename(y3_hhid = hh_a09) %>% 
  filter(y3_hhid %in% w3_ids$y3_hhid) %>% 
  filter(y4_hhid %in% tz4_cons$y4_hhid) %>%
  left_join(w3_ids, by = 'y3_hhid') %>% 
  distinct(y1_hhid, .keep_all = T) %>% 
  select(y1_hhid,y2_hhid,y3_hhid,y4_hhid)

w5_ids <- tz5_a %>% filter(hh_a10 == 1) %>% filter(tracking_class %in% c(1,2)) %>% 
  filter(y4_hhid %in% w4_ids$y4_hhid) %>%
  left_join(w4_ids, by = 'y4_hhid') %>% 
  select(y1_hhid, y2_hhid, y3_hhid, y4_hhid, sdd_hhid)

### geo variables
tz_geos <- tz1_geos %>% select(hhid, lat_modified, lon_modified) %>% 
  rename(lat = lat_modified, lon = lon_modified) %>% 
  left_join(tz1_a %>% select(hhid, clusterid), by = 'hhid') %>% 
  select(-hhid) %>% distinct()

ea_rur_urb <- tz1_a %>% 
  select(clusterid, rural) %>% distinct() %>% 
  mutate(rural = ifelse(rural == 'Rural',1,0))

tz_geos %<>% left_join(ea_rur_urb, by = 'clusterid') 

### long panel ids
long_panel_ids <- w5_ids %>% 
  left_join(tz1_a %>% select(hhid,clusterid), by = c('y1_hhid'='hhid')) %>% 
  left_join(tz_geos, by = 'clusterid')
  
#### the short panel ids
full_eas <- unique(tz1_a$clusterid)
long_panel_eas <- unique(long_panel_ids$clusterid)
short_panel_eas <- full_eas[(full_eas %in% long_panel_eas) == F]

short_panel_ids <- w3_ids %>% 
  left_join(tz1_a %>% select(hhid,clusterid), by = c('y1_hhid'='hhid')) %>% 
  filter(clusterid %in% short_panel_eas) %>% 
  left_join(tz_geos, by = 'clusterid')


#*******************************************************************************
#### Housing characteristics ####
#*******************************************************************************
extract_housing <- function(dat,id_var,rm,wl,rf,fl,toi,wat,fuel, electricity){
  vars <- c(id_var, rm,wl,rf,fl,toi,wat,fuel,electricity)
  aux <- dat %>% select(all_of(vars)) 
  names(aux) <- c(id_var,'rooms','wall','roof','floor','toilet','watsup','cooking_fuel','electric')
  aux <- aux %>% mutate(electric = ifelse(electric == 1,1,0))# %>% mutate_all(as.vector)
  return(aux)
}

tz1_house %<>% mutate(sjq19 = ifelse(is.na(sjq19),0,sjq19)) 
tz1_house <- extract_housing(tz1_house,'hhid','sjq3_1','sjq4','sjq5','sjq6','sjq16','sjq8','sjq17_1','sjq19')

tz2_house %<>% mutate(hh_j18 = ifelse(is.na(hh_j18),0,hh_j18)) 
tz2_house <- extract_housing(tz2_house,'y2_hhid','hh_j04_1','hh_j05','hh_j06','hh_j07','hh_j10','hh_j19','hh_j16','hh_j18')

tz3_house %<>% mutate(hh_i18 = ifelse(is.na(hh_i18),0,hh_i18)) 
tz3_house <- extract_housing(tz3_house,'y3_hhid','hh_i07_1','hh_i08','hh_i09','hh_i10','hh_i12','hh_i19','hh_i16','hh_i18')

tz4_house %<>% mutate(hh_i18 = ifelse(is.na(hh_i18),0,hh_i18)) 
tz4_house <- extract_housing(tz4_house,'y4_hhid','hh_i07_1','hh_i08','hh_i09','hh_i10','hh_i12','hh_i19','hh_i16','hh_i18')

tz5_house %<>% mutate(hh_i18 = ifelse(is.na(hh_i18),0,hh_i18)) 
tz5_house <- extract_housing(tz5_house,'sdd_hhid','hh_i07_1','hh_i08','hh_i09','hh_i10','hh_i12','hh_i19','hh_i16','hh_i18')

#*******************************************************************************
#### Recode housing ####
#*******************************************************************************
substrRight <- function(x, n){
  substr(x, nchar(x)-n+1, nchar(x))
}

recode_housing <- function(dat,file_path){
  file_name <- tail(strsplit(file_path,"/")[[1]],1)
  file_name <- sub(".csv","",file_name)
  var_name <- unlist(strsplit(file_name,"_recode"))[1]
  recode_df <- read.csv(file_path)
  aux <- dat %>% left_join(recode_df, by = var_name, all.x = T) %>% 
    select(-all_of(var_name))
  return(aux)
}

tz1_house <- recode_housing(tz1_house, "../../Data/lsms/Tanzania/recode//watsup_recode_w1.csv")
tz1_house <- recode_housing(tz1_house, "../../Data/lsms/Tanzania/recode//toilet_recode_w1.csv")

tz2_house <- recode_housing(tz2_house, "../../Data/lsms/Tanzania/recode//watsup_recode_w2_w3.csv")
tz2_house <- recode_housing(tz2_house, "../../Data/lsms/Tanzania/recode//toilet_recode_w2.csv")

tz3_house <- recode_housing(tz3_house, "../../Data/lsms/Tanzania/recode//watsup_recode_w2_w3.csv")
tz3_house <- recode_housing(tz3_house, "../../Data/lsms/Tanzania/recode//toilet_recode_w3_w4_w5.csv")

tz4_house <- recode_housing(tz4_house, "../../Data/lsms/Tanzania/recode//watsup_recode_w4_w5.csv")
tz4_house <- recode_housing(tz4_house, "../../Data/lsms/Tanzania/recode//toilet_recode_w3_w4_w5.csv")

tz5_house <- recode_housing(tz5_house, "../../Data/lsms/Tanzania/recode//watsup_recode_w4_w5.csv")
tz5_house <- recode_housing(tz5_house, "../../Data/lsms/Tanzania/recode//toilet_recode_w3_w4_w5.csv")

recode_files <- list.files("../../Data/lsms/Tanzania/recode/",full.names = T)
house_list <- list(tz1_house, tz2_house, tz3_house, tz4_house,tz5_house)
names(house_list) <- c('tz1_house', 'tz2_house', 'tz3_house', 'tz4_house','tz5_house')
for(file in recode_files){
  file_name <- tail(strsplit(file,"/")[[1]],1)
  file_name <- sub(".csv","",file_name)
  if(grepl('w1|w2|w3|w4|w5',file_name)){next}
  for(i in 1:length(house_list)){
    var_name <- unlist(strsplit(file_name,"_recode"))[1]
    recode_df <- read.csv(file)
    house_list[[i]] <- house_list[[i]] %>% left_join(recode_df, by = var_name, all.x =T) %>% 
      select(-all_of(var_name))
  }
}

list2env(house_list,globalenv())
rm(house_list,file, file_name,i,var_name,recode_files)

#*******************************************************************************
#### Assets ####
#*******************************************************************************
assets = cbind.data.frame(code = c(401, 406, 404, 427, 426, 425, 402, 403),
                          label = c("radio",'tv','fridge','bike','motorcycle','car', 'landline_phone', 'mob_phone'))


tz1_ass %<>% filter(sncode %in% assets$code) %>%
  left_join(assets, by = c('sncode' = 'code')) %>% 
  mutate(snq1 = ifelse(is.na(snq1),0,snq1)) %>% 
  mutate(snq1 = ifelse(snq1 > 0,1,0)) %>% 
  select(-sncode) %>% 
  pivot_wider(names_from = label, values_from = snq1) %>% 
  replace(is.na(.), 0) %>% 
  mutate(phone = ifelse(landline_phone + mob_phone > 0,1,0)) %>% 
  select(-landline_phone, -mob_phone)
  
tz1_ass_missing_ids <- tz1_a$hhid[(tz1_a$hhid %in% tz1_ass$hhid) == F]
tz1_ass_missing <- data.frame(matrix(0,nrow = length(tz1_ass_missing_ids),ncol =7))
names(tz1_ass_missing) <- c("radio",'tv','fridge','bike','motorcycle','car', 'phone')
tz1_ass_missing$hhid <- tz1_ass_missing_ids
tz1_ass_missing %<>% relocate(hhid) 

tz1_ass <- rbind(tz1_ass,tz1_ass_missing)

asset_list <- list(tz2_ass, tz3_ass, tz4_ass, tz5_ass)
names(asset_list) <- c('tz2_ass', 'tz3_ass', 'tz4_ass', 'tz5_ass')
for(i in 1:length(asset_list)){
  if(i == 1){
    asset_list[[i]]$hh_m01 <- asset_list[[i]]$hh_n01_2
  }
  asset_list[[i]] %<>% filter(itemcode %in% assets$code) %>% 
    left_join(assets, by = c('itemcode' = 'code')) %>% 
    mutate(own = ifelse(hh_m01 > 0,1,0)) %>% 
    select(matches('hhid'),label,own) %>% 
    pivot_wider(names_from = label, values_from = own) %>% 
    mutate(phone = ifelse(landline_phone + mob_phone > 0,1,0)) %>% 
    select(-landline_phone, -mob_phone)
}
list2env(asset_list,globalenv())
rm(asset_list,tz1_ass_missing,tz1_ass_missing_ids,assets)

#*******************************************************************************
#### consumption ####
#*******************************************************************************
# consumption data is usually reported in real currecny for the year of the survey
# e.g. Oct 2008 - Sep 2009 (spatial and temporal adjustment)
tza_cpi <- read.csv("../../Data/lsms/Tanzania/tza_cpi.csv") %>% 
  filter(Country.Code == "TZA") %>% 
  select(-Series.Name, -Series.Code, -Country.Name)
names(tza_cpi) <- c("country",paste0("y_",2008:2021))
tza_cpi %<>% 
  pivot_longer(cols = starts_with("y_"), names_to = "year", names_prefix = "y_", values_to = "yearly_cpi") %>% 
  mutate(deflator_2017 = 175.0378/yearly_cpi, 
         year = as.numeric(year)) %>% 
  select(year, deflator_2017)


# add year to all cons datasets
tz1_cons$year <- 2009
tz2_cons$year <- 2011
tz3_cons$year <- 2013
tz4_cons$year <- 2015
tz5_cons$year <- 2019

cons_list <- list(tz1_cons, tz2_cons, tz3_cons, tz4_cons, tz5_cons)
names(cons_list) <- c('tz1_cons', 'tz2_cons', 'tz3_cons', 'tz4_cons', 'tz5_cons')

for(i in 1:length(cons_list)){
  cons_list[[i]] <- cons_list[[i]] %>%  select(matches('hhid'),
                                               matches('adulteq'), 
                                               matches('hhsize'), expmR, year)
  if(i == 2){cons_list[[i]] %<>% select(-hhid_2008)}
  names(cons_list[[i]])[-1] <- c('adulteq', 'hh_size', 'expmR', 'year')
  cons_list[[i]] <- cons_list[[i]] %>% left_join(tza_cpi, by = 'year')
  cons_list[[i]]$cons_lcu_2017 <- (cons_list[[i]]$expmR * cons_list[[i]]$deflator_2017)/365
  cons_list[[i]]$cons_pc_lcu_2017 = cons_list[[i]]$cons_lcu_2017 / cons_list[[i]]$hh_size
  cons_list[[i]]$cons_pc_usd_2017 = cons_list[[i]]$cons_pc_lcu_2017 / 754.621459960938
  cons_list[[i]] %<>% select(-expmR,-cons_lcu_2017,-deflator_2017) %>% relocate(year) %>% 
    mutate_all(as.vector)
}
list2env(cons_list,globalenv())
rm(cons_list)

#*******************************************************************************
#### merge data ####
#*******************************************************************************
tz1 <- tz1_house %>% 
  left_join(tz1_ass, by = 'hhid') %>% 
  left_join(tz1_cons, by = 'hhid') %>% 
  mutate(wave = 1) %>% 
  mutate(start_year = 2008, start_month = 10, end_year = 2009, end_month = 09)

tz2 <- tz2_house %>% 
  left_join(tz2_ass, by = 'y2_hhid') %>% 
  left_join(tz2_cons, by = 'y2_hhid') %>% 
  mutate(wave = 2) %>% 
  mutate(start_year = 2010, start_month = 10, end_year = 2011, end_month = 09)

tz3 <- tz3_house %>% 
  left_join(tz3_ass, by = 'y3_hhid') %>% 
  left_join(tz3_cons, by = 'y3_hhid') %>% 
  mutate(wave = 3) %>% 
  mutate(start_year = 2012, start_month = 10, end_year = 2013, end_month = 11)

tz4 <- tz4_house %>% 
  left_join(tz4_ass, by = 'y4_hhid') %>% 
  left_join(tz4_cons, by = 'y4_hhid') %>% 
  mutate(wave = 4) %>% 
  mutate(start_year = 2014, start_month = 10, end_year = 2015, end_month = 10)

tz5 <- tz5_house %>% 
  left_join(tz5_ass, by = 'sdd_hhid') %>% 
  left_join(tz5_cons, by = 'sdd_hhid') %>% 
  mutate(wave = 5) %>% 
  mutate(start_year = 2019, start_month = 01, end_year = 2019, end_month = 12)

#*******************************************************************************
#### split data ####
#*******************************************************************************
short_panel <- rbind.data.frame(
  tz1 %>% filter(tz1$hhid %in% short_panel_ids$y1_hhid) %>% rename(y1_hhid = hhid),
  tz2 %>% filter(tz2$y2_hhid %in% short_panel_ids$y2_hhid) %>% 
    left_join(short_panel_ids %>% select(y1_hhid, y2_hhid), by = 'y2_hhid') %>% 
    select(-y2_hhid),
  tz3 %>% filter(tz3$y3_hhid %in% short_panel_ids$y3_hhid) %>% 
    left_join(short_panel_ids %>% select(y1_hhid, y3_hhid), by = 'y3_hhid') %>% 
    select(-y3_hhid)
)

long_panel <- rbind.data.frame(
  tz1 %>% filter(tz1$hhid %in% long_panel_ids$y1_hhid) %>% rename(y1_hhid = hhid),
  tz2 %>% filter(tz2$y2_hhid %in% long_panel_ids$y2_hhid) %>% 
    left_join(long_panel_ids %>% select(y1_hhid, y2_hhid), by = 'y2_hhid') %>% 
    select(-y2_hhid),
  tz3 %>% filter(tz3$y3_hhid %in% long_panel_ids$y3_hhid) %>% 
    left_join(long_panel_ids %>% select(y1_hhid, y3_hhid), by = 'y3_hhid') %>% 
    select(-y3_hhid),
  tz4 %>% filter(tz4$y4_hhid %in% long_panel_ids$y4_hhid) %>% 
    left_join(long_panel_ids %>% select(y1_hhid, y4_hhid), by = 'y4_hhid') %>% 
    select(-y4_hhid),
  tz5 %>% filter(tz5$sdd_hhid %in% long_panel_ids$sdd_hhid) %>% 
    left_join(long_panel_ids %>% select(y1_hhid, sdd_hhid), by = 'sdd_hhid') %>% 
    select(-sdd_hhid)
)

short_panel %<>% 
  left_join(short_panel_ids %>% select(y1_hhid,clusterid), by = 'y1_hhid') %>% 
  left_join(tz_geos, by = 'clusterid') %>% 
  rename(case_id = y1_hhid, cluster_id = clusterid) %>% 
  mutate(country = 'tza') %>% 
  mutate(case_id = paste(country, case_id,sep = "_"),
         cluster_id = paste(country, cluster_id, sep = "_")) %>% 
  select(country,start_year, start_month, end_year, end_month, wave,
         cluster_id,rural,lat,lon,case_id,rooms,electric,floor_qual,wall_qual,
         roof_qual,cooking_fuel_qual,
         toilet_qual,watsup_qual,radio,tv,bike,motorcycle,fridge,car,phone,
         hh_size,adulteq,cons_pc_lcu_2017,cons_pc_usd_2017) %>% 
  mutate_all(as.vector)

short_panel_attr <- tz1 %>% 
  left_join(tz1_a %>% select(hhid, clusterid), by = 'hhid') %>% 
  filter(clusterid %in% short_panel_eas) %>% 
  filter((hhid %in% short_panel$case_id)==F) %>% 
  left_join(tz_geos, by = 'clusterid') %>% 
  rename(case_id = hhid, cluster_id = clusterid) %>% 
  mutate(country = 'tza') %>% 
  mutate(case_id = paste(country, case_id,sep = "_"),
         cluster_id = paste(country, cluster_id, sep = "_")) %>% 
  select(country,start_year, start_month, end_year, end_month, wave,
         cluster_id,rural,lat,lon,case_id,rooms,electric,floor_qual,wall_qual,
         roof_qual,cooking_fuel_qual,
         toilet_qual,watsup_qual,radio,tv,bike,motorcycle,fridge,car,phone,
         hh_size,adulteq,cons_pc_lcu_2017,cons_pc_usd_2017) %>% 
  mutate_all(as.vector)

long_panel %<>% 
  left_join(long_panel_ids %>% select(y1_hhid,clusterid), by = 'y1_hhid') %>% 
  left_join(tz_geos, by = 'clusterid') %>% 
  rename(case_id = y1_hhid, cluster_id = clusterid) %>% 
  mutate(country = 'tza') %>% 
  mutate(case_id = paste(country, case_id,sep = "_"),
         cluster_id = paste(country, cluster_id, sep = "_")) %>% 
  select(country,start_year, start_month, end_year, end_month, wave,
         cluster_id,rural,lat,lon,case_id,rooms,electric,floor_qual,wall_qual,
         roof_qual,cooking_fuel_qual,
         toilet_qual,watsup_qual,radio,tv,bike,motorcycle,fridge,car,phone,
         hh_size,adulteq,cons_pc_lcu_2017,cons_pc_usd_2017) %>% 
  mutate_all(as.vector)

long_panel_attr <- tz1 %>% 
  left_join(tz1_a %>% select(hhid, clusterid), by = 'hhid') %>% 
  filter(clusterid %in% long_panel_eas) %>% 
  filter((hhid %in% long_panel$case_id)==F) %>% 
  left_join(tz_geos, by = 'clusterid') %>% 
  rename(case_id = hhid, cluster_id = clusterid) %>% 
  mutate(country = 'tza') %>% 
  mutate(case_id = paste(country, case_id,sep = "_"),
         cluster_id = paste(country, cluster_id, sep = "_")) %>% 
  select(country,start_year, start_month, end_year, end_month, wave,
         cluster_id,rural,lat,lon,case_id,rooms,electric,floor_qual,wall_qual,
         roof_qual,cooking_fuel_qual,
         toilet_qual,watsup_qual,radio,tv,bike,motorcycle,fridge,car,phone,
         hh_size,adulteq,cons_pc_lcu_2017,cons_pc_usd_2017) %>% 
  mutate_all(as.vector)

#*******************************************************************************
#### save data ####
#*******************************************************************************
write.csv(short_panel,"../../Data/processed/tza_labels_short.csv", row.names = F)
write.csv(short_panel_attr,"../../Data/processed/tza_labels_short_attr.csv", row.names = F)
write.csv(long_panel, '../../Data/processed/tza_labels_long.csv', row.names = F)
write.csv(long_panel_attr, "../../Data/processed/tza_labels_long_attr.csv", row.names = F)





















