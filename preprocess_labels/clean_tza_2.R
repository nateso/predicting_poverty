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
#### Tanzania second survey wave ####
#*******************************************************************************
tz4_a <- read_dta("../../Data/lsms/Tanzania/TZA_2014/hh_sec_a.dta")
tz4_house <- read_dta("../../Data/lsms/Tanzania/TZA_2014/hh_sec_i.dta")
tz4_ass <- read_dta("../../Data/lsms/Tanzania/TZA_2014/hh_sec_m.dta")
tz4_cons <- read_dta("../../Data/lsms/Tanzania/TZA_2020/consumption_real_y4.dta")

tz_geos <- read_dta("../../Data/lsms/Tanzania/TZA_2014/npsy4.ea.offset.dta")

tz5_a <- read_dta("../../Data/lsms/Tanzania/TZA_2020/hh_sec_a.dta")
tz5_house <- read_dta("../../Data/lsms/Tanzania/TZA_2020/hh_sec_i.dta")
tz5_ass <- read_dta("../../Data/lsms/Tanzania/TZA_2020/hh_sec_m.dta")
tz5_cons <- read_dta("../../Data/lsms/Tanzania/TZA_2020/consumption_real_y5.dta")

panel_key <- read_dta("../../Data/lsms/Tanzania/TZA_2020/npsy5.panel.key.dta")
#*******************************************************************************
#### id variables ####
#*******************************************************************************
w4_ids <- tz4_a %>% select(y4_hhid)

w5_ids <- tz5_a %>% 
  filter(hh_a10 == 1) %>% filter(tracking_class %in% c(1,2)) %>% 
  filter(y4_hhid %in% tz4_a$y4_hhid) %>% # make sure households are also in wave 1.
  select(y5_hhid, y4_hhid, clusterid, y5_cluster)

tz_geos %<>% mutate(rural = ifelse(clustertype == 1, 1, 0)) %>% 
  rename(lat = lat_modified, lon = lon_modified) %>% 
  select(clusterid, rural, lat, lon)

panel_ids <- w5_ids %>% 
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

tz4_house %<>% mutate(hh_i18 = ifelse(is.na(hh_i18),0,hh_i18)) 
tz4_house <- extract_housing(tz4_house,'y4_hhid','hh_i07_1','hh_i08','hh_i09','hh_i10','hh_i12','hh_i19','hh_i16','hh_i18')

tz5_house %<>% mutate(hh_i18 = ifelse(is.na(hh_i18),0,hh_i18)) 
tz5_house <- extract_housing(tz5_house,'y5_hhid','hh_i07_1','hh_i08','hh_i09','hh_i10','hh_i12','hh_i19','hh_i16','hh_i18')

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

tz4_house <- recode_housing(tz4_house, "../../Data/lsms/Tanzania/recode//wall_recode.csv")
tz4_house <- recode_housing(tz4_house, '../../Data/lsms/Tanzania/recode/floor_recode.csv')
tz4_house <- recode_housing(tz4_house, '../../Data/lsms/Tanzania/recode/roof_recode.csv')
tz4_house <- recode_housing(tz4_house, "../../Data/lsms/Tanzania/recode//watsup_recode_w4_w5.csv")
tz4_house <- recode_housing(tz4_house, "../../Data/lsms/Tanzania/recode//toilet_recode_w3_w4_w5.csv")
tz4_house <- recode_housing(tz4_house, "../../Data/lsms/Tanzania/recode/cooking_fuel_recode.csv")

tz5_house <- recode_housing(tz5_house, "../../Data/lsms/Tanzania/recode//wall_recode.csv")
tz5_house <- recode_housing(tz5_house, '../../Data/lsms/Tanzania/recode/floor_recode_w5_2.csv')
tz5_house <- recode_housing(tz5_house, '../../Data/lsms/Tanzania/recode/roof_recode.csv')
tz5_house <- recode_housing(tz5_house, "../../Data/lsms/Tanzania/recode//watsup_recode_w4_w5.csv")
tz5_house <- recode_housing(tz5_house, "../../Data/lsms/Tanzania/recode//toilet_recode_w3_w4_w5.csv")
tz5_house <- recode_housing(tz5_house, "../../Data/lsms/Tanzania/recode/cooking_fuel_recode.csv")

#*******************************************************************************
#### Assets ####
#*******************************************************************************
assets = cbind.data.frame(code = c(401, 406, 404, 427, 426, 425, 402, 403),
                          label = c("radio",'tv','fridge','bike','motorcycle','car', 'landline_phone', 'mob_phone'))

asset_list <- list(tz4_ass, tz5_ass)
names(asset_list) <- c('tz4_ass', 'tz5_ass')
for(i in 1:length(asset_list)){
  asset_list[[i]] %<>% filter(itemcode %in% assets$code) %>% 
    left_join(assets, by = c('itemcode' = 'code')) %>% 
    mutate(own = ifelse(is.na(hh_m01),0,1)) %>% 
    mutate(own = ifelse(own > 0,1,0)) %>% 
    select(matches('hhid'),label,own) %>% 
    pivot_wider(names_from = label, values_from = own) %>% 
    mutate(phone = ifelse(landline_phone + mob_phone > 0,1,0)) %>% 
    select(-landline_phone, -mob_phone)
}
list2env(asset_list,globalenv())

#*******************************************************************************
#### consumption ####
#*******************************************************************************
tza_cpi <- read.csv("../../Data/lsms/Tanzania/tza_cpi.csv") %>% 
  filter(Country.Code == "TZA") %>% 
  select(-Series.Name, -Series.Code, -Country.Name)
names(tza_cpi) <- c("country",paste0("y_",2008:2021))
tza_cpi %<>% 
  pivot_longer(cols = starts_with("y_"), names_to = "year", names_prefix = "y_", values_to = "yearly_cpi") %>% 
  mutate(deflator_2017 = 175.0378/yearly_cpi, 
         year = as.numeric(year)) %>% 
  select(year, deflator_2017)

tz4_cons$year = 2015
tz5_cons$year = 2021

cons_list <- list(tz4_cons, tz5_cons)
names(cons_list) <- c('tz4_cons', 'tz5_cons')

for(i in 1:length(cons_list)){
  cons_list[[i]] <- cons_list[[i]] %>%  
    mutate(expmR = expmR_pae * adulteq) %>% 
    select(matches('hhid'), adulteq, hhsize, expmR, year)
  cons_list[[i]] <- cons_list[[i]] %>% left_join(tza_cpi, by = 'year')
  cons_list[[i]]$cons_lcu_2017 <- (cons_list[[i]]$expmR * cons_list[[i]]$deflator_2017)/28
  cons_list[[i]]$cons_pc_lcu_2017 = cons_list[[i]]$cons_lcu_2017 / cons_list[[i]]$hhsize
  cons_list[[i]]$cons_pc_usd_2017 = cons_list[[i]]$cons_pc_lcu_2017 / 754.621459960938
  cons_list[[i]] %<>% select(-expmR,-cons_lcu_2017,-deflator_2017) %>% relocate(year) %>% 
    mutate_all(as.vector) %>% rename(hh_size = hhsize)
}
list2env(cons_list,globalenv())
rm(cons_list)

#*******************************************************************************
#### merge data ####
#*******************************************************************************
tz4 <- tz4_house %>% 
  left_join(tz4_ass, by = 'y4_hhid') %>% 
  left_join(tz4_cons, by = 'y4_hhid') %>% 
  mutate(wave = 1) %>% 
  mutate(start_year = 2014, start_month = 10, end_year = 2015, end_month = 10)


tz5 <- tz5_house %>% 
  left_join(tz5_ass, by = 'y5_hhid') %>% 
  left_join(tz5_cons, by = 'y5_hhid') %>% 
  mutate(wave = 2) %>% 
  mutate(start_year = 2020, start_month = 12, end_year = 2022, end_month = 01)


panel <- rbind.data.frame(
  tz4 %>% filter(tz4$y4_hhid %in% panel_ids$y4_hhid) %>% 
    left_join(panel_ids %>% select(y4_hhid, y5_hhid), by = 'y4_hhid') %>% 
    select(-y4_hhid),
  tz5 %>% filter(tz5$y5_hhid %in% panel_ids$y5_hhid)
)

panel %<>% 
  left_join(panel_ids %>% select(y5_hhid, y5_cluster, lat, lon, rural), by = 'y5_hhid') %>% 
  rename(case_id = y5_hhid, cluster_id = y5_cluster) %>% 
  mutate(country = 'tza') %>% 
  mutate(case_id = paste(country, case_id,sep = "_"),
         cluster_id = paste(country, cluster_id, sep = "_")) %>% 
  select(country,start_year, start_month, end_year, end_month, wave,
         cluster_id,rural,lat,lon,case_id,rooms,electric,floor_qual,wall_qual,
         roof_qual,cooking_fuel_qual,
         toilet_qual,watsup_qual,radio,tv,bike,motorcycle,fridge,car,phone,
         hh_size,adulteq,cons_pc_lcu_2017,cons_pc_usd_2017) %>% 
  mutate_all(as.vector)

panel_attr <- tz4 %>% 
  filter((y4_hhid %in% panel_ids$y4_hhid)==F) %>% 
  left_join(panel_ids %>% select(y4_hhid, clusterid, lat, lon, rural), by = 'y4_hhid') %>% 
  rename(case_id = y4_hhid, cluster_id = clusterid) %>% 
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
write.csv(panel,"../../Data/lsms/processed/tza_labels_refresh.csv",row.names = FALSE)
write.csv(panel_attr, '../../Data/lsms/processed/tza_labels_refresh_attr.csv', row.names = FALSE)












