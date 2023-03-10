library(dplyr)
library(tidyr)

#*******************************************************************************
#### impute missing value function ####
#*******************************************************************************

get_mode <- function(x, na.rm = FALSE) {
  if(na.rm){
    x = x[!is.na(x)]
  }
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

get_cluster_mode <- function(asset_dat){
  vnames <- names(asset_dat)[-1]
  mode_asset_val <- asset_dat %>%
    select(-unique_id) %>% 
    group_by(cluster_id) %>% 
    summarise(across(everything(), list(get_mode)))
  names(mode_asset_val) <- vnames
  return(mode_asset_val)
}

impute_missing_vals <- function(asset_dat){
  vnames <- names(asset_dat)[c(-1,-2)]
  mode_asset_val <- get_cluster_mode(asset_dat)
  
  n_rows_NA <- nrow(asset_dat) - nrow(na.omit(asset_dat))
  print(paste('Number of rows with missing values:',n_rows_NA))
  out <- asset_dat %>% 
    left_join(mode_asset_val, by = 'cluster_id', suffix = c("","_NA_fill"))
  for(var in vnames){
    na_fill_var <- paste0(var,"_NA_fill")
    out[[var]] <- ifelse(is.na(out[[var]]),out[[na_fill_var]],out[[var]])
  }
  out %<>% select(-ends_with("_NA_fill"))
  return(out)
}

#*******************************************************************************
#### PCA wrapper function ####
#*******************************************************************************

create_wealth_index <- function(dat){
  if(names(dat)[1] != 'unique_id'){
    stop("The first column should be named 'unique_id'!")
  }
  n_NAs <- nrow(dat) - nrow(na.omit(dat))
  no_NA_dat <- na.omit(dat)
  # run the PCA and store in dataframe
  pca <- prcomp(no_NA_dat[,c(-1,-2)], scale = T, center = T)
  pca_result <- data.frame(
    unique_id = no_NA_dat$unique_id,
    asset_index = pca$x[,1]
  )
  # function print outs
  var_expl = (pca$sdev^2 / sum(pca$sdev^2))[1]
  print(paste('Number of observations with missing values:', n_NAs))
  print(paste('First PC explains', round(var_expl,4) * 100,"% of the variance"))
  
  # ensure that each id of dat is also returned.
  aux <- dat %>% select(unique_id) %>% 
    left_join(pca_result, by = 'unique_id')
  return(aux$asset_index)
}








