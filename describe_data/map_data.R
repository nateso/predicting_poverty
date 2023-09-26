library(dplyr)
library(magrittr)
library(sf)
library(ggplot2)
library(viridis)
library(afriadmin)
library(countrycode)
library(oceanis)
library(leaflet)

setwd("~/Documents/AngSt/master_thesis/data_analysis/predicting_poverty")
rm(list = ls())

#*******************************************************************************
#### Load Data ####
#*******************************************************************************

# load the lsms data including the recentered locations
lsms_df <- read.csv("../Data/lsms/processed/labels_cluster_v1.csv")
lsms_countries <- c("ETH",'MWI','NGA','TZA',"UGA")

# subset to distinct clusters
cl_df <- lsms_df %>%
  distinct(cluster_id, .keep_all = T)

# load the African shapefiles
africa_boundaries <- st_read("../Data/geoboundaries/ne_10m_admin_0_countries_deu/ne_10m_admin_0_countries_deu.shp") %>% 
  filter(REGION_UN == 'Africa') %>% 
  select(ISO_A3, geometry) %>%
  st_make_valid()

# load the world bank poverty survey data
wb_survey_df <- read.csv("../Data/wb_poverty_data/pip_dataset.csv")

#*******************************************************************************
#### Map years since last survey ####
#*******************************************************************************

# rearrange the poverty dataset and filter out the appropriate countries
wb_survey_df$iso_code <- countrycode(wb_survey_df$country, origin = 'country.name', destination = 'iso3c')
wb_survey_df %<>%
  select(country, iso_code, year) %>% 
  distinct() %>% 
  filter(iso_code %in% africa_boundaries$ISO) %>% 
  group_by(iso_code) %>%
  filter(year == max(year)) %>%
  ungroup() %>% 
  mutate(years_since_last_survey = as.integer(2023 - year))

breaks = seq(0,15,3)
wb_survey_df$bin_years <- cut(wb_survey_df$years_since_last_survey, breaks, right = T)
levels(wb_survey_df$bin_years) = paste(breaks, breaks + 3, sep = " to ")

# get basic descriptive statistics
summary(wb_survey_df$bin_years)
summary(wb_survey_df$years_since_last_survey)

# create a data file to map all African countries
survey_shp <- africa_boundaries %>%
  left_join(wb_survey_df, by = c('ISO_A3' = 'iso_code'))

# map the data
color_palette <- c('#C7DCF0','#1D92C1','#2372B6','#07306B', '#9FCBE2', '#4293C7') 

survey_map <- ggplot(data = survey_shp) +
  geom_sf(aes(fill = bin_years)) +
  theme_void() +
  scale_fill_manual(values = color_palette,
                    guide = 'legend') +
  theme(legend.position = "bottom",
        legend.direction = "vertical",
        legend.justification = "left",
        legend.spacing.x = unit(.5,'cm'),
        legend.background = element_blank(),
        #legend.box.spacing = unit(-5, "cm"),
        legend.margin = margin(-9, 0, 0, 3, "cm"),
        plot.margin = margin(-3,-10,0,-1,'cm'),
        plot.background = element_blank()) +
  guides(fill = guide_legend(title = 'Years since last\nconsumption survey',
                             title.position="top", 
                             title.hjust = 0.5,
                             label.position = "left",
                             label.hjust = .5,
                             tickwidth = 0,
                             keywidth = unit(1, "cm"),
                             keyheight = unit(.5, "cm")))

survey_map

ggsave('figures/maps/WB_years_since_survey.png', device = 'png', plot = survey_map, scale = 1,
       width = 15, height = 15, units = 'cm', dpi = 500)



#*******************************************************************************
#### Map survey countries####
#*******************************************************************************

# create a bounding box to keep only countries close to the sampling areas
ylims <- c(-18, 15)
xlims <- c(0, 50)
box_coords <- tibble(x = xlims, y = ylims) %>% 
  st_as_sf(coords = c("x", "y")) %>% 
  st_set_crs(st_crs(africa_boundaries))

bounding_box <- st_bbox(box_coords) %>% st_as_sfc()

cntry_subset <- st_intersection(africa_boundaries, bounding_box) %>% 
  mutate(sample_country = as.factor(ifelse(ISO_A3 %in% lsms_countries, 1, 0)))
levels(cntry_subset$sample_country) <- c("No", "Yes")

cnames <- cntry_subset %>%
  filter(sample_country == 'Yes') %>%
  st_centroid()

coords <- st_coordinates(cnames$geometry)
cnames$lon <- coords[,1]
cnames$lat <- coords[,2]   

cnames <- st_drop_geometry(cnames)

color_palette = c('#E8E9EB', '#2372B6')

cntry_map <- ggplot(data = cntry_subset) +
  geom_sf(aes(fill = sample_country)) +
  theme_void() +
  scale_fill_manual(values = color_palette,
                    guide = 'legend') +
  geom_text(data = cnames, 
            aes(lon, lat, label = ISO_A3), size=5, col = 'black') +
  theme(legend.position = "bottom",
        legend.direction = "vertical",
        legend.justification = "left",
        legend.spacing.x = unit(.5,'cm'),
        legend.background = element_blank(),
        legend.box.spacing = unit(-4, "cm"),
        legend.margin = margin(0, 0, 0, .8, "cm"),
        plot.margin = margin(-6,0,-4,0,'cm'),
        plot.background = element_blank()) +
  guides(fill = guide_legend(title = 'Training country',
                             title.position="top", 
                             title.hjust = 0.5,
                             label.position = "left",
                             label.hjust = .5,
                             tickwidth = 0,
                             keywidth = unit(.5, "cm"),
                             keyheight = unit(.5, "cm")))


ggsave('figures/maps/training_countries.png', device = 'png', plot = cntry_map,scale = 1, 
        width = 15, height = 12, units = 'cm', dpi = 300)


#*******************************************************************************
#### Map clusters in survey countries ####
#*******************************************************************************

# create a bounding box to keep only countries close to the sampling areas
ylims <- c(-18, 15)
xlims <- c(0, 50)
box_coords <- tibble(x = xlims, y = ylims) %>% 
  st_as_sf(coords = c("x", "y")) %>% 
  st_set_crs(st_crs(africa_boundaries))

bounding_box <- st_bbox(box_coords) %>% st_as_sfc()

# subset the African shapefile with those countries that are close to the sample countries
cntry_subset <- st_intersection(africa_boundaries, bounding_box) %>% 
  mutate(sample_country = as.factor(ifelse(ISO_A3 %in% lsms_countries, 1, 0)))
levels(cntry_subset$sample_country) <- c("No", "Yes")

# The lats and lons and consumption expenditure are stored in the cl_df

# define the color palette to map the base map
color_palette = c('#E8E9EB', '#2372B6')

# subset the cntry_subset to the sample countries
sample_sf = cntry_subset %>% filter(sample_country == 'Yes')

cluster_map <- ggplot() +
  geom_sf(data = cntry_subset, fill = '#BAAA94', color = 'white', size = .2, linetype = 'solid') +
  geom_sf(data = sample_sf, fill = '#BAAA94', color = "black", size = 0.5, linetype = "solid") +
  geom_point(
    data = cl_df, 
    aes(
      x = lon, 
      y = lat,
      col = mean_pc_cons_usd_2017
      ),
    size = .2
    )+
  theme_void() +
  scale_color_viridis(option = 'inferno', limits = c(0,10)) +
  theme(legend.position = 'bottom',
        legend.direction = "vertical",
        legend.justification = "left",
        legend.spacing.x = unit(.2,'cm'),
        legend.spacing.y = unit(.4,'cm'),
        legend.background = element_blank(),
        legend.box.spacing = unit(-5, "cm"),
        legend.margin = margin(.2, 0, 0, .2, "cm"), # top, right, bottom, left
        plot.margin = margin(0,0,0,0,'cm'),
        plot.background = element_blank()) +
  guides(color = guide_colorbar(title = 'Daily Consumption\n(2017 int $)',
                             title.position="top", 
                             title.hjust = 0.5,
                             label.position = "left",
                             label.hjust = 0,
                             tickwidth = 0,
                             keywidth = unit(.5, "cm"),
                             keyheight = unit(.5, "cm")))

cluster_map

ggsave('figures/maps/cluster_map.png', device = 'png', plot = cluster_map,scale = 1, 
       width = 15, height = 15, units = 'cm', dpi = 300)


#*******************************************************************************
#### Map Number of clusters per district ####
#*******************************************************************************

uga_shp_2 <- st_read("../Data/geoboundaries/uga_boundaries/UGA_adm2_simplified.geojson")

uga_df <- lsms_df %>% 
  filter(country == 'uga') %>% 
  filter(start_year == 2011) %>% 
  select(cluster_id, lon, lat) %>%
  distinct(cluster_id, .keep_all = T)

uga_df_sf <- st_as_sf(uga_df, coords = c("lon", "lat"), crs = st_crs(uga_shp_2))

joined_data <- st_join(uga_shp_2, uga_df_sf)

# count the number of clusters in each district
district_counts <- joined_data %>%
  filter(!is.na(cluster_id)) %>% 
  group_by(shapeName) %>%
  summarize(Observation_Count = n()) %>% 
  st_drop_geometry()

# merge the count data and bin it
bin_edges <- c(0, 1, 5, 10, 20, Inf)
bin_labels <- c("0", "1 to 5", "5 to 10", "10 to 20", "> 20")
uga_shp_2 %<>% left_join(district_counts, by = 'shapeName') %>% 
  mutate(Observation_Count = ifelse(is.na(Observation_Count),0, Observation_Count))

uga_shp_2$binned_count <- cut(uga_shp_2$Observation_Count, breaks = bin_edges,
                              labels= bin_labels, include.lowest = TRUE, 
                              right = F)


# bin the count variable
color_palette <- c('#808080','#C7DCF0', '#1D92C1','#4293C7', '#2372B6', '#07306B','#1D92C1','#C7DCF0','#2372B6') 
color_palette <- c('#808080','#C7DCF0','#1D92C1','#2372B6','#07306B', '#9FCBE2', '#4293C7') 
uga_samples <- ggplot(data = uga_shp_2) +
  geom_sf(aes(fill = binned_count))+
  scale_fill_manual(values = color_palette,
                    guide = 'legend') +
  theme_void() +
  theme(legend.position = "bottom",
        legend.direction = "horizontal",
        legend.justification = "right",
        legend.spacing.x = unit(0.2,'cm'),
        legend.background = element_blank(),
        #legend.box.spacing = unit(-5, "cm"),
        legend.margin = margin(-2, 5, 0, 3, "cm"),
        plot.margin = margin(-1.5,1,0,0,'cm'),
        plot.background = element_blank()) +
  guides(fill = guide_legend(title = 'Number of clusters per district',
                             title.position="top", 
                             title.hjust = 0.5,
                             label.position = "bottom",
                             label.hjust = .5,
                             tickwidth = 0,
                             keywidth = unit(1, "cm"),
                             keyheight = unit(.5, "cm")))

uga_samples

ggsave('figures/maps/spatial_coverage_uga.png', device = 'png',
       plot = uga_samples, scale = 1, 
       width = 17, height = 17, units = 'cm', dpi = 500)


#*******************************************************************************
#### Map survey clusters####
#*******************************************************************************
uga_shp_2 <- st_read("../Data/geoboundaries/uga_boundaries/UGA_adm2_simplified.geojson")

uga_df <- cl_df %>% 
  filter(country == 'uga')

uga_samples <- ggplot(data = uga_shp_2) +
  geom_sf()+
  geom_point(data = uga_df, aes(x = lon, y = lat), col = '#07306B') +
  # scale_color_viridis(option = 'plasma', 
  #                     name = 'pc 2017\n$/day',
  #                     guide = 'colourbar') +
  theme_void() +
  theme(#legend.position = "top",
        # legend.direction = "vertical",
        # legend.justification = "left",
        # legend.spacing.x = unit(.5,'cm'),
        # legend.background = element_blank(),
        # legend.box.spacing = unit(-7, "cm"),
        # legend.margin = margin(0, 0, 0, 1.5, "cm"),
        plot.margin = margin(1,0,0,0,'cm'),
        plot.background = element_blank()) #+
  # guides(colour = guide_colourbar(
  #   title.position = 'top',
  #   title.hjust = 0.5,
  #   label.position = 'left'))

ggsave('figures/maps/uga_samples.png', device = 'png', plot = uga_samples,scale = 1, 
      width = 15, height = 15, units = 'cm', dpi = 300)


uga_df <- cl_df %>% 
  filter(country == 'uga') %>% 
  mutate(data = as.factor(rbinom(314, 1, 0.2)))
levels(uga_df$data) = c('known', 'unknown')

uga_samples <- ggplot(data = uga_shp_2) +
  geom_sf()+
  geom_point(data = uga_df, aes(x = lon, y = lat, col = data)) +
  scale_color_manual(values = c("#07306B", "red")) +
  theme_void() 


ggsave('figures/maps/uga_samples_to_predict.png', device = 'png', plot = uga_samples,scale = 1, 
        width = 15, height = 15, units = 'cm', dpi = 300)

#*******************************************************************************
#### Map UGA example CV ####
#*******************************************************************************
# load the lsms data with fold information
cv_df <- read.csv("../Data/lsms/processed/ex_cv_split.csv") %>%
  filter(country == 'uga') %>% 
  select(cluster_id, lat, lon, val_fold) %>% 
  distinct() %>% 
  mutate(is_fold_0 = as.factor(ifelse(val_fold == 3,1,0)))

# map the clusters for fold 3
uga_fold <- ggplot(data = uga_shp_2) +
  geom_sf(fill = 'grey', alpha = 0.3)+
  geom_point(data = cv_df, aes(x = lon, y = lat, col = is_fold_0)) +
  scale_color_manual(values = c("#07306B", "red"), 
                     labels = c('Training', 'Validation')) +
  labs(color = "Sample") +
  theme_void() +
  theme(legend.position = "bottom",
        legend.box = "horizontal",
        legend.margin = margin(t = -50, r = 0, b = 10, l = 0))

ggsave('figures/maps/uga_fold_3.png', device = 'png', plot = uga_fold,
       scale = 1, width = 15, height = 15, units = 'cm', dpi = 300)

# map the predictions for fold 3
 
# load the predictions
preds_df <- read.csv("results/predictions/baseline_preds.csv") %>% 
  left_join(cv_df %>% select(cluster_id, val_fold, lat, lon), by = 'cluster_id') %>% 
  left_join(lsms_df %>% select(unique_id, start_year), by = 'unique_id') %>% 
  filter(val_fold == 3)

# map the predictions
preds_map_09 <- ggplot(data = uga_shp_2) +
  geom_sf(fill = 'grey', alpha = 0.3)+
  geom_point(data = preds_df %>% filter(start_year == 2009), aes(x = lon, y = lat, col = y_hat)) +
  theme_void() +
  scale_color_viridis(option = 'plasma', 
                      name = 'log pc $/day',
                      guide = 'colourbar')

preds_map_10 <- ggplot(data = uga_shp_2) +
  geom_sf(fill = 'grey', alpha = 0.3)+
  geom_point(data = preds_df %>% filter(start_year == 2010), aes(x = lon, y = lat, col = y_hat)) +
  theme_void() +
  scale_color_viridis(option = 'plasma', 
                      name = 'log pc $/day',
                      guide = 'colourbar')


preds_map_11 <- ggplot(data = uga_shp_2) +
  geom_sf(fill = 'grey', alpha = 0.3) +
  geom_point(data = preds_df %>% filter(start_year == 2011), aes(x = lon, y = lat, col = y_hat)) +
  theme_void() +
  scale_color_viridis(option = 'plasma', 
                      name = '',  # Set legend title to empty string
                      guide = 'colourbar') + 
  theme(legend.position = "bottom",
        legend.box = "horizontal",
        legend.key.size = unit(1, "cm"), # Adjust the width of the legend key
        legend.key.height = unit(.5, "cm")) +  # Adjust the height of the legend key
  guides(color = guide_colorbar(title.position = "top")) 

ggsave('figures/maps/preds_uga_3_2009.png', device = 'png', plot = preds_map_09,
       scale = 1, width = 15, height = 15, units = 'cm', dpi = 300)

ggsave('figures/maps/preds_uga_3_2010.png', device = 'png', plot = preds_map_10,
       scale = 1, width = 15, height = 15, units = 'cm', dpi = 300)

ggsave('figures/maps/preds_uga_3_2011.png', device = 'png', plot = preds_map_11,
       scale = 1, width = 15, height = 15, units = 'cm', dpi = 300)

#*******************************************************************************
#### Map Recentered locations boxes ####
#*******************************************************************************

# select two-three clusters that are close to each other
#sel_cls <- c("uga_3040002", 'uga_3040007', 'uga_3040004', 'uga_3040009' )
#sel_cls <- c('uga_2090018', 'uga_2090029', 'uga_2090026')
#sel_cls <- c('eth_071801088800101','eth_071801088801001', 'eth_071801088802302')
sel_cls <- c('tza_05-03-073-05-012', 'tza_5308321','tza_05-03-083-03-311', 'tza_05-04-011-01-008')
sel_cls_df <- cl_df %>% 
  filter(cluster_id %in% sel_cls)

sel_cls_sf <- st_as_sf(sel_cls_df, coords = c('lon', 'lat'), crs = 4326)
circles <- st_buffer(sel_cls_sf$geometry, dist = (224*30)/2) 
bboxes <- c()
for(i in 1:length(sel_cls)){
  bboxes[i] = st_bbox(circles[i]) %>% 
    st_as_sfc() %>%
    st_as_sf()
}

c_lat = median(sel_cls_df$lat)
c_lon = median(sel_cls_df$lon)

map <- leaflet() %>%
  setView(lng = c_lon, lat = c_lat, zoom = 12) %>%
  addTiles() %>%
  addCircleMarkers(data = sel_cls_df,
                   lng = ~lon, lat = ~lat,
                   color = "red", radius = 2, fill = 'red', 
                   opacity = 1) %>% 
  addCircleMarkers(data = sel_cls_df, 
                   lng = ~lsms_lon, lat = ~lsms_lat,
                   color = 'black', radius = 2, fill = 'black',
                   opacity = 1)

for(i in 1:length(sel_cls)){
  map <- map %>% 
    addPolygons(data = bboxes[[i]], color = "blue",
                fillOpacity = .15, opacity = 0.4,weight = 2)
}

map <- map %>%
  addLegend(
    position = "topright",
    colors = c("black", "red", "blue"),
    labels = c("Original Location", "Recentered Location", "Cluster Grid Cell"),
    opacity = .8,
    labFormat = 'factor'
  )


map

export_png(map, 'figures/maps', 'bounding_boxes_tza')

##### map a single cluster #####

example_cluster <- 'uga_3040004'
example_unique_id <- 'uga_3040004_2018'

rect_buffer <- bboxes[[2]]

sel_cls_df <- cl_df %>% 
  filter(cluster_id == example_cluster)

map <- leaflet() %>%
  setView(lng = sel_cls_df$lon, lat = sel_cls_df$lat, zoom = 14) %>%
  addTiles() %>%
  addPolygons(data = rect_buffer, color = "blue",
              fillOpacity = 0, weight = .3)
  
export_png(map, 'figures/maps', 'osm_example_roi')











