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
  guides(fill = guide_legend(title = 'Years since last survey',
                             title.position="top", 
                             title.hjust = 0.5,
                             label.position = "left",
                             label.hjust = .5,
                             tickwidth = 0,
                             keywidth = unit(1, "cm"),
                             keyheight = unit(.5, "cm")))

survey_map

ggsave('figures/maps/WB_years_since_survey.png', device = 'png', plot = survey_map, scale = 1,
       width = 15, height = 15, units = 'cm', dpi = 300)



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
#### Map bounding boxes ####
#*******************************************************************************

# select two-three clusters that are close to each other
sel_cls <- c("uga_3040002", 'uga_3040007', 'uga_3040004', 'uga_3040009' )
sel_cls <- c('uga_2090018', 'uga_2090029', 'uga_2090026')
sel_cls <- c('eth_071801088800101','eth_071801088801001', 'eth_071801088802302')
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
                   color = "red", radius = 2, fill = 'red') %>% 
  addCircleMarkers(data = sel_cls_df, 
                   lng = ~lsms_lon, lat = ~lsms_lat,
                   color = 'black', radius = 2, fill = 'black')

for(i in 1:length(sel_cls)){
  map <- map %>% 
    addPolygons(data = bboxes[[i]], color = "blue",
                fillOpacity = .1, opacity = 0.2,weight = 2)
}

map <- map %>%
  addLegend(
    position = "bottomright",
    colors = c("black", "red", "blue"),
    labels = c("Original Location", "Recentered Location", "Region of Interest"),
    title = "Legend"
  )

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
              fillOpacity = 0, weight = .4)
  
export_png(map, 'figures/maps', 'osm_example_roi')











