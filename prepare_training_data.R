library(Seurat)
library(data.table)
library(tidyverse)

data <- Load10X_Spatial(
  data.dir = "/workspace/sptx/stand2/data/Adult_Mouse_Brain_FFPE",
  filename = "Visium_FFPE_Mouse_Brain_filtered_feature_bc_matrix.h5",
  assay = "Spatial"
)

# data <- SCTransform(data, assay = "Spatial", verbose = FALSE)
data <- SCTransform(data, assay = "Spatial", return.only.var.genes = FALSE, verbose = FALSE)

sct_data_df <- as.data.frame(as.matrix(data@assays$SCT@data))
sct_data_df %>% fwrite(file = "data.csv", row.names = T)

sct_rownames <- as.data.frame(row.names(sct_data_df))
sct_rownames %>% fwrite(file = "gene_list.csv", row.names = F, col.names = F, )
