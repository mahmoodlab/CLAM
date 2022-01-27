# **************************************************************************************************
# Functionality of this script:
## performing the hierarchical clustering with Ward.D2 linkage and Euclidean distance
## plotting clustered heatmap (with values normalized to [-1, 1] only for visualization)
## retrieving sample clusters

### To reproduce the same heatmap, you may need to adjust the cluster name and value of arg 'isReverse' in the dendsort function.

# **************************************************************************************
# Install package, only do it for the first time
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("CancerSubtypes")

install.packages("RColorBrewer")

install.packages('devtools')
devtools::install_github("jokergoo/ComplexHeatmap")

install.packages('dendextend')
install.packages("dendsort")
install.packages("stringr")
# **************************************************************************************

library(CancerSubtypes)
library(RColorBrewer)
library(ComplexHeatmap)
library('dendextend')
library(dendsort)
library(stringr)

# **************************************************************************************
series = 'tcga' # tcga, mondor

path <- paste(getwd(), "gene_clust", "results", series, sep = "/")

gene_signature = "T-cell_Exhaustion" # 6G_Interferon_Gamma, Gajewski_13G_Inflammatory, Inflammatory, Interferon_Gamma_Biology, Ribas_10G_Interferon_Gamma, T-cell_Exhaustion

# Set cluster number
ncluster_row = 2
ncluster_col = 3

# Sort and reorder dendrogram nodes
sreorder = TRUE

# Rename sample cluster
## May need to change here to reproduce the same heatmap
cluster01 <- "Cluster High" # Cluster High, Cluster Median, Cluster Low
cluster02 <- "Cluster Median"
cluster03 <- "Cluster Low"

# Exported figure size
width = 23
height = 10

# ***************************************************************************************
# Load data
if (series == "tcga") {
  data <- as.matrix(as.data.frame(read.csv(paste(path, "fpkm_final_add1-log2-zscore_336.csv", sep = "/"), sep="\t", row.names = 1)))
} else if (series == "mondor") { 
  data <- as.matrix(as.data.frame(read.csv(paste(path, "mondor_final_log2-zscore_139.csv", sep = "/"), sep="\t", row.names = 1)))
}
  
# Filter data by the signature
if (gene_signature == "Inflammatory") {
  geneS = c("CD274", "CD8A", "LAG3", "STAT1")
} else if (gene_signature == "Gajewski_13G_Inflammatory")  {
  geneS = c("CCL2","CCL4","CD8A","CXCL10","CXCL9","GZMK","HLA-DMA","HLA-DMB","HLA-DOA","HLA-DOB","ICOS","IRF1")
} else if (gene_signature == "6G_Interferon_Gamma")  { # exactly the same as the IFN-γ in Melanoma
  geneS = c("CXCL10","CXCL9","HLA-DRA","IDO1","IFNG","STAT1")
} else if (gene_signature == "Interferon_Gamma_Biology")  {
  geneS = c("CCL5","CD27","CXCL9","CXCR6","IDO1","STAT1")
} else if (gene_signature == "T-cell_Exhaustion")  {
  geneS = c("CD274","CD276","CD8A","LAG3","PDCD1LG2","TIGIT")
} else if (gene_signature == "Ribas_10G_Interferon_Gamma")  {
  geneS = c("CCR5","CXCL10","CXCL11","CXCL9","GZMA","HLA-DRA","IDO1","IFNG","PRF1","STAT1")
}
data <- data[geneS,]

# **************************************************************************************
# Clustering
dist_row = dist(data, method = "euclidean", upper=TRUE)
dist_col = dist(t(data), method = "euclidean", upper=TRUE)

hc_row = hclust(dist_row, method = "ward.D2")
hc_col = hclust(dist_col, method = "ward.D2")

if (sreorder) {
  ## May need to change value of arg 'isReverse' to reproduce the same heatmap
    hc_col = dendsort(hclust(dist(t(data), method = "euclidean"), method = "ward.D2"), isReverse = TRUE, type="average")
  sreorder <- "_reorder"
} else {
  sreorder <- ""
}

# Cut trees for fixed number
# order_clusters_as_data = FALSE: to organize the clusters by the labels in the dendrogram
### function "stats::cutree" may not return the same order as the tree, will cause problem when using "column_order(ht)"
### So be sure to use dendextend::cutree with "order_clusters_as_data = FALSE"
cluster_row <- dendextend::cutree(tree = hc_row, k = ncluster_row, order_clusters_as_data = FALSE) 
cluster_col <- dendextend::cutree(tree = hc_col, k = ncluster_col, order_clusters_as_data = FALSE)

# Make the order same as that of data (necessary following the dendextend::cutree)
cluster_row = cluster_row[rownames(data)]
cluster_col = cluster_col[colnames(data)]

# **************************************************************************************
# Reconstruct clusters
cluster_row <- data.frame("Gene" = as.character(stack(cluster_row)[,2]), "Cluster" = paste('Cluster', str_pad(stack(cluster_row)[,1], 2, "left", "0")))
cluster_col <- data.frame("Sample" = as.character(stack(cluster_col)[,2]), "Cluster" = paste('Cluster', str_pad(stack(cluster_col)[,1], 2, "left", "0")))

# rename and reorder sample clusters
if (ncluster_col == 3) {
  cluster_col[cluster_col=="Cluster 01"] <- cluster01
  cluster_col[cluster_col=="Cluster 02"] <- cluster02
  cluster_col[cluster_col=="Cluster 03"] <- cluster03
}

# Reconstruct clusters for heatmap
cluster_row_named <- cluster_row
cluster_col_named <- cluster_col

cluster_row <- data.frame("Gene" = cluster_row$Cluster) # Gene Clusters
cluster_col <- data.frame("Sample" = cluster_col$Cluster) # Sample Clusters

# **************************************************************************************
# Generate distinctive color for cluster labeling
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))
set.seed(695032) 

# Set red, yellow and blue for Cluster High, Median and Low, respectively, if set 3 sample clusters
if (ncluster_col == 3) {
  col_anno_colors <- c("#E41A1C","#FFD92F","#B3CDE3")
  #col_anno_colors <- c("#E41A1C","#1300ff","#29df2c")
  names(col_anno_colors) <- c("Cluster High", "Cluster Median", "Cluster Low")
} else {
  col_anno_colors = sample(col_vector, ncluster_col)
  # Expected a named list
  names(col_anno_colors) <- unique(cluster_col)[,1]}

row_anno_colors = sample(col_vector, ncluster_row)
#row_anno_colors = c("#1300ff", "#29df2c")
names(row_anno_colors) <- unique(cluster_row)[,1]

# Visualization
par(mfrow=c(1,2)) 
pie(rep(1,length(row_anno_colors)), col=row_anno_colors, labels=names(row_anno_colors))
pie(rep(1,length(col_anno_colors)), col=col_anno_colors, labels=names(col_anno_colors))

# *************************************************************************************
# Plot heatmap

# Scale rows to range [-1, 1] for visualization
data_scaled <- t(apply(data, 1, function(x) (((x-min(x))/(max(x)-min(x)) * 2 - 1))))

# heatmap settings
title <- paste("Clustered heatmap: HC Ward.D2 Euclidean")
cluster_rows = hc_row
cluster_cols = hc_col
cutree_rows = ncluster_row
cutree_cols = ncluster_col
gaps_row = NULL
gaps_col = NULL

# heatmap
my_heatmap <- function(bottom_annotation = NULL) {
  pheatmap(data_scaled,
           main = title,
           name = "Normalized FPKM Matrix",
           color = rev(brewer.pal(n = 11, name = "RdYlBu")),
           #color = turbo(11, alpha = 1, begin = 0, end = 1), 
           #color = colorRampPalette(c("green", "black", "red"))(n = 1000),
           border_color = NA, 
           cellwidth = 3, cellheight = 40,
           show_rownames = FALSE, show_colnames = FALSE,
           cluster_rows = cluster_rows, 
           cluster_cols = cluster_cols,
           cutree_rows = cutree_rows,
           cutree_cols = cutree_cols,
           gaps_row = gaps_row,
           gaps_col = gaps_col,
           left_annotation = rowAnnotation(df = cluster_row,
                                           col = list('Gene' = row_anno_colors), # Gene.Clusters
                                           annotation_name_side = "top", show_annotation_name = FALSE,
                                           #show_legend = c("bar" = FALSE), # turn off color bar/legend
                                           annotation_name_rot = 90),
           top_annotation = HeatmapAnnotation(df = cluster_col, 
                                              col = list('Sample' = col_anno_colors),   # # Sample.Clusters, annotation_name_rot = 90) only 4 angles
                                              #show_legend = c("bar" = FALSE), # turn off color bar/legend
                                              show_annotation_name = FALSE),
           bottom_annotation = bottom_annotation,
           annotation_legend = TRUE, 
           #legend=FALSE, # turn off color bar/legend 
           scale = 'none')
}

# visualize heatmap
ht = draw(my_heatmap()) 

# ***********************************************************************************
# Retrieve and export sample clusters
 
ccl.list <- column_order(ht) # Extract clusters (output is a list)

lapply(ccl.list, function(x) length(x))  # check/confirm size clusters

# Loop to extract samples for each cluster.
for (i in 1:length(column_order(ht))){ # to loop the clusters
  # samples belonging to this cluster
  # drop=FALSE to keep as matrix when only 1 column
  cclu <- colnames(data[, column_order(ht)[[i]], drop = FALSE]) 
  if (i == 1) {
    cout <- cbind(cclu, paste("Cluster ", str_pad(i, 2, "left", "0"), sep=""))
    colnames(cout) <- c("Sample", "Cluster")
  } else {
    cclu <- cbind(cclu, paste("Cluster ", str_pad(i, 2, "left", "0"), sep=""))
    cout <- rbind(cout, cclu)
  }
}

# rename retrieved clusters
cout[cout=="Cluster 01"] <- cluster01
cout[cout=="Cluster 02"] <- cluster02
cout[cout=="Cluster 03"] <- cluster03

# export
write.table(cout, file= paste(path, "/sample_clusters_",gene_signature,"_zscore_hc_ward.D2_euclidean_",ncluster_col,sreorder,".csv", sep = ""),
            sep=",", quote=F, row.names=FALSE)

# Set all gene names to black
row_label_colors = rep("#000000", dim(cluster_row)[1])

# ***********************
# Color sample labels by clusters

for (i in 1:dim(cluster_col)[1]){
  if (i == 1) {
    tmp <- cluster_col[i,1]
    col_label_colors <- cbind(tmp, paste(col_anno_colors[cluster_col[i,1]]))
  } else {
    tmp <- cluster_col[i,1]
    tmp <- cbind(tmp, paste(col_anno_colors[cluster_col[i,1]]))
    col_label_colors <- rbind(col_label_colors, tmp)
  }
}
rm(tmp)
col_label_colors = col_label_colors[, 2]

# ************************************************************************************
# Plot final heatmap

# Reconstruct heatmap
b <- HeatmapAnnotation(goo = anno_text(colnames(data), gp = gpar(fontsize = 0.5, col = col_label_colors)))

q <- my_heatmap(bottom_annotation = b) + rowAnnotation(foo = anno_text(rownames(data), rot = 16,
                                                                       gp = gpar(fontsize = 28, col = row_label_colors)))

# Export heatmap
pdf(paste(path, "/clustermap_",gene_signature,"_zscore_hc_ward.D2_euclidean_",ncluster_row,"_",ncluster_col,sreorder,".pdf", sep = ""),
      compress=FALSE, width=width, height=height) # 

draw(q)

# Close heatmap
dev.off()
#dev.off()
