from collections import defaultdict
from collections import OrderedDict

import sys
import os
import rmstime
import rmscmdline
import rmslogging    
import pdb

COMMAND_LINE_DEF_FILE = "./trimGeneSetsCommandLine.txt"

def main():
    (start_time_secs, pretty_start_time) = rmstime.get_time_and_pretty_time()
    print("pretty_start:", pretty_start_time)
    
    my_args = rmscmdline.get_args(start_time_secs, pretty_start_time, COMMAND_LINE_DEF_FILE)
    logfile = rmslogging.open_log_file(my_args["log_file"])
    rmslogging.write_args_and_files(my_args, sys.argv[0], COMMAND_LINE_DEF_FILE, logfile)
    
    in_genes = my_args["in_genes"]
    in_genesets = my_args["in_genesets"]
    out_file = os.path.join(my_args["out_dir"], "output_genesets.tsv")
    num_genes_required = int(my_args["num_genes_required"])
    num_top_genes_from_in_genes = int(my_args["num_top_genes_from_in_genes"])
   
    # Load data from in_genes into a dictionary
    in_gene_dict = {}
    with open(in_genes, 'r') as in_genes_f:
        for i, line in enumerate(in_genes_f.readlines()):
            if i == 0:
                continue
            if i > num_top_genes_from_in_genes:
                break
            cols = line.strip().split("\t")
            gene = cols[0]
            in_gene_dict[gene] = "X"
            
    with open(out_file, 'w') as out_f, open(in_genesets, 'r') as in_geneset_f:
        for line in in_geneset_f.readlines():
            cols = line.strip().split("\t")
            genes = cols[2:]  #using a gene set file that has a column for the gene set name, and then a column of a hyperlink of where it came from, then genes after that.
            gene_match_ctr = 0
            for gene in genes:
                #print (gene)
                if gene in in_gene_dict:
                    gene_match_ctr = gene_match_ctr + 1
            if gene_match_ctr >= num_genes_required:
                out_f.write(line)
    
    rmslogging.close_log_file(logfile)  
    (end_time_secs, x) = rmstime.get_time_and_pretty_time()
    total_elapsed_time = end_time_secs - start_time_secs
    print("All done. Total elapsed time: " + str(total_elapsed_time) + " seconds.\n")      

if __name__ == "__main__":
    main()