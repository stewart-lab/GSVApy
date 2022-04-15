from collections import defaultdict
from collections import OrderedDict

import sys
import os
import cmdlogtime   
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pdb
import statsmodels.stats.multitest as sm

COMMAND_LINE_DEF_FILE = "./runMannWhitneyCommandLine.txt"
WRITE_MSGS_TO_STDOUT = 1
def main():
    (start_time_secs, pretty_start_time, my_args, logfile) = cmdlogtime.begin(COMMAND_LINE_DEF_FILE, WRITE_MSGS_TO_STDOUT)
    
    in_file = my_args["in_file"]
    paired_data = my_args["paired_data"]
    
    if paired_data:
        out_file_name = "output_wilcoxonpaired.tsv"
    else:
        out_file_name = "output_mw.tsv"    
    out_file = os.path.join(my_args["out_dir"], out_file_name)
    out_wilks = os.path.join(my_args["out_dir"], "output_wilks.tsv")
    cond1_names = my_args["cond1_names"]
    cond2_names = my_args["cond2_names"]
    
    
    cond1_names = cond1_names.split(",")
    cond2_names = cond2_names.split(",")
    for cond in cond1_names:
        print("cond1:", cond)
    for cond in cond2_names:
        print("cond2:", cond)
   
    # Load data from input File
    df = pd.read_csv(in_file, header=0,  sep="\t")
    print(df.head(2))
    # just for fun, make a box plot
    #df.boxplot(column=[cond1_names[0], cond2_names[0]], grid=False)
    #plt.show()
    
    # check to see if the data is normally distributed in each row. Probably not enough data in each row to know,
    #  so probably this isn't that useful. I'm putting the code in and you can run it by typing -show_w on command line
    #  if most p-values are < 0.05, then the data is not normally distributed, and you should use a
    #  mann-whitney test, and NOT a t-test.
    #df2 = df[cond1_names + cond2_names]
    stat_dict = OrderedDict()
    pval_dict = OrderedDict()
    adj_pval_dict = OrderedDict()
    pval_list = []
    with open(out_file, 'w') as out_mw:
        if (my_args["show_wilks"]):
            with open(out_wilks, 'w') as out_wilks:
                normal_count = 0
                not_normal_count = 0
                out_wilks.write(f"Normality\tTerm\tw\tpvalue\n")  
                for _, row in df.iterrows():
                    w, pvalue = stats.shapiro(row[cond1_names + cond2_names])
                    if pvalue < 0.05:
                        out_wilks.write(f"NO\t{row[0]}\t{w}\t{pvalue}\n")
                        not_normal_count = not_normal_count + 1
                    else:
                        out_wilks.write(f"YES\t{row[0]}\t{w}\t{pvalue}\n")  
                        normal_count = normal_count + 1  
                out_wilks.write(f"Normal Count\t{normal_count}\tNOT Normal Count\t{not_normal_count}\n")          
        out_mw.write(f"Term\tStat\tP Value\tAdj P Value\n")  
        for _, row in df.iterrows():
            if paired_data:
                stat_obj = stats.wilcoxon(x=row[cond1_names].astype(float), y=row[cond2_names].astype(float), alternative="two-sided")       
           
            else:
                stat_obj = stats.mannwhitneyu(x=row[cond1_names].astype(float), y=row[cond2_names].astype(float), alternative="two-sided")       
           
           
            stat_dict[row[0]] = stat_obj.statistic
            pval_dict[row[0]] = stat_obj.pvalue
            pval_list.append(stat_obj.pvalue)
            #print ("pvalue:", stat_obj.pvalue)
            
            #out_mw.write(f"{row[0]}\t{mw.statistic}\t{mw.pvalue}\n")
           
        rej, adj_pvals, _, _ = sm.multipletests(pval_list, alpha=0.05, method='fdr_bh')      
        for i, term in enumerate(stat_dict.keys()): 
            #print("i:", i, " adjpval:", adj_pvals[i])
            adj_pval_dict[term] = adj_pvals[i]
        for term in stat_dict.keys():
            if pval_dict[term] < 1.1:  #  Used to be < 0.05, now, let's let them all through.
                out_mw.write(f"{term}\t{stat_dict[term]}\t{pval_dict[term]}\t{adj_pval_dict[term]}\n")
                        
    cmdlogtime.end(logfile, start_time_secs, WRITE_MSGS_TO_STDOUT)      

if __name__ == "__main__":
    main()