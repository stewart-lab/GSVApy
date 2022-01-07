from collections import defaultdict
from collections import OrderedDict

import sys
import os
import rmstime
import rmscmdline
import rmslogging    
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pdb
import statsmodels.stats.multitest as sm

COMMAND_LINE_DEF_FILE = "./runMannWhitneyCommandLine.txt"

def main():
    (start_time_secs, pretty_start_time) = rmstime.get_time_and_pretty_time()
    print("pretty_start:", pretty_start_time)
    
    my_args = rmscmdline.get_args(start_time_secs, pretty_start_time, COMMAND_LINE_DEF_FILE)
    logfile = rmslogging.open_log_file(my_args["log_file"])
    rmslogging.write_args_and_files(my_args, sys.argv[0], COMMAND_LINE_DEF_FILE, logfile)
    
    in_file = my_args["in_file"]
    out_file = os.path.join(my_args["out_dir"], "output_mw.tsv")
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
    mw_stat_dict = OrderedDict()
    mw_pval_dict = OrderedDict()
    mw_adj_pval_dict = OrderedDict()
    mw_pval_list = []
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
        out_mw.write(f"Term\tMW Stat\tMW P Value\tMW Adj P Value\n")  
        for _, row in df.iterrows():
            mw = stats.mannwhitneyu(x=row[cond1_names].astype(float), y=row[cond2_names].astype(float), alternative="two-sided")       
           
           
            mw_stat_dict[row[0]] = mw.statistic
            mw_pval_dict[row[0]] = mw.pvalue
            mw_pval_list.append(mw.pvalue)
            #print ("mw.pvalue:", mw.pvalue)
            #out_mw.write(f"{row[0]}\t{mw.statistic}\t{mw.pvalue}\n")
           
        rej, adj_pvals, _, _ = sm.multipletests(mw_pval_list, alpha=0.05, method='fdr_bh')      
        for i, term in enumerate(mw_stat_dict.keys()): 
            #print("i:", i, " adjpval:", adj_pvals[i])
            mw_adj_pval_dict[term] = adj_pvals[i]
        for term in mw_stat_dict.keys():
            if mw_pval_dict[term] < 0.05:
                out_mw.write(f"{term}\t{mw_stat_dict[term]}\t{mw_pval_dict[term]}\t{mw_adj_pval_dict[term]}\n")
                        
    rmslogging.close_log_file(logfile)  
    (end_time_secs, x) = rmstime.get_time_and_pretty_time()
    total_elapsed_time = end_time_secs - start_time_secs
    print("All done. Total elapsed time: " + str(total_elapsed_time) + " seconds.\n")      

if __name__ == "__main__":
    main()