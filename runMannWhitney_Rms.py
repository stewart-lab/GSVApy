from collections import defaultdict
from collections import OrderedDict, defaultdict

import sys
import os
import cmdlogtime   
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pdb
import statsmodels.stats.multitest as sm
import random

COMMAND_LINE_DEF_FILE = "./runMannWhitneyCommandLine.txt"
SHUFFLE_LABELS = False
MW = True  #  True means do MannWhitney test. False means do t-test
#MW = False
def main():
    (start_time_secs, pretty_start_time, my_args, logfile) = cmdlogtime.begin(COMMAND_LINE_DEF_FILE)
    in_file = my_args["in_file"]
    paired_data = my_args["paired_data"]
    if MW: 
        if paired_data:
            out_file_name = "output_wilcoxonpaired.tsv"
        else:
            out_file_name = "output_mw.tsv" 
    else:
        if paired_data:
            out_file_name = "output_t_paired.tsv"
        else:
            out_file_name = "output_t.tsv"   
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
    cond1_len = len(cond1_names)
    #cond2_len = len(cond2_names)
    
    all_conds  = cond1_names + cond2_names
    print(all_conds)
    print(cond1_len)
    #print(cond2_len)
    shuf_conds = all_conds.copy()
    
    # check to see if the data is normally distributed in each row. Probably not enough data in each row to know,
    #  so probably this isn't that useful. I'm putting the code in and you can run it by typing -show_w on command line
    #  if most p-values are < 0.05, then the data is not normally distributed, and you should use a
    #  mann-whitney test, and NOT a t-test.
    #df2 = df[cond1_names + cond2_names]
    stat_dict = OrderedDict()
    pval_dict = OrderedDict()
    adj_pval_dict = OrderedDict()
    adj_empir_row_pval_dict = OrderedDict()
    pval_list = []

    if (my_args["show_wilks"]):
        with open(out_wilks, 'w') as out_wilks:
            normal_count = 0
            not_normal_count = 0
            out_wilks.write(f"Normality\tTerm\tw\tpvalue\n")  
            for _, row in df.iterrows():
                #import pdb
                #pdb.set_trace()
                if (row.loc['PERMUTED'] != 0):
                    break
                w, pvalue = stats.shapiro(row[cond1_names + cond2_names])
                if pvalue < 0.05:
                    out_wilks.write(f"NO\t{row[0]}\t{w}\t{pvalue}\n")
                    not_normal_count = not_normal_count + 1
                else:
                    out_wilks.write(f"YES\t{row[0]}\t{w}\t{pvalue}\n")  
                    normal_count = normal_count + 1  
            out_wilks.write(f"Normal Count\t{normal_count}\tNOT Normal Count\t{not_normal_count}\n")          
        
    for _, row in df.iterrows():
        if (row.loc['PERMUTED'] != 0):
            break
        stat_obj = calc_stats(cond1_names, cond2_names, paired_data, row,  MW)
        #if paired_data:
        #    stat_obj = stats.wilcoxon(x=row[cond1_names].astype(float), y=row[cond2_names].astype(float), alternative="two-sided")       
        #else:
        #    stat_obj = stats.mannwhitneyu(x=row[cond1_names].astype(float), y=row[cond2_names].astype(float), alternative="two-sided")       
           
        stat_dict[row[0]] = stat_obj.statistic
        pval_dict[row[0]] = stat_obj.pvalue
        pval_list.append(stat_obj.pvalue)
    # Here is where I would do a bunch of permutations of the data, then somehow report an empirical p-value in addition to b+h
    #  THis is for an empirical p-value based on swapping the column labels (sample names)
    print(df)
    if SHUFFLE_LABELS:
        empir_label_pvals = calculate_empirical_label_pvalues(df, shuf_conds, paired_data, cond1_len, pval_dict)
    else:
        empir_row_pvals = calculate_empirical_row_pvalues(df, paired_data, pval_dict, cond1_names, cond2_names)
    empir_row_pval_list = []
    for key,  pval in empir_row_pvals.items():
        #print ("key: ", key,  "pval:", pval)
        #import pdb
        #pdb.set_trace()
        empir_row_pval_list.append(float(pval))
    adj_empir_row_pvals = []
    with open(out_file, 'w') as out_mw: 
        if MW:
            pval_type = "MW P Value"
        else:
             pval_type = "t P Value"
        if SHUFFLE_LABELS:
            out_mw.write(f"Term\tStat\t{pval_type}\tAdj P Value (B+H)\tEmpirical P Value (shuffle Labels)\n")  
        else:
            out_mw.write(f"Term\tStat\t{pval_type}\tAdj P Value (B+H)\tEmpirical P Value (shuffle rows)\tAdj Empir Row Pval\n")        
        rej, adj_pvals, _, _ = sm.multipletests(pval_list, alpha=0.05, method='fdr_bh')      
        for i, term in enumerate(stat_dict.keys()): 
            adj_pval_dict[term] = '{:.4f}'.format(adj_pvals[i])
        rej, adj_empir_row_pvals, _, _ = sm.multipletests(empir_row_pval_list, alpha=0.05, method='fdr_bh')      
        for i, term in enumerate(stat_dict.keys()): 
            adj_empir_row_pval_dict[term] = '{:.4f}'.format(adj_empir_row_pvals[i])    
        for term in stat_dict.keys():
            if pval_dict[term] < 1.1:  #  Used to be < 0.05, now, let's let them all through.
                if SHUFFLE_LABELS:
                    out_mw.write(f"{term}\t{stat_dict[term]}\t{pval_dict[term]}\t{adj_pval_dict[term]}\t{empir_label_pvals[term]}\n")
                else:
                    out_mw.write(f"{term}\t{stat_dict[term]}\t{pval_dict[term]}\t{adj_pval_dict[term]}\t{empir_row_pvals[term]}\t{adj_empir_row_pval_dict[term]}\n")      
    cmdlogtime.end(logfile, start_time_secs)    
# --------------------------------------- FUNCTIONS ---------------------------------  
def calc_stats(cond1_names, cond2_names, paired_data, row, MW):
    if MW:
        if paired_data:
            stat_obj = stats.wilcoxon(x=row[cond1_names].astype(float), y=row[cond2_names].astype(float), alternative="two-sided")       
        else:
            stat_obj = stats.mannwhitneyu(x=row[cond1_names].astype(float), y=row[cond2_names].astype(float), alternative="two-sided")  
    else:  # t-test.  
        if paired_data:
            stat_obj = stats.ttest_rel(a=row[cond1_names].astype(float), b=row[cond2_names].astype(float), permutations=0)  
        else:
            stat_obj = stats.ttest_ind(a=row[cond1_names].astype(float), b=row[cond2_names].astype(float), permutations=0)
    return stat_obj

def calculate_empirical_label_pvalues(df, shuf_conds, paired_data, cond1_len, pval_dict):
    empir_label_pvals = OrderedDict()
    NUM_PERMS = 100  # RMS, make this an input parameter for label permutation
    perm_fraction = 1/NUM_PERMS
    print(perm_fraction)
    ctr = 0
    for _, row in df.iterrows():
        ctr = ctr +1
        perm_pvals = []
        for i in range(0,NUM_PERMS):  
            random.shuffle(shuf_conds)
            #print("shuf_conds:", shuf_conds)
            shuf_cond1_names = shuf_conds[:cond1_len]
            shuf_cond2_names = shuf_conds[cond1_len:]
            #print("sc1:", shuf_cond1_names)
            #print("sc2:", shuf_cond2_names)
            stat_obj = calc_stats(shuf_cond1_names, shuf_cond2_names, paired_data, row, MW)
            perm_pvals.append(stat_obj.pvalue)
        perm_pvals.sort()
        current_fraction = perm_fraction
        #print("calculating permpval for ", pval_dict[row[0]])
        got_pval = False
        for perm_pval in perm_pvals:
            if pval_dict[row[0]] <= perm_pval:  # RMS. SHould this be less than, or less than or equal to?  Less than is more conservative
                this_perm_pval = '{:.4f}'.format(current_fraction)
                empir_label_pvals[row[0]] = this_perm_pval
                #print("calculated pval based on permutation: ", this_perm_pval)
                got_pval = True
                break
            #print("perm_pval:", perm_pval)
            current_fraction = current_fraction + perm_fraction
        #sys.exit()
        if (not got_pval):
            empir_label_pvals[row[0]]  = 1
            #print("calculated pval based on permutation, set to 1: ")
        
        #if (ctr >  20):
        #    break
    
    #print(calc_permed_pvals)        
    #sys.exit()                
    return empir_label_pvals
    
def calculate_empirical_row_pvalues(df, paired_data, pval_dict, cond1_names, cond2_names):
    empir_row_pvals = OrderedDict()
    perm_pvals_by_term = defaultdict(list)
    
    ctr = 0
    perm_to_process = 0
    print ("in calc empir row")
    # calculate all the pvalues based on the permuted row data and stuff into perm_pvals_by_term
    for _, row in df.iterrows():
        #print("in for")
        #import pdb
        #pdb.set_trace()
        if (row.loc['PERMUTED'] == 0): 
            continue
        
        perm_to_process = int(row.loc['PERMUTED'])
        #print ("process this,", row , ' as  perm#:', perm_to_process)
        
        ctr = ctr +1
        perm_pvals = []
        stat_obj = calc_stats(cond1_names, cond2_names, paired_data, row, MW)
        perm_pvals_by_term[row[0]].append(stat_obj.pvalue)
   
    NUM_PERMS = perm_to_process
    perm_fraction = 1/NUM_PERMS
    print(perm_fraction)   
    # now take all these permuted pvals and calculate an empirical pval  
    for _, row in df.iterrows():
        perm_pvals_by_term[row[0]].sort()
        current_fraction = perm_fraction
        #print("calculating permpval for ", pval_dict[row[0]])
        got_pval = False
        for perm_pval in perm_pvals_by_term[row[0]]:
            if pval_dict[row[0]] <= perm_pval:  # RMS. SHould this be less than, or less than or equal to?  Less than is more conservative
                this_perm_pval = '{:.4f}'.format(current_fraction)
                empir_row_pvals[row[0]] = this_perm_pval
                #print("calculated pval based on permutation: ", this_perm_pval)
                got_pval = True
                break
            #print("perm_pval:", perm_pval)
            current_fraction = current_fraction + perm_fraction
        #sys.exit()
        if (not got_pval):
            empir_row_pvals[row[0]]  = 1
            #print("calculated pval based on permutation, set to 1: ")
        
        #if (ctr >  20):
        #    break
    
    #print(calc_permed_pvals)        
    #sys.exit()                
    return empir_row_pvals
if __name__ == "__main__":
    main()