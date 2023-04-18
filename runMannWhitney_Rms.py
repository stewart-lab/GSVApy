from collections import OrderedDict, defaultdict
import sys
import os
import cmdlogtime
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multitest as sm
import random

COMMAND_LINE_DEF_FILE = "./runMannWhitneyCommandLine.txt"


def main():
    (start_time_secs, pretty_start_time, my_args, logfile) = cmdlogtime.begin(COMMAND_LINE_DEF_FILE)
    in_file = my_args["in_file"]
    paired_data = my_args["paired_data"]
    MW = my_args['mann_whitney']
    use_pval_for_perm = my_args['use_pval_for_perm']
    permute = my_args['permute']
    shuffle_labels = my_args['shuffle_labels']
    if shuffle_labels:
        print("Not sure if shuffle labels is working properly. Bug Ron to test it!")
        if not permute:
            print("Should set permute flag if choosing shuffle_labels.")
            sys.exit()
    if use_pval_for_perm:
        if not permute:
            print("Should set permute flag if choosing use_pval_for_perm.")
            sys.exit()

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
    df = pd.read_csv(in_file, header=0, sep="\t")
    print(df.head(2))

    # just for fun, make a box plot
    # df.boxplot(column=[cond1_names[0], cond2_names[0]], grid=False)
    # plt.show()
    cond1_len = len(cond1_names)
    # cond2_len = len(cond2_names)

    all_conds = cond1_names + cond2_names
    print(all_conds)
    print(cond1_len)
    # print(cond2_len)
    shuf_conds = all_conds.copy()

    # check to see if the data is normally distributed in each row. Probably not enough data in each row to know,
    #  so probably this isn't that useful. I'm putting the code in and you can run it by typing -show_w on command line
    #  if most p-values are < 0.05, then the data is not normally distributed, and you should use a
    #  mann-whitney test, and NOT a t-test.
    # df2 = df[cond1_names + cond2_names]
    stat_dict = OrderedDict()
    mod_stat_or_p_val_dict = OrderedDict()
    pval_dict = OrderedDict()
    adj_pval_dict = OrderedDict()
    adj_empir_pval_dict = OrderedDict()
    pval_list = []

    if (my_args["show_wilks"]):
        with open(out_wilks, 'w') as out_wilks:
            normal_count = 0
            not_normal_count = 0
            out_wilks.write("Normality\tTerm\tw\tpvalue\n")
            for _, row in df.iterrows():
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
        stat_obj, mod_stat = calc_stats(cond1_names, cond2_names, paired_data, row, MW, use_pval_for_perm)

        stat_dict[row[0]] = stat_obj.statistic
        pval_dict[row[0]] = stat_obj.pvalue
        mod_stat_or_p_val_dict[row[0]] = mod_stat
        pval_list.append(stat_obj.pvalue)
    # Here is where I would do a bunch of permutations of the data, then somehow report an empirical p-value in addition to b+h
    #  THis is for an empirical p-value based on swapping the column labels (sample names)
    print(df)

    if permute:
        if shuffle_labels:
            empir_pvals = calculate_empirical_label_pvalues(df, shuf_conds, paired_data, cond1_len, mod_stat_or_p_val_dict, MW, use_pval_for_perm)
        else:
            empir_pvals = calculate_empirical_row_pvalues(df, paired_data, mod_stat_or_p_val_dict, cond1_names, cond2_names, MW, use_pval_for_perm)
        empir_pval_list = []
        for key, pval in empir_pvals.items():
            empir_pval_list.append(float(pval))
        adj_empir_pvals = []

    with open(out_file, 'w') as out_mw:
        if MW:
            pval_type = "MW P Value"
        else:
            pval_type = "t P Value"
        if permute:
            if use_pval_for_perm:
                shuffle_pvals_or_stats = " using pvals"
            else:
                shuffle_pvals_or_stats = " using mod stats"
            if shuffle_labels:
                out_mw.write(f"Term\tStat\t{pval_type}\tAdj P Value (B+H)\tEmpirical P Value (shuffle Labels {shuffle_pvals_or_stats})\tAdj Empir Label Pval)\n")
            else:
                out_mw.write(f"Term\tStat\t{pval_type}\tAdj P Value (B+H)\tEmpirical P Value (shuffle rows {shuffle_pvals_or_stats})\tAdj Empir Row Pval\n")
        else:
            out_mw.write(f"Term\tStat\t{pval_type}\tAdj P Value (B+H)\n")
        rej, adj_pvals, _, _ = sm.multipletests(pval_list, alpha=0.05, method='fdr_bh')
        for i, term in enumerate(stat_dict.keys()):
            adj_pval_dict[term] = '{:.4f}'.format(adj_pvals[i])
        if permute:
            rej, adj_empir_pvals, _, _ = sm.multipletests(empir_pval_list, alpha=0.05, method='fdr_bh')
            for i, term in enumerate(stat_dict.keys()):
                adj_empir_pval_dict[term] = '{:.4f}'.format(adj_empir_pvals[i])
        for term in stat_dict.keys():
            if pval_dict[term] < 1.1:  # Used to be < 0.05, now, let's let them all through.
                if permute:
                    if shuffle_labels:
                        out_mw.write(f"{term}\t{stat_dict[term]}\t{pval_dict[term]}\t{adj_pval_dict[term]}\t{empir_pvals[term]}\t{adj_empir_pval_dict[term]}\n")
                    else:
                        out_mw.write(f"{term}\t{stat_dict[term]}\t{pval_dict[term]}\t{adj_pval_dict[term]}\t{empir_pvals[term]}\t{adj_empir_pval_dict[term]}\n")
                else:
                    out_mw.write(f"{term}\t{stat_dict[term]}\t{pval_dict[term]}\t{adj_pval_dict[term]}\n")
    cmdlogtime.end(logfile, start_time_secs)


# --------------------------------------- FUNCTIONS ---------------------------------
def calc_stats(cond1_names, cond2_names, paired_data, row, MW, use_pval_for_perm):
    middle_mw_score = (len(cond1_names) * len(cond2_names)) / 2  # maximum MW score is #ofSamplesOfCondition1 * #ofSamplesOfCondition2.
    if MW:
        if paired_data:
            stat_obj = stats.wilcoxon(x=row[cond1_names].astype(float), y=row[cond2_names].astype(float), alternative="two-sided")
        else:
            stat_obj = stats.mannwhitneyu(x=row[cond1_names].astype(float), y=row[cond2_names].astype(float), alternative="two-sided")
    else:  # t-test.
        if paired_data:
            stat_obj = stats.ttest_rel(a=row[cond1_names].astype(float), b=row[cond2_names].astype(float))
        else:
            stat_obj = stats.ttest_ind(a=row[cond1_names].astype(float), b=row[cond2_names].astype(float), permutations=0)

    if use_pval_for_perm:
        mod_stat = stat_obj.pvalue
    else:
        if MW:
            # rms.  this modification needs to be based on the middle_mw_score
            mod_stat = - abs(middle_mw_score - stat_obj.statistic)
        else:  # t-test
            mod_stat = stat_obj.statistic
            if mod_stat > 0:
                mod_stat = - mod_stat
    return stat_obj, mod_stat


def calculate_empirical_label_pvalues(df, shuf_conds, paired_data, cond1_len, mod_stat_or_p_val_dict, MW, use_pval_for_perm):
    empir_label_pvals = OrderedDict()
    NUM_PERMS = 100  # RMS, make this an input parameter for label permutation
    perm_fraction = 1 / NUM_PERMS
    print(perm_fraction)
    ctr = 0
    for _, row in df.iterrows():
        ctr = ctr + 1
        perm_stats_or_pvals = []
        for i in range(0, NUM_PERMS):
            random.shuffle(shuf_conds)
            shuf_cond1_names = shuf_conds[:cond1_len]
            shuf_cond2_names = shuf_conds[cond1_len:]
            stat_obj, mod_stat = calc_stats(shuf_cond1_names, shuf_cond2_names, paired_data, row, MW, use_pval_for_perm)
            perm_stats_or_pvals.append(mod_stat)

        perm_stats_or_pvals.sort()
        current_fraction = perm_fraction
        got_pval = False
        for perm_stat_or_pval in perm_stats_or_pvals:
            if mod_stat_or_p_val_dict[row[0]] <= perm_stat_or_pval:  # RMS. SHould this be less than, or less than or equal to?  Less than is more conservative
                this_perm_pval = '{:.4f}'.format(current_fraction)
                empir_label_pvals[row[0]] = this_perm_pval
                # print("calculated pval based on permutation: ", this_perm_pval)
                got_pval = True
                break
            current_fraction = current_fraction + perm_fraction
        if (not got_pval):
            empir_label_pvals[row[0]] = 1
    return empir_label_pvals


def calculate_empirical_row_pvalues(df, paired_data, mod_stat_or_pval_dict, cond1_names, cond2_names, MW, use_pval_for_perm):
    empir_row_pvals = OrderedDict()
    perm_stats_or_pvals_by_term = defaultdict(list)
    ctr = 0
    perm_to_process = 0
    print("in calc empir row")
    # calculate all the pvalues based on the permuted row data and stuff into perm_pvals_by_term
    for _, row in df.iterrows():
        if (row.loc['PERMUTED'] == 0):
            continue
        perm_to_process = int(row.loc['PERMUTED'])
        ctr = ctr + 1
        # perm_pvals = []
        stat_obj, mod_stat = calc_stats(cond1_names, cond2_names, paired_data, row, MW, use_pval_for_perm)

        perm_stats_or_pvals_by_term[row[0]].append(mod_stat)

    NUM_PERMS = perm_to_process
    perm_fraction = 1 / NUM_PERMS
    print(perm_fraction)
    # now take all these permuted pvals or stats and calculate an empirical pval
    for _, row in df.iterrows():
        # perm_pvals_by_term[row[0]].sort()
        perm_stats_or_pvals_by_term[row[0]].sort()
        current_fraction = perm_fraction
        # print("calculating permpval for ", pval_dict[row[0]])
        got_pval = False
        for perm_stat_or_pval in perm_stats_or_pvals_by_term[row[0]]:
            if mod_stat_or_pval_dict[row[0]] <= perm_stat_or_pval:  # RMS,  SHould this be less than, or less than or equal to?  Less than is more conservative
                this_perm_pval = '{:.4f}'.format(current_fraction)
                empir_row_pvals[row[0]] = this_perm_pval
                # print("calculated pval based on permutation: ", this_perm_pval)
                got_pval = True
                break
            # print("perm_pval:", perm_pval)
            current_fraction = current_fraction + perm_fraction
        # sys.exit()
        if (not got_pval):
            empir_row_pvals[row[0]] = 1
            # print("calculated pval based on permutation, set to 1: ")
    return empir_row_pvals


if __name__ == "__main__":
    main()
