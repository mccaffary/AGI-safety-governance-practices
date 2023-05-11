from multiprocessing.sharedctypes import Value
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sys
import os
import csv
import pandas as pd
from typing import List
import scipy.cluster.hierarchy as sch
from collections import Counter, OrderedDict, defaultdict
from matplotlib.offsetbox import AnchoredText
import dabest
from sklearn.utils import resample
import scipy.stats
import math
from tabulate import tabulate
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
from scipy.stats import weibull_min
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
import scipy.stats as stats
import matplotlib.font_manager as fm
prop = fm.FontProperties(fname='/Users/mccaffary/Desktop/proxima_nova/Proxima_Nova_Reg.otf')
prop_bold = fm.FontProperties(fname='/Users/mccaffary/Desktop/proxima_nova/Proxima_Nova_Semibold.otf')

###############
### GENERAL ###
###############


def void():
    pass




def default_plotting_params() -> None:

    """
    
    Run this function at the start of a notebook to set default
    aesthetics and plotting parameters
    
    """

    sns.set()
    sns.set_style('white')
    params = {'legend.fontsize': 'xx-large',
            'text.usetex':False,
            'axes.labelsize': 'xx-large',
            'axes.titlesize':'xx-large',
            'xtick.labelsize':'x-large',
            'ytick.labelsize':'x-large'}

    plt.rcParams.update(params)





def load_csv(numeric_file: str, label_file: str) -> pd.core.frame.DataFrame:

    """
    
    Loads numeric and label .csv files into Pandas DataFrame
    NB: files must be in 'data/' sub-directory
    
    """

    numeric_data = pd.read_csv("data/" + numeric_file)
    label_data = pd.read_csv("data/" + label_file)
    return numeric_data, label_data




def series2list(series: pd.Series) -> list:
    """
    
    Convert a pandas series to list
    
    """
    return series.tolist()




def discretise_cmap(cmap: str, n: int) -> list:
    colormap = plt.cm.get_cmap(cmap)
    cmap = [colormap(i / (n - 1)) for i in range(n)]
    return cmap



def generate_gov_mechanisms_str() -> list[str]:
    govmechanisms = [f"govmechanisms_{i}" for i in range(1, 51)]
    return govmechanisms



def generate_gov_mechanisms_labels() -> list[str]:
    govmechanisms_labels = ["Alignment techniques", "Pre-registration of large training runs", \
                           "Gradual scaling", "Dangerous capabilities evaluations", \
                           "Pausing training of dangerous models", "Staged deployment", \
                           "API access to powerful models", "No open-sourcing", "Safety restrictions", \
                           "Notify a state actor before deployment", "Monitor systems and their uses", \
                           "Report safety incidents", "Treat updates similarly to new models", \
                           "Emergency response plan", "Pre-deployment risk assessment", \
                           "Enterprise risk management", "Internal audit", "Red teaming", \
                           "Third-party model audits", "Researcher model access", "Bug bounty programs", \
                           "Security standards", "Military-grade information security", \
                           "Increasing levels of external scrutiny", "Avoiding hype", \
                           "Publish views about AGI risk", "Internal review before publication", \
                           "Pre-training risk assessment", "Model containment", "Safety vs. capabilities", \
                           "Notify affected parties", "Notify other labs", "Avoid capabilities jumps", \
                           "KYC screening", "Treat internal deployments similar to external deployments", \
                           "Post-deployment evaluations", "Board risk committee", "Chief risk officer", \
                           "Third-party governance audits", "Dual control", "Inter-lab scrutiny", \
                           "Tracking model weights", "Security incident response plan", \
                           "Industry sharing of security information", "Protection against espionage", \
                           "Publish alignment strategy", "Statement about governance structure", \
                           "Publish results of internal risk assessments", \
                           "Publish results of external scrutiny", "Background checks"]
    return govmechanisms_labels




def filter_responses(numeric_data, gov_mechanism) -> list[int]:
    resps = series2list(numeric_data[gov_mechanism][2:])
    resps_str = [str(elem) for elem in resps]
    resps_nan_filter = [elem for elem in resps_str if (elem != "nan")]
    # generate lists with "I don't know" responses recoded or removed (for mean calculation)
    resps_dont_know_replace = ["3" if (elem == "-88") else elem for elem in resps_nan_filter]
    resps_dont_know_remove = [elem for elem in resps_nan_filter if (elem != "-88")]
    # convert elements of lists to integers
    resps_dont_know_replace_int = [int(elem) for elem in resps_dont_know_replace]
    resps_dont_know_remove_int = [int(elem) for elem in resps_dont_know_remove]
    return resps_dont_know_replace_int, resps_dont_know_remove_int
    


def percentage_breakdown(numbers: list) -> dict:
    count_dict = Counter(numbers)
    total = len(numbers)
    percentage_dict = {key: (value / total) * 100 for key, value in count_dict.items()}
    all_keys = set(range(-2, 4))
    result = {key: (percentage_dict[key] if key in percentage_dict else 0) for key in all_keys}
    sorted_result = OrderedDict(sorted(result.items()))
    return sorted_result





##############################
### STACKED BAR PLOT f(x)s ###
##############################


def filter_responses_stacked_bar_plot(numeric_data, gov_mechanism) -> list[int]:
    resps = series2list(numeric_data[gov_mechanism][2:])
    resps_str = [str(elem) for elem in resps]
    resps_nan_filter = [elem for elem in resps_str if (elem != "nan")]
    # generate lists with "I don't know" responses recoded or removed (for mean calculation)
    resps_dont_know_replace = ["-3" if (elem == "-88") else elem for elem in resps_nan_filter]
    resps_dont_know_remove = [elem for elem in resps_nan_filter if (elem != "-88")]
    # convert elements of lists to integers
    resps_dont_know_replace_int = [int(elem) for elem in resps_dont_know_replace]
    resps_dont_know_remove_int = [int(elem) for elem in resps_dont_know_remove]
    return resps_dont_know_replace_int, resps_dont_know_remove_int
    
    

    
def percentage_breakdown_stacked_bar_plot(numbers: list) -> dict:
    count_dict = Counter(numbers)
    total = len(numbers)
    percentage_dict = {key: (value / total) * 100 for key, value in count_dict.items()}
    all_keys = set(range(-3, 3))
    result = {key: (percentage_dict[key] if key in percentage_dict else 0) for key in all_keys}
    sorted_result = OrderedDict(sorted(result.items()))
    return sorted_result



def sort_by_first_element_stacked_bar_plot(lists, labels):
    sorted_indices = sorted(range(len(lists)), key=lambda i: lists[i][0], reverse=False)
    sorted_lists = [lists[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    return sorted_lists, sorted_labels



def generate_reordered_response_counts(gov_mechanism_titles_, gov_mechanisms_filtered_responses_stacked_bar_plot) -> list[int]:
    response_count_per_mechanism = []
    for sublist in gov_mechanisms_filtered_responses_stacked_bar_plot:
        response_count_per_mechanism.append(len(sublist))

    gov_mechanism_titles = generate_gov_mechanisms_labels()
    response_count_per_mechanism_idx = list(zip(gov_mechanism_titles, response_count_per_mechanism))

    # re-order based on the order determined by the stacked bar plot
    original_tuples = response_count_per_mechanism_idx
    original_dict = dict(original_tuples)

    reordered_response_count = [(title, original_dict[title]) for title in gov_mechanism_titles_]
    reordered_response_count_vals = [elem[1] for elem in reordered_response_count]
    return reordered_response_count_vals




def visualise_responses_stacked_barplpot(data_stacked_bar_plot, gov_mechanism_titles_, reordered_response_count_vals) -> None:
    data = np.array(data_stacked_bar_plot)
    data_cum = data.cumsum(axis=1)

    colors = ["#046e99", "#4fa2c4", "#ffffff", "#c21d4c", "#9b173d", "#BFBFBF"]
    text_colors = ["white", "white", "grey", "white", "white", "white"]

    fig, ax = plt.subplots(figsize=(25, 40))

    legend_labels = ["I don't know", "Strongly disagree", "Somewhat disagree", "Neither agree nor disagree", \
                    "Somewhat agree", "Strongly agree"][::-1]

    for i in range(data.shape[1]):
        if i == 0:
            ax.barh(y=range(data.shape[0]), width=data[:, i], \
                    color=colors[i], label=legend_labels[i], edgecolor="k", linewidth=2)
        else:
            ax.barh(y=range(data.shape[0]), width=data[:, i], left=data_cum[:, i-1], \
                    color=colors[i], label=legend_labels[i], edgecolor="k", linewidth=2)

        for j, value in enumerate(data[:, i]):
            if value > 0:
                rounded_value = round(value)
                ax.text(x=data_cum[j, i-1] + value/2 if i > 0 else value/2, y=j, s=f'{rounded_value}%', 
                        va='center', ha='center', fontsize=12, color=text_colors[i], fontweight='bold')

    for j, n in enumerate(reordered_response_count_vals):
        ax.text(x=data_cum[j, -1] + 2, y=j, s=f'n = {n}', va='center', ha='left', fontsize=16, color='k')

    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels([gov_mechanism_titles_[i] for i in range(data.shape[0])])
    ax.set_xlabel("\nPercentage of respondents")

    xticks = ax.get_xticks()
    ax.set_xticklabels([f'{int(tick)}%' for tick in xticks])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), frameon=False, ncol=6, fontsize=18)
    sns.despine(left=True, right=True, top=True, bottom=True)
    plt.show()



def visualise_responses_stacked_barplpot_final(data_stacked_bar_plot, gov_mechanism_titles_, \
                                              reordered_response_count_vals) -> None:

    prop = fm.FontProperties(fname='/Users/mccaffary/Desktop/proxima_nova/Proxima_Nova_Reg.otf')
    prop_bold = fm.FontProperties(fname='/Users/mccaffary/Desktop/proxima_nova/Proxima_Nova_Semibold.otf')

    data = np.array(data_stacked_bar_plot)
    data_cum = data.cumsum(axis=1)

    colors = ["#00709d", "#20a5c8", "white", "#d3014a", "#ad003c", "#BFBFBF"]
    text_colors = ["white", "white", "grey", "white", "white", "white"]
    #text_colors = ["k", "k", "k", "k", "k", "k"]

    fig, ax = plt.subplots(figsize=(28, 50), dpi=100)

    legend_labels = ["I don't know", "Strongly disagree", "Somewhat disagree", "Neither agree nor disagree", \
                    "Somewhat agree", "Strongly agree"][::-1]

    for i in range(data.shape[1]):
        if i == 0:
            ax.barh(y=range(data.shape[0]), width=data[:, i], \
                    color=colors[i], label=legend_labels[i], edgecolor="k", linewidth=2)
        else:
            ax.barh(y=range(data.shape[0]), width=data[:, i], left=data_cum[:, i-1], \
                    color=colors[i], label=legend_labels[i], edgecolor="k", linewidth=2)

        for j, value in enumerate(data[:, i]):
            if value > 0:
                rounded_value = round(value)
                ax.text(x=data_cum[j, i-1] + value/2 if i > 0 else value/2, y=j, s=f'{rounded_value}%', 
                        va='center', ha='center', fontsize=22, color=text_colors[i], \
                        fontproperties=prop_bold)

    for j, n in enumerate(reordered_response_count_vals):
        ax.text(x=data_cum[j, -1] + 1, y=j, s=f'n = {n}', va='center', ha='left', \
                fontsize=22, color='k', fontproperties=prop_bold)

    ax.set_yticks(range(data.shape[0]))
    
    # make some of the long gov mechanism titles shorter...
    # dictionary with the original strings as keys and the replacements as values
    replacement_dict = {
        'Treat internal deployments similar to external deployments': 'Internal deployments = external deployments',
        'Publish results of internal risk assessments' : 'Publish internal risk assessment results',
        'No open-sourcing' : 'No unsafe open-sourcing'
    }

    # Use list comprehension to replace strings
    gov_mechanism_titles_2 = [replacement_dict.get(title, title) for title in gov_mechanism_titles_]
    ax.set_yticklabels([gov_mechanism_titles_2[i] for i in range(data.shape[0])], \
                      fontproperties=prop, fontsize=28)
    
    ax.set_xlabel("\nPercentage of respondents", labelpad=-50, \
                  fontproperties=prop_bold, fontsize=34)

    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xticklabels([f'{tick}%' for tick in np.arange(0, 101, 10)], fontproperties=prop, fontsize=32)

    # Adjust the position of x-tick labels
    for label in ax.get_xticklabels():
        label.set_y(+0.03)  # adjust this value as needed


    ax.legend(loc='upper center', bbox_to_anchor=(0.375, 0.98), frameon=False, \
              ncol=6, fontsize=20, \
              prop={'fname':'/Users/mccaffary/Desktop/proxima_nova/Proxima_Nova_Semibold.otf', 'size':'28'})
    sns.despine(left=True, right=True, top=True, bottom=True)
    #plt.savefig("figure_1.png", dpi=1000, bbox_inches = 'tight')
    plt.show()





#########################
### DEMOGRAPHIC f(x)s ###
#########################


def percentage_breakdown_gender(numbers: list) -> dict:
    count_dict = Counter(numbers)
    total = len(numbers)
    percentage_dict = {key: (value / total) * 100 for key, value in count_dict.items()}
    all_keys = set(range(1,4))
    result = {key: (percentage_dict[key] if key in percentage_dict else 0) for key in all_keys}
    sorted_result = OrderedDict(sorted(result.items()))
    return sorted_result



def percentage_breakdown_sector(numbers: list) -> dict:
    count_dict = Counter(numbers)
    total = len(numbers)
    percentage_dict = {key: (value / total) * 100 for key, value in count_dict.items()}
    all_keys = set(range(1,6))
    result = {key: (percentage_dict[key] if key in percentage_dict else 0) for key in all_keys}
    sorted_result = OrderedDict(sorted(result.items()))
    return sorted_result




def generate_gender_responses(numeric_data):
    gender_responses_ = series2list(numeric_data["gender"])[2:]
    gender_responses_str = [str(elem) for elem in gender_responses_]
    gender_responses_nan_filter = [elem for elem in gender_responses_str if (elem != "nan")]
    gender_responses_replace = ["3" if (elem == "-99") else elem for elem in gender_responses_nan_filter]
    gender_responses = [int(elem) for elem in gender_responses_replace]
    return gender_responses
    
    

def separate_elements(input_list):
    output_list = []
    for item in input_list:
        if ',' in item:
            output_list.extend(item.split(','))
        else:
            output_list.append(item)
    return output_list




def generate_sector_responses(numeric_data):
    sector_responses_ = series2list(numeric_data["sector"])[2:]
    sector_responses_str = [str(elem) for elem in sector_responses_]
    sector_responses_nan_filter = [elem for elem in sector_responses_str if (elem != "nan")]
    sector_responses_separated = separate_elements(sector_responses_nan_filter)
    sector_responses_replace = ["9" if (elem == "-99") else elem for elem in sector_responses_separated]
    sector_responses = [int(elem) for elem in sector_responses_replace]
    return sector_responses



def recode_sector(sector_responses: list) -> list:
    
    agi_labs = []
    academia = []
    civil_society = []
    other = []
    prefer_not = []
    
    for response in sector_responses:
        if response == 1:
            agi_labs.append(1)
        elif response == 7:
            academia.append(2)
        elif (response == 4) or (response == 5):
            civil_society.append(3)
        elif (response == 8) or (response == 2) or (response == 6) or (response == 3):
            other.append(4)
        elif response == 9:
            prefer_not.append(5)
        else:
            raise ValueError
            
    all_responses = agi_labs + academia + civil_society + other + prefer_not
    return all_responses




def visualise_gender_sector_distributions(gender_responses, sector_responses_recoded) -> None:
    plt.rcParams['figure.figsize'] = (20, 10)
    # Create a single figure
    fig, (ax1, ax2) = plt.subplots(1, 2)
    data = percentage_breakdown_gender(gender_responses).values()
    labels = ["Man", "Woman", "Prefer not to say"]
    # Define your custom colors
    colors = ['#045ea7', '#e19a04', '#bfbfbf']
    # First subplot
    ax1.pie(data, labels=labels, autopct=lambda p: f'{p:.1f}%', startangle=90, colors=colors,
            textprops={'fontsize': 22})
    # Draw a white circle at the center to create a donut chart
    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax1.add_artist(center_circle)
    ax1.set_title('Gender distribution\n', fontsize=40)
    # Equal aspect ratio ensures that the pie chart is circular
    ax1.axis('equal')

    # Sample data
    data = percentage_breakdown_sector(sector_responses_recoded).values()
    labels = ["AGI lab", "Academia", "Civil society", "Other", "Prefer not to say"]
    # Second subplot
    ax2.pie(data, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 20})
    # Draw a white circle at the center to create a donut chart
    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax2.add_artist(center_circle)
    ax2.set_title('Sector distribution\n', fontsize=40)
    # Equal aspect ratio ensures that the pie chart is circular
    ax2.axis('equal')
    # Adjust layout
    plt.tight_layout()
    plt.show()




def visualise_demographics_figure_2(gender_responses, sector_responses_recoded):
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.figure(figsize=(20,10), dpi=100)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # First subplot
    data = percentage_breakdown_sector(sector_responses_recoded).values()
    labels = ["AGI lab", "Academia", "Civil society", "Other", "Prefer not\nto say"]
    labels = [f'{label}\n{value:.1f}%' for label, value in zip(labels, data)]
    colors_sector = ["#d4014a", "#029cce", "#f89d05", "#046dc2", "#bfbfbf"]
    wedges1, text1 = ax1.pie(data, labels=labels, startangle=90, textprops={'fontsize': 20}, \
                             labeldistance=1.3, colors=colors_sector)

    for text in text1:
        pos = text.get_position()
        #ax1.plot([0, pos[0]], [0, pos[1]], color='black', zorder=0)
        ax1.plot([0, pos[0]*0.935], [0, pos[1]*0.935], color='black', zorder=0)

    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax1.add_artist(center_circle)
    ax1.set_title('\nSector distribution\n\n', fontsize=28, fontproperties=prop_bold)
    ax1.axis('equal')

    # Second subplot
    data = percentage_breakdown_gender(gender_responses).values()
    labels = ["Man", "Woman", "Prefer not\nto say"]
    colors = ['#00317d', '#e48f00', '#bfbfbf']
    labels = [f'{label}\n{value:.1f}%' for label, value in zip(labels, data)]
    wedges2, text2 = ax2.pie(data, labels=labels, startangle=90, colors=colors,
                            textprops={'fontsize': 22, 'fontproperties': prop}, labeldistance=1.3)

    for text in text2:
        pos = text.get_position()
        #ax2.plot([0, pos[0]], [0, pos[1]], color='black', zorder=0)
        ax2.plot([0, pos[0]*0.935], [0, pos[1]*0.935], color='black', zorder=0)

    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax2.add_artist(center_circle)
    ax2.set_title('\nGender distribution\n\n', fontsize=28, fontproperties=prop_bold)
    ax2.axis('equal')

    plt.tight_layout()
    #plt.savefig("figure_2.pdf", dpi=1500, bbox_inches = 'tight')

    plt.show();




###############################
### FIGURES 3, 4, & 5 f(x)s ###
###############################



def compute_gov_mechanism_mean(data: list) -> list:
    means = []
    for sublist in data:
        filtered_sublist = [value for value in sublist if value != 3]
        mean = sum(filtered_sublist) / len(filtered_sublist)
        means.append(mean)
    return means



def compute_gov_mechanism_standard_error(data: list) -> list:
    standard_errors = []
    for sublist in data:
        filtered_sublist = [value for value in sublist if value != 3]
        num_samples = len(filtered_sublist)
        if num_samples > 1:
            mean = sum(filtered_sublist) / num_samples
            variance = sum((x - mean) ** 2 for x in filtered_sublist) / (num_samples - 1)
            standard_deviation = np.sqrt(variance)
            standard_error = standard_deviation / np.sqrt(num_samples)
        else:
            standard_error = 0
        standard_errors.append(standard_error)
    return standard_errors



def visualise_lowest_highest_mean_agreement_mechanisms(gov_mechanisms_means_full_vals_, \
                                                      errors_lowest_mean_agreement, \
                                                      errors_highest_mean_agreement) -> None:
    
    default_plotting_params()
    plt.rcParams['figure.figsize'] = (16,10)

    cmap_lowest_old = discretise_cmap(cmap="Greens_r", n=5)
    cmap_highest_old = discretise_cmap(cmap="Blues", n=5)

    cmap_highest = ["#165a72", "#1a6985", "#1e7898", "#2187ab", "#2596be"][::-1]
    cmap_lowest = ["#d3eaf2", "#bee0ec", "#a8d5e5", "#92cbdf", "#7cc0d8"]

    ### first plot ###
    plt.subplot(121)
    plt.bar(x=np.arange(len(gov_mechanisms_means_full_vals_[::-1][:5])), height=gov_mechanisms_means_full_vals_[::-1][:5],
            color=cmap_lowest, edgecolor="k", linewidth=2)

    # Add error bars to the first plot
    for idx, (mean, error) in enumerate(zip(gov_mechanisms_means_full_vals_[::-1][:5], errors_lowest_mean_agreement)):
        plt.vlines(x=idx, ymin=mean - error, ymax=mean + error, colors='k', linewidth=1.5)

    for idx in range(len(gov_mechanisms_means_full_vals_[::-1][:5])):
        plt.axvline(idx, color="grey", linestyle="--", alpha=0.3, ymin=0.11, ymax=0.89)

    plt.ylim(-2.15, 2.15)
    y_labels = ["Strongly\ndisagree\n(-2)", "Somewhat\ndisagree\n(-1)", "Neither agree\nnor disagree\n(0)", \
                "Somewhat\nagree\n(1)", "Strongly\nagree\n(2)"]
    plt.yticks(np.arange(-2, 3), y_labels)

    # Use yaxis.labelpad to adjust the distance between the y-tick labels and the y-axis
    ax = plt.gca()
    ax.yaxis.labelpad = 5
    ax.tick_params(axis='y', which='both', rotation=0)

    plt.axhline(0, color="grey", linestyle="--")
    sns.despine(ax=plt.gca(), left=False, bottom=True) 

    # Annotate top of bars in the first subplot
    for idx, value in enumerate(gov_mechanisms_means_full_vals_[::-1][:5]):
        plt.text(idx, value + 0.35, f'{value:.1f}', ha='center', va='bottom', fontsize=15)

    x_labels_lowest = ['Notify other labs', 'Avoid capabilities\njumps', 'Inter-lab scrutiny', \
                        'Notify affected\nparties', 'Notify state actor\nbefore deployment']

    plt.xticks(np.arange(len(x_labels_lowest)), x_labels_lowest, rotation=90)
    plt.title("Lowest mean agreement\n\n")

    ### second plot ###
    plt.subplot(122)
    plt.bar(x=np.arange(len(gov_mechanisms_means_full_vals_[:5])), height=gov_mechanisms_means_full_vals_[:5][::-1],
            color=cmap_highest, edgecolor="k", linewidth=2)

    # Add error bars to the second plot
    for idx, (mean, error) in enumerate(zip(gov_mechanisms_means_full_vals_[:5][::-1], errors_highest_mean_agreement)):
        plt.vlines(x=idx, ymin=mean - error, ymax=mean + error, colors='k', linewidth=1.5)

    for idx in range(len(gov_mechanisms_means_full_vals_[:5])):
        plt.axvline(idx, color="grey", linestyle="--", alpha=0.3, ymin=0.11, ymax=0.89)

    plt.ylim(-2.15, 2.15)
    plt.yticks([])
    plt.axhline(0, color="grey", linestyle="--")
    sns.despine(ax=plt.gca(), left=True, bottom=True)  # Remove the left spine in the second subplot
    # Annotate top of bars in the second subplot
    for idx, value in enumerate(gov_mechanisms_means_full_vals_[:5][::-1]):
        plt.text(idx, value + 0.175, f'{value:.1f}', ha='center', va='bottom', fontsize=15)

    x_labels_highest = ['Pre-deployment\nrisk assessment', 'Dangerous\ncapabilities\nevaluations', \
                        'Third-party\nmodel audits', 'Safety restrictions', 'Red teaming'][::-1]

    plt.xticks(np.arange(len(x_labels_highest)), x_labels_highest, rotation=90)
    plt.title("Highest mean agreement\n\n")


    plt.tight_layout()
    plt.show()



def visualise_lowest_highest_mean_agreement_mechanisms_final(gov_mechanisms_means_full_vals_, \
                                                          errors_lowest_mean_agreement, \
                                                          errors_highest_mean_agreement) -> None:
    
    default_plotting_params()
    #plt.rcParams['figure.figsize'] = (16,10)
    plt.figure(figsize=(16,10), dpi=100)

    cmap_lowest_old = discretise_cmap(cmap="Greens_r", n=5)
    cmap_highest_old = discretise_cmap(cmap="Blues", n=5)

    cmap_highest = ["#005d74", "#016b88", "#00799b", "#0089af", "#009ac2"]
    cmap_lowest = ["#65c3dc", "#80cde2", "#9cd7e8", "#b5e1ed", "#cdebf3"]

    ### second plot first now ###
    plt.subplot(121)
    ax = plt.gca()
    ax.xaxis.tick_top()
    bar2 = plt.bar(x=np.arange(len(gov_mechanisms_means_full_vals_[:5])), 
                   height=gov_mechanisms_means_full_vals_[:5],
                   color=cmap_highest, edgecolor="k", linewidth=2)

    for idx, rect in enumerate(bar2):
        height = rect.get_height()
        plt.vlines(x=idx, ymin=height+0.1, ymax=height, linestyles='dashed', color='grey')

    ax.yaxis.set_tick_params(size=5, width=2)

    for idx, (mean, error) in enumerate(zip(gov_mechanisms_means_full_vals_[:5], errors_highest_mean_agreement)):
        plt.vlines(x=idx, ymin=mean - error, ymax=mean + error, colors='k', linewidth=1.5)

    plt.ylim(-2.15, 2.15)
    y_labels = ["Strongly\ndisagree\n(-2)", "Somewhat\ndisagree\n(-1)", "Neither agree\nnor disagree\n(0)", \
                "Somewhat\nagree\n(1)", "Strongly\nagree\n(2)"]
    plt.yticks(np.arange(-2, 3), y_labels, fontproperties=prop, fontsize=18)

    ax = plt.gca()
    ax.yaxis.labelpad = 5
    ax.tick_params(axis='y', which='both', rotation=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_tick_params(length=0)  # Set tick length to 0


    plt.axhline(0, xmin=-20, xmax=20, color="black", linestyle="--")
    sns.despine(ax=plt.gca(), left=False, bottom=True) 

    for idx, value in enumerate(gov_mechanisms_means_full_vals_[:5]):
        plt.text(idx, value + 0.2, f'{value:.1f}', ha='center', va='bottom', \
                 fontsize=15, fontproperties=prop)

    x_labels_highest = ['Pre-deployment\nrisk assessment', 'Dangerous\ncapabilities\nevaluations', \
                        'Third-party\nmodel audits', 'Safety restrictions', 'Red teaming']
    
    plt.xticks(np.arange(len(x_labels_highest)), x_labels_highest, rotation=45, \
           fontproperties=prop, fontsize=18, ha="left", y=1.05)  # added 'y' parameter

    #plt.title("Highest mean agreement\n\n", fontproperties=prop_bold, fontsize=24)
    ax.annotate("Highest mean agreement\n\n", 
             xy=(0.5, 1.28), xytext=(0, 10),
             xycoords='axes fraction', textcoords='offset points',
             size=24, ha='center', va='bottom',
             fontproperties=prop_bold)

    ### first plot now ###
    plt.subplot(122)
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_tick_params(length=0)  # Set tick length to 0

    bar1 = plt.bar(x=np.arange(len(gov_mechanisms_means_full_vals_[::-1][:5])), 
                   height=gov_mechanisms_means_full_vals_[::-1][:5][::-1],
                   color=cmap_lowest, edgecolor="k", linewidth=2)

    for idx, rect in enumerate(bar1):
        height = rect.get_height()
        plt.vlines(x=idx, ymin=height+0.6, ymax=2.15, linestyles='dashed', color='grey')

    ax.yaxis.set_tick_params(size=5, width=2)

    for idx, (mean, error) in enumerate(zip(gov_mechanisms_means_full_vals_[::-1][:5][::-1], errors_lowest_mean_agreement[::-1])):
        plt.vlines(x=idx, ymin=mean - error, ymax=mean + error, colors='k', linewidth=1.5)

    plt.ylim(-2.15, 2.15)
    plt.yticks([])

    plt.axhline(0, xmin=-20, xmax=20, color="black", linestyle="--")
    sns.despine(ax=plt.gca(), left=True, bottom=True) 

    for idx, value in enumerate(gov_mechanisms_means_full_vals_[::-1][:5][::-1]):
        plt.text(idx, value + 0.35, f'{value:.1f}', ha='center', va='bottom', \
                 fontsize=15, fontproperties=prop)

    x_labels_lowest = ['Notify other labs', 'Avoid capabilities\njumps', 'Inter-lab scrutiny', \
                        'Notify affected\nparties', 'Notify state actor\nbefore deployment'][::-1]
    
    plt.xticks(np.arange(len(x_labels_lowest)), x_labels_lowest, rotation=45, \
           fontproperties=prop, fontsize=18, ha="left", y=1.05)  # added 'y' parameter

    #plt.title("Lowest mean agreement\n\n", fontproperties=prop_bold, fontsize=24)
    ax.annotate("Lowest mean agreement\n\n", 
             xy=(0.5, 1.28), xytext=(0, 10),
             xycoords='axes fraction', textcoords='offset points',
             size=24, ha='center', va='bottom',
             fontproperties=prop_bold)

    plt.tight_layout()
    #plt.savefig("figure_3.pdf", dpi=1000, bbox_inches = 'tight')
    plt.show()
    



def generate_gov_mechanisms_percentages(gov_mechanisms_filtered_responses_full):
    gov_mechanisms_percentages = []
    for sublist in gov_mechanisms_filtered_responses_full:
        sublist_length = len(sublist)
        percentages = [(sublist.count(value) / sublist_length) * 100 for value in range(-2, 4)]
        gov_mechanisms_percentages.append(percentages)
    return gov_mechanisms_percentages




def visualise_mean_agreement_all_mechanisms_final(gov_mechanisms_means_full_vals_, \
                                           gov_mechanisms_se_full_vals_, \
                                           gov_mechanisms_means_full_idx_, \
                                           gov_mechanism_titles_ordered_full_edit) -> None:

    #plt.rcParams['figure.figsize'] = (25, 45)
    plt.figure(figsize=(25,45), dpi=100)

    cmap_ = discretise_cmap(cmap="Blues", n=len(gov_mechanisms_means_full_vals_))

    y_positions = np.arange(len(gov_mechanisms_means_full_vals_[::-1]))
    plt.scatter(x=gov_mechanisms_means_full_vals_[::-1], y=y_positions, \
                c=cmap_, edgecolor="k", linewidth=2, s=200, zorder=10)

    # Add horizontal lines for +/- standard error
    for idx, (mean, se) in enumerate(zip(gov_mechanisms_means_full_vals_[::-1], gov_mechanisms_se_full_vals_[:])):
        plt.hlines(y=idx, xmin=mean - se, xmax=mean + se, \
                   colors="grey", linewidth=2)
        
        # Add annotations next to each scatter point
        #plt.text(mean + se + 0.05, y_positions[idx], f'{mean:.1f}', ha='left', va='center', fontsize=18) 
        plt.text(mean - se - 0.1, y_positions[idx], f'{mean:.1f}', ha='left', va='center', \
                 fontsize=22, fontproperties=prop) 

    # Add faint grey boxes around the scattered points and y-tick labels
    ax = plt.gca()
    #for idx in y_positions:
        #rect = patches.Rectangle((-2.1, idx - 0.4), 4.2, 0.8, linewidth=1, edgecolor='gray', facecolor='none', alpha=0.3)
        #ax.add_patch(rect)

    plt.xlim(-2.02, 2.02)
    plt.axvline(0, c="k", linestyle="--")
    plt.axvline(-2, c="grey", linestyle="--", alpha=0.4)
    plt.axvline(-1, c="grey", linestyle="--", alpha=0.4)
    plt.axvline(1, c="grey", linestyle="--", alpha=0.4)
    plt.axvline(2, c="grey", linestyle="--", alpha=0.4)
    for i in range(51):
        plt.axhline(i-0.5, c="grey", alpha=0.7)
    
    gov_mechanism_titles = generate_gov_mechanisms_labels()
    gov_mechanism_titles_ordered_full = [gov_mechanism_titles[idx] for idx in gov_mechanisms_means_full_idx_][::-1]


    x_labels = ["Strongly\ndisagree\n(-2)", "Somewhat\ndisagree\n(-1)", "Neither agree\nnor disagree\n(0)", \
                "Somewhat\nagree\n(1)", "Strongly\nagree\n(2)"]
    #plt.xticks(np.arange(-2,3), x_labels, fontproperties=prop_bold, fontsize=24)
    plt.xticks(np.arange(-2, 3, 1), x_labels, fontproperties=prop_bold, fontsize=24)

    # Add slight ticks to x-axis and y-axis
    ax = plt.gca()
    #ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    #ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    plt.tick_params(axis='both', which='both', direction='out', bottom=True, top=False, left=True, right=False)

    plt.ylim(-0.5,49.51)

    # Custom y-tick parameters
    plt.tick_params(axis='y', which='both', right=True, labelright=True, \
                    left=True, labelleft=False, length=3, pad=10)

    # Set y-tick labels to the right of the plot
    gov_mechanism_titles_ordered_full_edit_space_save = [
        'Internal deployments = external deployments' 
        if title == 'Treat internal deployments similar to external deployments' 
        else 'Publish internal risk assessment results' 
        if title == 'Publish results of internal risk assessments' 
        else "No unsafe open-sourcing"
        if title == "No open-sourcing"
        else title 
        for title in gov_mechanism_titles_ordered_full
        ]
    
    ax.set_yticks(range(len(gov_mechanism_titles_ordered_full_edit_space_save)))
    ax.set_yticklabels(gov_mechanism_titles_ordered_full_edit_space_save, ha='left', \
                       fontproperties=prop_bold, fontsize=20)

    sns.despine(left=True, right=False)  # change left=False to left=True
    #ax.yaxis.set_ticks_position('none')
    plt.title("Mean agreement for AGI safety and governance practices\n", fontproperties=prop_bold, fontsize=30)
    #plt.savefig("figure_4.png", dpi=1000, bbox_inches = 'tight')
    plt.show()




def visualise_highest_proportion_mechanisms_figure_5(intermediate_dont_know_vals, \
                                                    gov_mechanism_titles_top_5_dont_know, \
                                                    intermediate_neither_vals, \
                                                    gov_mechanism_titles_top_5_neither) -> None:
    default_plotting_params()
    plt.rcParams['figure.figsize'] = (16,10)

    cmap_dont_know = discretise_cmap(cmap="Greys_r", n=5)
    cmap_neither = discretise_cmap(cmap="Greens_r", n=5)

    ### first plot ###
    plt.subplot(121)
    bars = plt.bar(x=np.arange(len(intermediate_dont_know_vals)), height=intermediate_dont_know_vals,
            color=cmap_dont_know, edgecolor="k", linewidth=2)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{round(height)}%', \
                 ha='center', va='bottom', fontsize=16)

    plt.xticks(np.arange(len(gov_mechanism_titles_top_5_dont_know)), gov_mechanism_titles_top_5_dont_know, rotation=90)
    plt.ylabel("percentage of responses (%)\n")
    plt.title("Highest proportion of \n\"I don't know\" responses\n\n")


    ### second plot ###
    plt.subplot(122)
    bars = plt.bar(x=np.arange(len(intermediate_neither_vals)), height=sorted(intermediate_neither_vals, reverse=True),
            color=cmap_neither, edgecolor="k", linewidth=2)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{round(height)}%', \
                 ha='center', va='bottom', fontsize=16)

    plt.xticks(np.arange(len(gov_mechanism_titles_top_5_neither)), gov_mechanism_titles_top_5_neither, rotation=90)
    plt.ylabel("percentage of responses (%)\n")
    plt.title("Highest proportion of \n\"Neither agree nor disagree\" responses\n\n")
    plt.tight_layout()
    sns.despine()
    plt.show()



def visualise_highest_proportion_mechanisms_figure_5_final(intermediate_dont_know_vals, \
                                                    gov_mechanism_titles_top_5_dont_know, \
                                                    intermediate_neither_vals, \
                                                    gov_mechanism_titles_top_5_neither) -> None:
    default_plotting_params()
    #plt.rcParams['figure.figsize'] = (21,12)
    plt.figure(figsize=(21,12), dpi=100)

    cmap_dont_know = discretise_cmap(cmap="Greys_r", n=5)
    cmap_neither = discretise_cmap(cmap="Greens_r", n=5)
    cmap_ = ["#f9c978", "#f3b247", "#eb9f20", "#cc8207", "#bc7805"][::-1]
    cmap_neither = ["#f9c978", "#f9c978", "#f9c978", "#f9c978", "#f3b247", "#eb9f20", "#cc8207", "#bc7805"][::-1]

    ### first plot ###
    ax1 = plt.subplot(121)
    bars = ax1.bar(x=np.arange(len(intermediate_dont_know_vals)), height=intermediate_dont_know_vals,
            color=cmap_, edgecolor="k", linewidth=2)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{round(height)}%', \
                 ha='center', va='bottom', fontsize=20, fontproperties=prop)
        ax1.vlines(x=bar.get_x() + bar.get_width()/2, ymin=height+3, ymax=40, linestyles='dashed', color='grey')

    ax1.xaxis.tick_top()
    gov_mechanism_titles_top_5_dont_know = ['Enterprise risk\nmanagement', 'Notify affected\nparties', \
                                        'Inter-lab\nscrutiny', 'Notify other\nlabs', 'Security\nstandards']

    ax1.set_xticks(np.arange(len(gov_mechanism_titles_top_5_dont_know)))
    ax1.set_xticklabels(gov_mechanism_titles_top_5_dont_know, rotation=45, ha='left', fontproperties=prop, fontsize=22)
    ax1.set_ylabel("Percentage of respondents\n", fontproperties=prop, fontsize=24)
    #title1 = ax1.set_title("Highest proportion of \n\"I don't know\" responses\n\n", \
             #fontproperties=prop_bold, fontsize=28)
    #title1.set_position([.5, 1.])
    
    ax1.annotate("Highest proportion of \n\"I don't know\" responses", 
             xy=(0.5, 1.32), xytext=(0, 10),
             xycoords='axes fraction', textcoords='offset points',
             size=28, ha='center', va='bottom',
             fontproperties=prop_bold)
    
    ax1.set_ylim(0,40)
    ax1.set_yticks(np.arange(0, 45, 5))
    ax1.set_yticklabels([f'{i}%' for i in range(0, 45, 5)], fontproperties=prop, fontsize=20)

    # Continue with the second plot...
    ### second plot ###
    ax2 = plt.subplot(122)
    intermediate_neither_vals_extended = intermediate_neither_vals + [intermediate_neither_vals[-1]]*3
    gov_mechanism_titles_top_5_neither_extended = ['Notify other\nlabs', 'Notify affected\nparties', \
                                               'Avoid capabilities\njumps', 'Tracking model\nweights', \
                                               'Gradual scaling', "Avoiding hype", \
                                              'Enterprise risk\nmanagement', \
                                              'Notify state actor\nbefore deployment']
    
    bars = ax2.bar(x=np.arange(len(intermediate_neither_vals_extended)), height=sorted(intermediate_neither_vals_extended, reverse=True),
            color=cmap_neither, edgecolor="k", linewidth=2)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{round(height)}%', \
                 ha='center', va='bottom', fontsize=16, fontproperties=prop)
        ax2.vlines(x=bar.get_x() + bar.get_width()/2, ymin=height+3, ymax=40, linestyles='dashed', color='grey')

    ax2.xaxis.tick_top()
    ax2.set_xticks(np.arange(len(gov_mechanism_titles_top_5_neither_extended)))
    ax2.set_xticklabels(gov_mechanism_titles_top_5_neither_extended, rotation=45, ha='left', fontproperties=prop, fontsize=20)
    ax2.set_ylabel("Percentage of respondents\n", fontproperties=prop, fontsize=24)
    ax2.set_ylim(0,35)
    ax2.set_yticks(np.arange(0, 40, 5))
    ax2.set_yticklabels([f'{i}%' for i in range(0, 40, 5)], fontproperties=prop, fontsize=20)
    #title2 = ax2.set_title("Highest proportion of \n\"Neither agree nor disagree\" responses\n", \
              #fontproperties=prop_bold, fontsize=28)
    #title2.set_position([.5, 1.02])
    
    ax2.annotate("Highest proportion of \n\"Neither agree nor disagree\" responses", 
             xy=(0.5, 1.32), xytext=(0, 10),
             xycoords='axes fraction', textcoords='offset points',
             size=28, ha='center', va='bottom',
             fontproperties=prop_bold)
    
    plt.tight_layout()
    sns.despine()
    #plt.savefig("figure5.pdf", dpi=1000, bbox_inches = 'tight')
    plt.show()


 


def visualise_mean_agreement_all_mechanisms(gov_mechanisms_means_full_vals_, \
                                           gov_mechanisms_se_full_vals_, \
                                           gov_mechanisms_means_full_idx_, \
                                           gov_mechanism_titles_ordered_full_edit) -> None:

    plt.rcParams['figure.figsize'] = (25, 40)
    cmap_ = discretise_cmap(cmap="Blues", n=len(gov_mechanisms_means_full_vals_))

    y_positions = np.arange(len(gov_mechanisms_means_full_vals_[::-1]))
    plt.scatter(x=gov_mechanisms_means_full_vals_[::-1], y=y_positions, \
                c=cmap_, edgecolor="k", linewidth=2, s=200, zorder=10)

    # Add horizontal lines for +/- standard error
    for idx, (mean, se) in enumerate(zip(gov_mechanisms_means_full_vals_[::-1], gov_mechanisms_se_full_vals_[:])):
        plt.hlines(y=idx, xmin=mean - se, xmax=mean + se, \
                   colors="grey", linewidth=2)

    # Add faint grey boxes around the scattered points and y-tick labels
    ax = plt.gca()
    for idx in y_positions:
        rect = patches.Rectangle((-2.1, idx - 0.4), 4.2, 0.8, linewidth=1, edgecolor='gray', facecolor='none', alpha=0.3)
        ax.add_patch(rect)

    plt.xlim(-2.1, 2.1)
    plt.axvline(0, c="grey", linestyle="--")
    gov_mechanism_titles = generate_gov_mechanisms_labels()
    gov_mechanism_titles_ordered_full = [gov_mechanism_titles[idx] for idx in gov_mechanisms_means_full_idx_][::-1]


    x_labels = ["Strongly\ndisagree\n(-2)", "Somewhat\ndisagree\n(-1)", "Neither agree\nnor disagree\n(0)", \
                "Somewhat\nagree\n(1)", "Strongly\nagree\n(2)"]
    plt.xticks(np.arange(-2,3), x_labels)

    # Add slight ticks to x-axis and y-axis
    ax = plt.gca()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    plt.tick_params(axis='both', which='both', direction='in', bottom=True, top=False, left=True, right=False)

    plt.ylim(-1,50)

    # Custom y-tick parameters
    plt.tick_params(axis='y', which='both', right=True, labelright=True, left=False, labelleft=False, length=3, pad=10)

    # Set y-tick labels to the right of the plot
    ax.set_yticks(range(len(gov_mechanism_titles_ordered_full_edit)))
    ax.set_yticklabels(gov_mechanism_titles_ordered_full_edit, ha='left')

    sns.despine(left=True)  # change left=False to left=True
    ax.yaxis.set_ticks_position('none')

    plt.show()

  

####################################
### DESCRIPTIVE STATISTICS f(x)s ###
####################################



def sort_by_first_element(lists, labels):
    sorted_indices = sorted(range(len(lists)), key=lambda i: lists[i][0], reverse=False)
    sorted_lists = [lists[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    return sorted_lists, sorted_labels



def compute_proportion_responses_strongly_agree_majority(responses: list[list[float]]):
    majority_strongly_agree = []
    for sublist in responses:
        if sublist[4] > 50:
            majority_strongly_agree.append(sublist)
    percentage_majority_strongly_agree = len(majority_strongly_agree) / 50 * 100
    return percentage_majority_strongly_agree



def compute_proportion_responses_somehwat_strongly_agree_majority(responses: list[list[float]]):
    majority_strongly_agree = []
    for sublist in responses:
        if (sublist[4] + sublist[3]) > 50:
            majority_strongly_agree.append(sublist)
    percentage_majority_strongly_agree = len(majority_strongly_agree) / 50 * 100
    return percentage_majority_strongly_agree



def mean_disagreement_across_mechanisms(gov_mechanisms_percentage_breakdown) -> float:
    total_disagreement = []
    for sublist in gov_mechanisms_percentage_breakdown:
        disagreement = sublist[0] + sublist[1]
        total_disagreement.append(disagreement)
    mean_disagreement = (sum(total_disagreement) / len(gov_mechanisms_percentage_breakdown))
    return mean_disagreement


def total_disagreement_across_mechanisms(gov_mechanisms_percentage_breakdown) -> float:
    total_disagreement = []
    for sublist in gov_mechanisms_percentage_breakdown:
        disagreement = sublist[0] + sublist[1]
        total_disagreement.append(disagreement)
    return total_disagreement



##############################
### STATISTICAL TEST f(x)s ###
##############################



def generate_gender_indices(numeric_data):
    gender_responses_ = series2list(numeric_data["gender"])[2:]
    gender_responses_str = [str(elem) for elem in gender_responses_]
    gender_responses_idx = list(zip(np.arange(len(gender_responses_str)).tolist(), gender_responses_str))
    man_idx = [elem[0] for elem in gender_responses_idx if (elem[1] == "1")]
    woman_idx = [elem[0] for elem in gender_responses_idx if (elem[1] == "2")]
    return man_idx, woman_idx



def filter_responses_gender(numeric_data, gov_mechanism, man_idx, woman_idx) -> list[int]:
    all_resps = series2list(numeric_data[gov_mechanism][2:])
    
    # Filter responses for each category
    man_resps = [all_resps[i] for i in man_idx]
    woman_resps = [all_resps[i] for i in woman_idx]
    
    def process_responses(resps):
        # Remove nan responses
        resps_str = [str(elem) for elem in resps]
        resps_nan_filter = [elem for elem in resps_str if (elem != "nan")]
        
        # Generate lists with "I don't know" responses recoded or removed (for mean calculation)
        resps_dont_know_replace = ["3" if (elem == "-88") else elem for elem in resps_nan_filter]
        resps_dont_know_remove = [elem for elem in resps_nan_filter if (elem != "-88")]
        
        # Convert elements of lists to integers
        resps_dont_know_replace_int = [int(elem) for elem in resps_dont_know_replace]
        resps_dont_know_remove_int = [int(elem) for elem in resps_dont_know_remove]
        
        return resps_dont_know_replace_int, resps_dont_know_remove_int
    
    man_processed = process_responses(man_resps)
    woman_processed = process_responses(woman_resps)
    
    return man_processed, woman_processed





def compare_distributions(responses_man, responses_woman):
    results = []
    
    for sublist_man, sublist_woman in zip(responses_man, responses_woman):
        # Combine the responses into a single list
        combined_responses = sublist_man + sublist_woman

        # Calculate the unique values and their counts
        unique_values, counts = np.unique(combined_responses, return_counts=True)

        # Create a contingency table
        contingency_table = np.zeros((2, len(unique_values)))
        
        for idx, value in enumerate(unique_values):
            contingency_table[0, idx] = sublist_man.count(value)
            contingency_table[1, idx] = sublist_woman.count(value)

        # Perform the chi-square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        # Store the results
        results.append((chi2, p, dof, expected))
    
    return results



def generate_sector_responses_chi_sq(numeric_data):
    sector_responses_ = series2list(numeric_data["sector"])[2:]
    sector_responses_str = [str(elem) for elem in sector_responses_]
    sector_responses_nan_filter = [elem for elem in sector_responses_str if (elem != "nan")]
    #sector_responses_separated = separate_elements(sector_responses_nan_filter)
    sector_responses_separated_1 = ["7" if (elem == "7,8") else elem for elem in sector_responses_nan_filter]
    sector_responses_separated_2 = ["5" if (elem == "5,7") else elem for elem in sector_responses_separated_1]
    sector_responses_separated_3 = ["1" if (elem == "1,2") else elem for elem in sector_responses_separated_2]
    sector_responses_separated_4 = ["7" if (elem == "4,5,7") else elem for elem in sector_responses_separated_3]
    sector_responses_for_test = ["9" if (elem == "-99") else elem for elem in sector_responses_separated_4]
    return sector_responses_for_test




def recode_sector_responses(numeric_data):
    
    recoding_dict = {
        '1': '1',
        '7': '2',
        '4': '3',
        '5': '3',
        '8': '4',
        '2': '4',
        '6': '4',
        '3': '4',
        '9': '5'
    }

    sector_responses_for_test = generate_sector_responses_chi_sq(numeric_data)
    sector_resps_recoded = [recoding_dict[value] for value in sector_responses_for_test]

    return sector_resps_recoded




def generate_recoded_sector_indices_chi_sq(numeric_data):
    sector_resps_recoded = recode_sector_responses(numeric_data)
    sector_resps_recoded_idx = list(zip(np.arange(len(sector_resps_recoded)), sector_resps_recoded))
    agi_lab_idx = [elem[0] for elem in sector_resps_recoded_idx if (elem[1] == "1")]
    academia_idx = [elem[0] for elem in sector_resps_recoded_idx if (elem[1] == "2")]
    civil_society_idx = [elem[0] for elem in sector_resps_recoded_idx if (elem[1] == "3")]
    other_idx = [elem[0] for elem in sector_resps_recoded_idx if (elem[1] == "4")]
    prefer_not_idx = [elem[0] for elem in sector_resps_recoded_idx if (elem[1] == "5")]
    return agi_lab_idx, academia_idx, civil_society_idx, other_idx, prefer_not_idx




def filter_responses_category(numeric_data, gov_mechanism, agi_lab_idx, academia_idx, \
                              civil_society_idx, other_idx, prefer_not_idx) -> list[int]:
    all_resps = series2list(numeric_data[gov_mechanism][2:])
    
    # Filter responses for each category
    agi_lab_resps = [all_resps[i] for i in agi_lab_idx]
    academia_resps = [all_resps[i] for i in academia_idx]
    civil_society_resps = [all_resps[i] for i in civil_society_idx]
    other_resps = [all_resps[i] for i in other_idx]
    prefer_not_resps = [all_resps[i] for i in prefer_not_idx]
    
    def process_responses(resps):
        # Remove nan responses
        resps_str = [str(elem) for elem in resps]
        resps_nan_filter = [elem for elem in resps_str if (elem != "nan")]
        
        # Generate lists with "I don't know" responses recoded or removed (for mean calculation)
        resps_dont_know_replace = ["3" if (elem == "-88") else elem for elem in resps_nan_filter]
        resps_dont_know_remove = [elem for elem in resps_nan_filter if (elem != "-88")]
        
        # Convert elements of lists to integers
        resps_dont_know_replace_int = [int(elem) for elem in resps_dont_know_replace]
        resps_dont_know_remove_int = [int(elem) for elem in resps_dont_know_remove]
        
        return resps_dont_know_replace_int, resps_dont_know_remove_int
    
    agi_lab_processed = process_responses(agi_lab_resps)
    academia_processed = process_responses(academia_resps)
    civil_society_processed = process_responses(civil_society_resps)
    other_processed = process_responses(other_resps)
    prefer_not_processed = process_responses(prefer_not_resps)
    
    return agi_lab_processed, academia_processed, civil_society_processed, other_processed, prefer_not_processed



def filter_responses_by_category(numeric_data, gov_mechanism, idx_list) -> list[int]:
    all_resps = series2list(numeric_data[gov_mechanism][2:])
    
    # Filter responses for the given category
    resps = [all_resps[i] for i in idx_list]

    # Remove nan responses
    resps_str = [str(elem) for elem in resps]
    resps_nan_filter = [elem for elem in resps_str if (elem != "nan")]

    # Generate lists with "I don't know" responses recoded or removed (for mean calculation)
    resps_dont_know_replace = ["3" if (elem == "-88") else elem for elem in resps_nan_filter]
    resps_dont_know_remove = [elem for elem in resps_nan_filter if (elem != "-88")]

    # Convert elements of lists to integers
    resps_dont_know_replace_int = [int(elem) for elem in resps_dont_know_replace]
    resps_dont_know_remove_int = [int(elem) for elem in resps_dont_know_remove]

    return resps_dont_know_replace_int, resps_dont_know_remove_int




##################################
### SUPPLEMENTARY FIGURE f(x)s ###
##################################


def compute_standard_errors_(data_array):
    # Calculate the standard deviation while ignoring "nan" entries
    std_dev = np.nanstd(data_array, axis=1)

    # Calculate the sample size for each list, ignoring "nan" entries
    sample_size = np.sum(~np.isnan(data_array), axis=1)

    # Compute the standard error for each list
    standard_errors = std_dev / np.sqrt(sample_size)

    return standard_errors


# DM – most of the supplementary f(x)s still need to be added here, but i do not think this is a good prioritisation of time



###################
### TABLE f(x)s ###
###################



def generate_latex_table_1(data, titles):
    # Define the integers we want to count
    integers = [-2, -1, 0, 1, 2, 3]

    # Create the table header
    header1 = "\\textbf{AGI safety and} & \\textbf{Strongly} & \\textbf{Somewhat} & \\textbf{Neither agree} & \\textbf{Somewhat} & \\textbf{Strongly} & \\textbf{I don't} & & & \\\\"
    header2 = "\\textbf{governance practice} & \\textbf{disagree (-2)} & \\textbf{disagree (-1)} & \\textbf{nor disagree (0)} & \\textbf{agree (1)} & \\textbf{agree (2)} & \\textbf{know (-88)} & \\textbf{Total disagreement} & \\textbf{Total agreement} & \\textbf{n} \\\\"
    
    # Create the table rows
    table_rows = []
    for sublist, title in zip(data, titles):
        row = f"{title}"
        counts = {i: sublist.count(i) for i in integers}
        for i in integers:
            row += f" & {counts[i]}"
        row += f" & {counts[-2] + counts[-1]} & {counts[1] + counts[2]} & {len(sublist)} \\\\"
        table_rows.append(row)

    # Join the table rows into a single string
    table_rows_str = "\n".join(table_rows)

    # Generate the LaTeX table
    latex_table = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{rotating}}

\\begin{{document}}

\\begin{{sidewaystable}}
\\fontsize{{10}}{{12}}\\selectfont
\\centering
\\begin{{tabular}}{{l *{{11}}{{r}}}}
\\toprule
{header1}
{header2}
\\midrule
{table_rows_str}
\\bottomrule
\\end{{tabular}}
\\end{{sidewaystable}}

\\end{{document}}
"""
    return latex_table



def generate_latex_table_2(data, titles):
    # Define the integers we want to count
    integers = [-2, -1, 0, 1, 2, 3]

    # Create the table header
    header1 = "\\textbf{AGI safety and} & \\textbf{Strongly} & \\textbf{Somewhat} & \\textbf{Neither agree} & \\textbf{Somewhat} & \\textbf{Strongly} & \\textbf{I don't} & & & \\\\"
    header2 = "\\textbf{governance practice} & \\textbf{disagree (-2)} & \\textbf{disagree (-1)} & \\textbf{nor disagree (0)} & \\textbf{agree (1)} & \\textbf{agree (2)} & \\textbf{know (-88)} & \\textbf{Total disagreement} & \\textbf{Total agreement} & \\textbf{n} \\\\"
    
    # Create the table rows
    table_rows = []
    for sublist, title in zip(data, titles):
        row = f"{title}"
        counts = {i: sublist.count(i) for i in integers}
        length = len(sublist)
        percentages = {i: counts[i] / length * 100 for i in integers}
        for i in integers:
            row += f" & {percentages[i]:.1f}\\%"
        total_agreement = percentages[1] + percentages[2]
        total_disagreement = percentages[-2] + percentages[-1]
        row += f" & {total_disagreement:.1f}\\% & {total_agreement:.1f}\\% & {length} \\\\"
        table_rows.append(row)

    # Join the table rows into a single string
    table_rows_str = "\n".join(table_rows)

    # Generate the LaTeX table
    latex_table = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{rotating}}

\\begin{{document}}

\\begin{{sidewaystable}}
\\fontsize{{10}}{{12}}\\selectfont
\\centering
\\begin{{tabular}}{{l *{{11}}{{r}}}}
\\toprule
{header1}
{header2}
\\midrule
{table_rows_str}
\\bottomrule
\\end{{tabular}}
\\end{{sidewaystable}}

\\end{{document}}
"""
    return latex_table





def generate_latex_table_3(data, titles):
    # Create the LaTeX table rows
    table_rows = []

    for i, (sublist, title) in enumerate(zip(data, titles)):
        mean = np.mean(sublist)
        median = np.median(sublist)
        std_error = stats.sem(sublist)
        variance = np.var(sublist, ddof=1)
        first_quartile = np.percentile(sublist, 25)
        third_quartile = np.percentile(sublist, 75)
        iqr = third_quartile - first_quartile
        length = len(sublist)

        row = f"{title} & {mean:.2f} & {median:.2f} & {std_error:.2f} & {variance:.2f} & {first_quartile:.2f} & {third_quartile:.2f} & {iqr:.2f} & {length} \\\\"
        table_rows.append(row)

    # Join the table rows into a single string
    table_rows_str = "\n".join(table_rows)

    # Generate the LaTeX table
    latex_table = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{rotating}}
\\usepackage{{adjustbox}}

\\begin{{document}}
\\begin{{sidewaystable}}
\\centering
\\fontsize{{8}}{{10}}\\selectfont
\\begin{{adjustbox}}{{width=\\textwidth, center}}
\\begin{{tabular}}{{l r r r r r r r r}}
\\toprule
\\textbf{{Title}} & \\textbf{{Mean}} & \\textbf{{Median}} & \\textbf{{Standard Error}} & \\textbf{{Variance}} & \\textbf{{First Quartile}} & \\textbf{{Third Quartile}} & \\textbf{{Inter-quartile Range}} & \\textbf{{Length}} \\\\
\\midrule
{table_rows_str}
\\bottomrule
\\end{{tabular}}
\\end{{adjustbox}}
\\end{{sidewaystable}}

\\end{{document}}
"""
    return latex_table




def generate_latex_table_4(agi_lab_data, academia_data, civil_society_data, titles, fontsize=10):
    table_rows = []

    for i, (agi_lab_sublist, academia_sublist, civil_society_sublist, title) in enumerate(zip(agi_lab_data, academia_data, civil_society_data, titles)):
        agi_lab_mean = np.nanmean(agi_lab_sublist)
        academia_mean = np.nanmean(academia_sublist)
        civil_society_mean = np.nanmean(civil_society_sublist)
        agi_lab_std_error = stats.sem(agi_lab_sublist, nan_policy='omit')
        academia_std_error = stats.sem(academia_sublist, nan_policy='omit')
        civil_society_std_error = stats.sem(civil_society_sublist, nan_policy='omit')
        agi_lab_length = len(agi_lab_sublist)
        academia_length = len(academia_sublist)
        civil_society_length = len(civil_society_sublist)

        row = f"{title} & {agi_lab_mean:.2f} & {academia_mean:.2f} & {civil_society_mean:.2f} & {agi_lab_std_error:.2f} & {academia_std_error:.2f} & {civil_society_std_error:.2f} & {agi_lab_length} & {academia_length} & {civil_society_length} \\\\"
        table_rows.append(row)

    table_rows_str = "\n".join(table_rows)

    latex_table = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{array}}
\\usepackage{{rotating}}

\\begin{{document}}

\\begin{{sidewaystable}}
    \\centering
    \\fontsize{{{fontsize}}}{{12}}\\selectfont
    \\begin{{tabular}}{{l*{{9}}{{r}}}}
        \\toprule
        & \\multicolumn{{3}}{{c}}{{\\textbf{{Mean}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Standard error}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{n}}}} \\\\
        \\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}} \\cmidrule(lr){{8-10}}
        \\textbf{{AGI safety and governance practice}} & \\textbf{{AGI Lab}} & \\textbf{{Academia}} & \\textbf{{Civil society}} & \\textbf{{AGI Lab}} & \\textbf{{Academia}} & \\textbf{{Civil society}} & \\textbf{{AGI Lab}} & \\textbf{{Academia}} & \\textbf{{Civil society}} \\\\
        \\midrule
        {table_rows_str}
        \\bottomrule
    \\end{{tabular}}
\\end{{sidewaystable}}

\\end{{document}}
"""
    return latex_table





def generate_latex_table_5(agi_lab_data, everyone_else_data, titles, fontsize=10):
    table_rows = []

    for i, (agi_lab_sublist, everyone_else_sublist, title) in enumerate(zip(agi_lab_data, everyone_else_data, titles)):
        agi_lab_mean = np.mean(agi_lab_sublist)
        everyone_else_mean = np.mean(everyone_else_sublist)
        agi_lab_std_error = stats.sem(agi_lab_sublist)
        everyone_else_std_error = stats.sem(everyone_else_sublist)
        agi_lab_length = len(agi_lab_sublist)
        everyone_else_length = len(everyone_else_sublist)

        row = f"{title} & {agi_lab_mean:.2f} & {everyone_else_mean:.2f} & {agi_lab_std_error:.2f} & {everyone_else_std_error:.2f} & {agi_lab_length} & {everyone_else_length} \\\\"
        table_rows.append(row)

    table_rows_str = "\n".join(table_rows)

    latex_table = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{array}}
\\usepackage{{rotating}}

\\begin{{document}}

\\begin{{sidewaystable}}
    \\centering
    \\fontsize{{{fontsize}}}{{12}}\\selectfont
    \\begin{{tabular}}{{l*{{6}}{{r}}}}
        \\toprule
        & \\multicolumn{{2}}{{c}}{{\\textbf{{Mean}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Standard error}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{n}}}} \\\\
        \\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}}
        \\textbf{{AGI safety and governance practice}} & \\textbf{{AGI Lab}} & \\textbf{{Everyone else}} & \\textbf{{AGI Lab}} & \\textbf{{Everyone else}} & \\textbf{{AGI Lab}} & \\textbf{{Everyone else}} \\\\
        \\midrule
        {table_rows_str}
        \\bottomrule
    \\end{{tabular}}
\\end{{sidewaystable}}

\\end{{document}}
"""
    return latex_table





def generate_latex_table_6(men_data, women_data, titles, fontsize=10):
    table_rows = []

    for i, (men_sublist, women_sublist, title) in enumerate(zip(men_data, women_data, titles)):
        men_mean = np.mean(men_sublist)
        women_mean = np.mean(women_sublist)
        men_std_error = stats.sem(men_sublist)
        women_std_error = stats.sem(women_sublist)
        men_length = len(men_sublist)
        women_length = len(women_sublist)

        row = f"{title} & {men_mean:.2f} & {women_mean:.2f} & {men_std_error:.2f} & {women_std_error:.2f} & {men_length} & {women_length} \\\\"
        table_rows.append(row)

    table_rows_str = "\n".join(table_rows)

    latex_table = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{array}}
\\usepackage{{rotating}}

\\begin{{document}}

\\begin{{sidewaystable}}
    \\centering
    \\fontsize{{{fontsize}}}{{12}}\\selectfont
    \\begin{{tabular}}{{l*{{6}}{{r}}}}
        \\toprule
        & \\multicolumn{{2}}{{c}}{{\\textbf{{Mean}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Standard error}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{n}}}} \\\\
        \\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}}
        \\textbf{{AGI safety and governance practice}} & \\textbf{{Men}} & \\textbf{{Women}} & \\textbf{{Men}} & \\textbf{{Women}} & \\textbf{{Men}} & \\textbf{{Women}} \\\\
        \\midrule
        {table_rows_str}
        \\bottomrule
    \\end{{tabular}}
\\end{{sidewaystable}}

\\end{{document}}
"""
    return latex_table



def generate_latex_table_7(label_data):
    gender_responses_ = series2list(label_data["gender"])[2:]
    gender_responses_str = [str(elem) for elem in gender_responses_]
    gender_responses_nan_filter = [elem for elem in gender_responses_str if (elem != "nan")]

    data = gender_responses_nan_filter

    # Count the unique string elements
    counts = Counter(data)

    # Calculate the total number of elements
    total = len(data)

    # Calculate the percentage breakdown for each unique string element
    percentages = {key: (value / total) * 100 for key, value in counts.items()}

    # Create a DataFrame to store the results
    results = pd.DataFrame(columns=["Gender", "Count", "Percentage"])

    # Fill the DataFrame with the count and percentage breakdown for each unique string element
    for key, value in counts.items():
        results = results.append({"Gender": key, "Count": value, "Percentage": f"{percentages[key]:.1f}%"}, ignore_index=True)

    # Set the index to the "Gender" column
    results.set_index("Gender", inplace=True)

    # Generate the LaTeX table
    latex_table = results.to_latex()

    # Include the necessary LaTeX preamble and table environment
    latex_preamble = r"""\documentclass{article}
\usepackage{booktabs}
\begin{document}
"""

    latex_end = r"""
\end{document}"""

    latex_full_document = latex_preamble + "\n\\begin{table}\n\\centering\n" + latex_table + "\n\\end{table}\n" + latex_end

    return latex_full_document




def generate_latex_table_8(data, unique_labels):
    # Initialize a dictionary to store counts for each unique label
    label_counts = defaultdict(int)

    # Count the occurrences of each unique label in the data
    for label in unique_labels:
        for item in data:
            if label in item:
                label_counts[label] += 1

    # Calculate the total number of labels
    total = sum(label_counts.values())

    # Calculate the percentage breakdown for each unique label
    percentages = {key: (value / total) * 100 for key, value in label_counts.items()}

    # Create the LaTeX table rows
    table_rows = []
    for key, value in label_counts.items():
        percentage = percentages[key]
        row = f"{key} & & {percentage:.1f}\% & {value} \\\\"
        table_rows.append(row)

    # Join the table rows into a single string
    table_rows_str = "\n".join(table_rows)

    # Generate the LaTeX table
    latex_table = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{caption}}
\\usepackage{{tabularx}}

\\begin{{document}}

\\begin{{table}}[ht]
\\centering
\\begin{{tabularx}}{{1.1\\textwidth}}{{>{{\\raggedright\\arraybackslash}}X >{{\\raggedright\\arraybackslash}}X r r}}
\\toprule
\\textbf{{Sector}} & \\textbf{{Sector subgroup}} & \\textbf{{Percentage of total sample}} & \\textbf{{Raw frequency}} \\\\
\\midrule
{table_rows_str}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}

\\end{{document}}
"""
    return latex_table






