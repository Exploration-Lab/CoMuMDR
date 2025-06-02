
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser


def make_plot(epoch_scores, output_file):
    plt.figure(figsize=(4*1.5, 3*1.5), dpi=200)
    sns.lineplot(data=epoch_scores, x='Epoch', y='F1', hue='Model', style='Method', markers=True, dashes=False, alpha=0.7)
    plt.ylim(0, 1)
    plt.xlim(0, 11)
    # remove the right and top spines
    sns.despine()
    # set the font size for the axes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    # set fontsize of the legend
    plt.legend(fontsize=12, title_fontsize='13', loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2)

    # Write the value of the best F1 score for each model and method
    best_scores = epoch_scores.loc[epoch_scores.groupby(['Model', 'Method'])['F1'].idxmax()]
    for _, row in best_scores.iterrows():
        plt.text(row['Epoch'], row['F1'] + 0.02, f"{row['F1']:.4f}", ha='center', va='bottom', fontsize=12)


    plt.savefig(output_file, bbox_inches='tight')

if __name__ == '__main__':
    # setup argument parser to take the csv file and output file as arguments
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='epoch_scores.csv', help='Input CSV file with epoch scores')
    parser.add_argument('--output', type=str, default='epoch_scores.pdf', help='Output PDF file for the plot')
    args = parser.parse_args()
    # Read the input file
    epoch_scores = pd.read_csv(args.input)
    # Create the plot
    make_plot(epoch_scores, args.output)
    print(f"Plot saved to {args.output}")



