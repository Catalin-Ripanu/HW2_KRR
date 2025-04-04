from typing import List, Dict, Tuple, Iterator
from bayes_net import BayesNet, JunctionTree, sumout, normalize, AirQualityBayesNet
from itertools import product
import numpy as np
from argparse import ArgumentParser, Namespace
import pandas as pd
import matplotlib.pyplot as plt
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import os
import random
import requests
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import zipfile
import io
import argparse
import seaborn as sns

def all_dicts(variables: List[str]) -> Iterator[Dict[str, int]]:
    for keys in product(*([[0, 1]] * len(variables))):
        yield dict(zip(variables, keys))


def cross_entropy(bn1: BayesNet, bn2: BayesNet, nsamples: int = None) -> float:
    cross_ent = 0.0
    if nsamples is None:
        bn1_vars = bn1.nodes.keys()
        for sample in all_dicts(bn1_vars):
            cross_ent -= np.exp(bn1.sample_log_prob(sample)) * bn2.sample_log_prob(
                sample
            )
    else:
        for _ in range(nsamples):
            cross_ent -= bn2.sample_log_prob(bn1.sample())
        cross_ent /= nsamples
    return cross_ent


def read_samples(file_name: str) -> List[Dict[str, int]]:
    samples = []
    with open(file_name, "r") as handler:
        lines = handler.readlines()
        # read first line of file to get variables in order
        variables = [str(v) for v in lines[0].split()]

        for i in range(1, len(lines)):
            vals = [int(v) for v in lines[i].split()]
            sample = dict(zip(variables, vals))
            samples.append(sample)

    return samples


class MLEBayesNet(BayesNet):
    """
    Placeholder class for the Bayesian Network that will learn CPDs using the frequentist MLE
    """

    def __init__(self, bn_file: str = "data/bnet"):
        super(MLEBayesNet, self).__init__(bn_file=bn_file)
        self._reset_cpds()

    def _reset_cpds(self) -> None:
        """
        Reset the conditional probability distributions to a value of 0.5
        The values have to be **learned** from the samples.
        """
        for node_name in self.nodes:
            self.nodes[node_name].cpd["prob"] = 0.5

    def learn_cpds(self, samples: List[Dict[str, int]], alpha: float = 1.0) -> None:
        """
        Learn the CPD using Maximum Likelihood Estimation

        Args:
            samples: List of dictionaries with samples read from input file
            alpha: Laplace smoothing parameter (default = 1.0)
        """

        # for each node in BN
        for node_name in self.nodes:

            # get the current_node
            current_node = self.nodes[node_name]
            parent_names = []

            # get the parents' names of the current node
            for par_name in current_node.parent_nodes:
                parent_names.append(par_name.var_name)

            # take into account each combination
            for par_values in product([0, 1], repeat=len(parent_names)):

                # filter samples which do not meet the following criteria
                match_samples = [
                    s
                    for s in samples
                    if all(
                        s[pname] == pval
                        for pname, pval in dict(zip(parent_names, par_values)).items()
                    )
                ]

                # count all samples in which the current_node is observed
                pos_count = sum(1 for sample in match_samples if sample[node_name] == 1)

                # compute the probability using Laplace smoothing in order to not make 0's
                smoothed_prob = (pos_count + alpha) / (len(match_samples) + 2 * alpha)

                current_node.cpd = current_node.cpd.sort_index()

                # update the CPD of the current_node -> value = 1
                index = tuple([1] + list(par_values))
                current_node.cpd.loc[index, "prob"] = smoothed_prob

                # update the CPD of the current_node -> value = 0
                index = tuple([0] + list(par_values))
                current_node.cpd.loc[index, "prob"] = 1 - smoothed_prob


class EMBayesNet(MLEBayesNet):
    def __init__(self, bn_file: str = "data/bnet") -> None:
        """Initialize EMBayesNet with proper variable name mapping."""
        # First read the network structure to get the variable mappings
        self.reverse_name_mapping = {}
        self.forward_name_mapping = {}
        
        with open(bn_file, 'r') as f:
            num_nodes = int(f.readline().split()[0])
            for _ in range(num_nodes):
                line = f.readline().strip()
                node = line.split(';')[0].strip()
                
                # Convert back to original name
                if node == 'Temp':
                    original_name = 'T'
                elif '_' in node:
                    # Convert safe name back to original format
                    # For example: CO_GT -> CO(GT), PT08_S1_CO -> PT08.S1(CO)
                    parts = node.split('_')
                    if parts[0] == 'PT08':
                        # Handle PT08 sensor names
                        original_name = f"{parts[0]}.{parts[1]}({'.'.join(parts[2:])})"
                    else:
                        # Handle other measurement names
                        original_name = f"{parts[0]}({'.'.join(parts[1:])})"
                else:
                    # Simple names like RH, AH stay as is
                    original_name = node
                
                self.reverse_name_mapping[node] = original_name
                self.forward_name_mapping[original_name] = node
        
        # Print mappings for debugging
        print("\nVariable name mappings in EMBayesNet:")
        for safe_name, original_name in self.reverse_name_mapping.items():
            print(f"{safe_name} -> {original_name}")
        
        # Now initialize the parent class
        super(EMBayesNet, self).__init__(bn_file)
        self.bn_file = bn_file
        self.cpds = {}
        self.logger = logging.getLogger(__name__)
        self.log_likelihoods = []
        self.param_history = []
        self.imputed_samples = []  # Initialize as empty list
        self._jt_cache = {}  # Cache for junction trees
        
        # Print mappings for debugging
        print("\nVariable name mappings:")
        for safe_name, original_name in self.reverse_name_mapping.items():
            print(f"{safe_name} -> {original_name}")

    def visualize_missing_pattern(self, df: pd.DataFrame, output_file: str):
        """
        Create a visualization of the missing data pattern.
        
        Args:
            df: DataFrame with missing values
            output_file: Path to save the visualization
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Calculate missing percentages
            missing_mask = df == 2
            missing_percentages = (missing_mask.sum() / len(df) * 100)
            total_missing = missing_mask.sum().sum() / (len(df) * len(df.columns)) * 100
            
            # Sort variables by missing percentage
            sorted_vars = missing_percentages.sort_values(ascending=False).index
            
            # Create figure
            plt.figure(figsize=(15, 10))
            
            # Create heatmap with better visibility
            plt.imshow(missing_mask[sorted_vars].T, cmap='RdYlBu_r', aspect='auto', 
                      interpolation='nearest')
            plt.colorbar(label='Missing')
            
            # Add variable names with percentages and better formatting
            plt.yticks(range(len(sorted_vars)), 
                      [f"{var}\n({missing_percentages[var]:.1f}%)" for var in sorted_vars])
            plt.xticks([])  # Hide x-axis ticks
            
            plt.title(f'Missing Data Pattern\n(Variables sorted by % missing)\nTotal Missing: {total_missing:.1f}%', 
                     pad=20, 
                     fontsize=14)
            plt.xlabel('Samples')
            plt.ylabel('Variables')
            
            # Add grid for better readability
            plt.grid(True, which='major', color='gray', linestyle='-', alpha=0.2)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating missing pattern visualization: {str(e)}")
            raise

    def visualize_imputed_pattern(self, samples: List[Dict[str, int]], output_file: str):
        """
        Create a visualization of the imputed values pattern.
        
        Args:
            samples: List of dictionaries containing the samples
            output_file: Path to save the visualization
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Convert imputed samples to DataFrame
            if isinstance(self.imputed_samples[-1], pd.DataFrame):
                df = self.imputed_samples[-1]
            else:
                df = pd.DataFrame(samples)
            
            # Ensure numeric values
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            binary_mask = df.copy()
            
            # Calculate percentages
            zeros_percentage = (binary_mask == 0).mean() * 100
            ones_percentage = (binary_mask == 1).mean() * 100
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot imputed values pattern with better visibility
            im1 = ax1.imshow(binary_mask.T, cmap='coolwarm', aspect='auto', 
                           interpolation='nearest')
            plt.colorbar(im1, ax=ax1, label='Value (0 or 1)')
            
            # Add variable names with better formatting
            ax1.set_yticks(range(len(df.columns)))
            ax1.set_yticklabels([f"{col}" for col in df.columns])
            ax1.set_xticks([])  # Hide x-axis ticks
            
            # Add grid for better readability
            ax1.grid(True, which='major', color='gray', linestyle='-', alpha=0.2)
            
            ax1.set_title('Imputed Values Pattern')
            ax1.set_xlabel('Samples')
            ax1.set_ylabel('Variables')
            
            # Plot percentage distribution
            distribution_df = pd.DataFrame({
                'Zeros %': zeros_percentage,
                'Ones %': ones_percentage
            }).T
            
            im2 = ax2.imshow(distribution_df, cmap='coolwarm', aspect='auto')
            plt.colorbar(im2, ax=ax2, label='Percentage')
            
            # Add percentage labels with better visibility
            for i in range(2):
                for j, var in enumerate(df.columns):
                    value = distribution_df.iloc[i, j]
                    # Add background box for better visibility
                    ax2.text(j, i, f'{value:.1f}%',
                            ha='center', va='center',
                            color='black' if 20 < value < 80 else 'white',
                            fontweight='bold',
                            bbox=dict(facecolor='white' if 20 < value < 80 else 'none',
                                    alpha=0.5,
                                    edgecolor='none'))
            
            # Add labels with better formatting
            ax2.set_xticks(range(len(df.columns)))
            ax2.set_xticklabels(df.columns, rotation=45, ha='right')
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['Zeros %', 'Ones %'])
            
            ax2.set_title('Value Distribution After Imputation')
            
            # Add grid for better readability
            ax2.grid(True, which='major', color='gray', linestyle='-', alpha=0.2)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating imputed pattern visualization: {str(e)}")
            raise

    def visualize_imputation_comparison(self, true_samples: List[Dict[str, int]], 
                                      missing_samples: List[Dict[str, int]], 
                                      output_file: str):
        """
        Create a visualization comparing true values with imputed values.
        
        Args:
            true_samples: List of dictionaries containing the true values
            missing_samples: List of dictionaries containing samples with missing values
            output_file: Path to save the visualization
        """
        # Convert to DataFrames
        true_df = pd.DataFrame(true_samples)
        missing_df = pd.DataFrame(missing_samples)
        
        # Print debugging information
        print("\nDebugging DataFrame shapes and columns:")
        print("True DataFrame shape:", true_df.shape)
        print("True DataFrame columns:", true_df.columns.tolist())
        print("Missing DataFrame shape:", missing_df.shape)
        print("Missing DataFrame columns:", missing_df.columns.tolist())
        
        # Handle imputed samples based on type
        if isinstance(self.imputed_samples[-1], pd.DataFrame):
            imputed_df = self.imputed_samples[-1].copy()  # Make a copy to avoid modifying original
            print("Imputed DataFrame shape:", imputed_df.shape)
            print("Imputed DataFrame columns:", imputed_df.columns.tolist())
        else:
            # Convert dictionary of tuples to DataFrame
            imputed_data = []
            for sample in missing_samples:
                sample_tuple = tuple(sorted(sample.items()))
                if sample_tuple in self.imputed_samples[-1]:
                    imputed_data.append(self.imputed_samples[-1][sample_tuple])
                else:
                    # If not found, use original sample as fallback
                    imputed_data.append(sample)
            imputed_df = pd.DataFrame(imputed_data)
            print("Created Imputed DataFrame shape:", imputed_df.shape)
            print("Created Imputed DataFrame columns:", imputed_df.columns.tolist())
        
        # Ensure all DataFrames have the same columns and indices
        all_columns = sorted(list(set(true_df.columns) | set(missing_df.columns) | set(imputed_df.columns)))
        print("\nAll columns:", all_columns)
        
        # Create a full-sized imputed DataFrame initialized with original values
        full_imputed_df = missing_df.copy()
        
        # Update only the rows that were imputed
        if len(imputed_df) < len(missing_df):
            # Find indices where values were missing (equal to 2)
            missing_indices = missing_df.apply(lambda x: (x == 2).any(), axis=1)
            missing_rows = missing_indices[missing_indices].index
            
            # Ensure we have enough imputed values
            if len(imputed_df) >= len(missing_rows):
                # Convert imputed_df to the same dtype as full_imputed_df
                imputed_values = imputed_df.astype(full_imputed_df.dtypes.to_dict())
                full_imputed_df.loc[missing_rows] = imputed_values
            else:
                print(f"Warning: Not enough imputed values. Expected {len(missing_rows)}, got {len(imputed_df)}")
                # Update as many rows as we can
                n_rows_to_update = len(imputed_df)
                rows_to_update = missing_rows[:n_rows_to_update]
                
                # Convert imputed_df to the same dtype as full_imputed_df
                imputed_values = imputed_df.iloc[:n_rows_to_update].astype(full_imputed_df.dtypes.to_dict())
                full_imputed_df.loc[rows_to_update] = imputed_values
                
                # Mark remaining missing rows with NaN
                remaining_rows = missing_rows[n_rows_to_update:]
                if len(remaining_rows) > 0:
                    full_imputed_df.loc[remaining_rows] = np.nan
        else:
            full_imputed_df = imputed_df.astype(full_imputed_df.dtypes.to_dict())
        
        # Reindex all DataFrames with the same columns
        true_df = true_df.reindex(columns=all_columns)
        missing_df = missing_df.reindex(columns=all_columns)
        full_imputed_df = full_imputed_df.reindex(columns=all_columns)
        
        # Reset indices to ensure alignment
        true_df = true_df.reset_index(drop=True)
        missing_df = missing_df.reset_index(drop=True)
        full_imputed_df = full_imputed_df.reset_index(drop=True)
        
        print("\nAfter reindexing:")
        print("True DataFrame shape:", true_df.shape)
        print("Missing DataFrame shape:", missing_df.shape)
        print("Full Imputed DataFrame shape:", full_imputed_df.shape)
        
        # Ensure all values are numeric
        for df in [true_df, missing_df, full_imputed_df]:
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Verify alignment before comparison
        print("\nVerifying alignment:")
        print("Index equality:", 
              (true_df.index == missing_df.index).all() and 
              (true_df.index == full_imputed_df.index).all())
        print("Column equality:", 
              (true_df.columns == missing_df.columns).all() and 
              (true_df.columns == full_imputed_df.columns).all())
        
        # Create missing mask and handle NaN values
        missing_mask = missing_df == 2
        
        # For the comparison, treat NaN values in imputed data as different
        imputed_comparison = full_imputed_df.fillna(-999)  # Use a sentinel value for NaN
        
        # Compute differences with explicit alignment
        differences = ((true_df != imputed_comparison) & missing_mask).astype(float)
        
        # Set style for better readability
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Difference heatmap
        plt.subplot(2, 2, 1)
        plt.imshow(differences.T, cmap='Reds', aspect='auto')
        plt.colorbar(label='Different (1) vs Same (0)')
        
        # Add variable names
        plt.yticks(range(len(true_df.columns)), true_df.columns)
        plt.xticks([])  # Hide x-axis ticks
        
        plt.title('Differences Between True and Imputed Values\n(Only for Missing Values)', pad=10)
        plt.xlabel('Samples')
        plt.ylabel('Variables')
        
        # 2. Agreement percentages
        plt.subplot(2, 2, 2)
        agreement_pct = (~differences.astype(bool)).mean() * 100  # Convert to bool first
        agreement_df = pd.DataFrame({
            'Agreement %': agreement_pct
        }).T
        
        plt.imshow(agreement_df, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        plt.colorbar(label='Agreement Percentage')
        
        # Add percentage labels
        for i in range(1):  # Changed from 2 to 1 since we only have one row
            for j, var in enumerate(true_df.columns):
                value = agreement_pct[var]
                plt.text(j, i, f'{value:.1f}%',
                        ha='center', va='center',
                        color='black' if value > 50 else 'white',
                        fontweight='bold')
        
        # Add variable names
        plt.xticks(range(len(true_df.columns)), true_df.columns, rotation=45, ha='right')
        plt.yticks([])  # Hide y-axis ticks since we only have one row
        
        plt.title('Agreement Percentage by Variable\n(Only for Missing Values)', pad=10)
        
        # 3. Value distribution comparison
        plt.subplot(2, 1, 2)
        
        # Prepare data for plotting
        variables = list(true_df.columns)  # Convert to list to avoid index issues
        x = np.arange(len(variables))
        width = 0.35
        
        # Initialize arrays for true and imputed values
        true_values = []
        imputed_values = []
        
        # Collect data for each variable
        for var in variables:
            # Get indices where values were missing
            missing_mask_var = missing_df[var] == 2
            
            if missing_mask_var.sum() > 0:
                true_values.append((true_df.loc[missing_mask_var, var] == 1).mean() * 100)
                imputed_values.append((full_imputed_df.loc[missing_mask_var, var] == 1).mean() * 100)
            else:
                true_values.append(0)
                imputed_values.append(0)
        
        # Create bars
        rects1 = plt.bar(x - width/2, true_values, width, label='True', color='lightblue')
        rects2 = plt.bar(x + width/2, imputed_values, width, label='Imputed', color='lightcoral')
        
        # Add value labels on top of bars
        plt.bar_label(rects1, fmt='%.1f%%', padding=3)
        plt.bar_label(rects2, fmt='%.1f%%', padding=3)
        
        # Customize plot
        plt.title('Value Distribution Comparison\n(Only for Missing Values)', pad=10)
        plt.xlabel('Variables')
        plt.ylabel('Percentage of 1s')
        plt.xticks(x, variables, rotation=45, ha='right')
        plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate missing counts and percentages for each variable
        missing_counts = {}
        missing_percentages = {}
        total_missing_count = 0
        
        for var in variables:
            # Count values equal to 2 (missing) in the missing_df
            count = (missing_df[var] == 2).sum()
            missing_counts[var] = count
            missing_percentages[var] = (count / len(missing_df)) * 100
            total_missing_count += count
        
        # Calculate total percentage
        total_values = len(missing_df) * len(variables)
        total_missing_percentage = (total_missing_count / total_values) * 100
        
        print("\n=== Missing Data Summary ===")
        print(f"Total samples: {len(missing_df):,}")
        print(f"Total variables: {len(variables)}")
        print(f"Total missing values: {total_missing_count:,} ({total_missing_percentage:.1f}%)")
        
        print("\nMissing values by variable (sorted by percentage):")
        print("-" * 50)
        print(f"{'Variable':<15} {'Count':>10} {'Percentage':>12}")
        print("-" * 50)
        
        # Sort variables by missing percentage in descending order
        sorted_vars = sorted(variables, key=lambda x: missing_percentages[x], reverse=True)
        for var in sorted_vars:
            count = missing_counts[var]
            percentage = missing_percentages[var]
            print(f"{var:<15} {int(count):>10,} {percentage:>11.1f}%")

    def compare_with_true_values(self, true_samples: List[Dict[str, int]], 
                               missing_samples: List[Dict[str, int]], 
                               output_dir: str = './'):
        """
        Comprehensive comparison between imputed and true values.
        Creates both per-variable metrics and a global confusion matrix.
        """
        results = {}
        detailed_metrics = {}
        
        # Convert to DataFrames for easier handling
        true_df = pd.DataFrame(true_samples)
        missing_df = pd.DataFrame(missing_samples)
        
        # Handle imputed samples based on type
        if isinstance(self.imputed_samples[-1], pd.DataFrame):
            imputed_df = self.imputed_samples[-1]
        else:
            # Convert dictionary of tuples to DataFrame
            imputed_data = []
            for sample in missing_samples:
                sample_tuple = tuple(sorted(sample.items()))
                if sample_tuple in self.imputed_samples[-1]:
                    imputed_data.append(self.imputed_samples[-1][sample_tuple])
                else:
                    # If not found, use original sample as fallback
                    imputed_data.append(sample)
            imputed_df = pd.DataFrame(imputed_data)
        
        # Ensure all values are numeric
        for df in [true_df, missing_df, imputed_df]:
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # For global confusion matrix
        all_true_values = []
        all_imputed_values = []
        
        # Analyze each variable
        for var in true_df.columns:
            # Get indices where values were missing
            missing_mask = missing_df[var] == 2
            
            if missing_mask.sum() > 0:
                true_values = true_df.loc[missing_mask, var]
                imputed_values = imputed_df.loc[missing_mask, var]
                
                # Add to global arrays
                all_true_values.extend(true_values)
                all_imputed_values.extend(imputed_values)
                
                # Calculate per-variable metrics
                acc = accuracy_score(true_values, imputed_values)
                cm = confusion_matrix(true_values, imputed_values)
                report = classification_report(true_values, imputed_values, output_dict=True)
                
                results[var] = {
                    'accuracy': acc,
                    'confusion_matrix': cm,
                    'classification_report': report
                }
                
                # Visualize per-variable confusion matrix
                plt.figure(figsize=(8, 6))
                plt.imshow(cm, cmap='Blues', aspect='auto')
                plt.title(f'Confusion Matrix for {var}')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.savefig(f'{output_dir}/confusion_matrix_{var}.png')
                plt.close()
                
                detailed_metrics[var] = report
        
        # Calculate and visualize global confusion matrix
        if all_true_values:
            global_cm = confusion_matrix(all_true_values, all_imputed_values)
            global_acc = accuracy_score(all_true_values, all_imputed_values)
            global_report = classification_report(all_true_values, all_imputed_values, output_dict=True)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(global_cm, cmap='Blues', aspect='auto')
            plt.title('Global Confusion Matrix (All Variables)')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(f'{output_dir}/confusion_matrix_global.png')
            plt.close()
            
            # Add global metrics
            results['global'] = {
                'accuracy': global_acc,
                'confusion_matrix': global_cm,
                'classification_report': global_report
            }
            detailed_metrics['global'] = global_report
        
        # Create summary visualization
        accuracies = [res['accuracy'] for var, res in results.items() if var != 'global']
        plt.figure(figsize=(10, 5))
        plt.bar(x=list(var for var in results.keys() if var != 'global'), height=accuracies)
        plt.title('Imputation Accuracy by Variable')
        plt.xlabel('Variable')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/imputation_accuracy_summary.png')
        plt.close()
        
        return results, detailed_metrics

    def visualize_convergence(self, output_dir: str = "./output"):
        """
        Visualize the convergence of the EM algorithm by plotting:
        1. Log-likelihood trajectory
        2. Parameter changes over iterations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure all data arrays have the same length
        n_iterations = len(self.param_history)
        iterations = list(range(n_iterations))
        
        # Pad or truncate log-likelihoods if necessary
        log_likelihoods = self.log_likelihoods
        if len(log_likelihoods) > n_iterations:
            log_likelihoods = log_likelihoods[:n_iterations]
        elif len(log_likelihoods) < n_iterations:
            # Pad with the last value
            last_ll = log_likelihoods[-1] if log_likelihoods else 0
            log_likelihoods.extend([last_ll] * (n_iterations - len(log_likelihoods)))
        
        # Plot log-likelihood trajectory
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, log_likelihoods, 'b-', label='Log-Likelihood')
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.title('EM Algorithm Convergence')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'em_convergence.png'), bbox_inches='tight')
        plt.close()
        
        # Initialize convergence data with matched lengths
        convergence_data = {
            'iteration': iterations,
            'log_likelihood': log_likelihoods
        }
        
        if self.param_history:
            # Plot parameter trajectories
            plt.figure(figsize=(15, 8))
            
            # Track valid parameters for plotting
            valid_params = {}
            
            # First, collect valid parameter trajectories
            node_names = list(self.param_history[0].keys()) if self.param_history else []
            for node_name in node_names:
                param_values = []
                valid = True
                
                for params in self.param_history:
                    if node_name not in params or not params[node_name]:
                        self.logger.debug(f"Missing parameters for node {node_name} in history")
                        valid = False
                        break
                    try:
                        # Get all parameter values for this node
                        node_params = params[node_name]
                        # Use the first parameter value as representative
                        first_key = next(iter(node_params.keys()))
                        param_values.append(node_params[first_key])
                    except (StopIteration, KeyError) as e:
                        self.logger.debug(f"Error getting parameters for node {node_name}: {str(e)}")
                        valid = False
                        break
                
                if valid and param_values:
                    if len(param_values) == n_iterations:
                        valid_params[node_name] = param_values
                        plt.plot(iterations, param_values, '-', label=node_name)
                        # Add to convergence data
                        convergence_data[f'param_{node_name}'] = param_values
                    else:
                        self.logger.debug(
                            f"Parameter trajectory for {node_name} has {len(param_values)} values "
                            f"but expected {n_iterations}"
                        )
            
            if valid_params:
                plt.xlabel('Iteration')
                plt.ylabel('Parameter Value')
                plt.title('Parameter Trajectories')
                plt.grid(True)
                if len(valid_params) > 10:  # If too many parameters, adjust legend
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'parameter_trajectories.png'), bbox_inches='tight')
                plt.close()
            else:
                self.logger.warning("No valid parameter trajectories to plot")
        
        # Verify all arrays have the same length before creating DataFrame
        lengths = {k: len(v) for k, v in convergence_data.items()}
        if len(set(lengths.values())) > 1:
            self.logger.error(f"Inconsistent lengths in convergence data: {lengths}")
            # Filter out columns with wrong length
            convergence_data = {
                k: v for k, v in convergence_data.items() 
                if len(v) == n_iterations
            }
        
        # Create DataFrame only with valid columns
        df = pd.DataFrame(convergence_data)
        df.to_csv(os.path.join(output_dir, 'convergence_data.csv'), index=False)

    def _initialize_parameters(self) -> None:
        """Initialize parameters with better defaults"""
        self.logger.info("Initializing parameters...")
        for node_name in self.nodes:
            node = self.nodes[node_name]
            node.cpd = node.cpd.sort_index()
            temp_cpd = node.cpd.copy()

            # Initialize with slightly biased probabilities to break symmetry
            for parent_combo in product([0, 1], repeat=len(node.parent_nodes)):
                # Use different ranges for root and non-root nodes
                if not node.parent_nodes:
                    # Root nodes get more balanced probabilities
                    prob = random.uniform(0.4, 0.6)
                else:
                    # Non-root nodes get more varied probabilities
                    prob = random.uniform(0.2, 0.8)
                
                # Ensure numerical stability
                prob = max(min(prob, 1.0 - 1e-10), 1e-10)
                
                # Set probabilities
                temp_cpd.at[tuple([1] + list(parent_combo)), "prob"] = prob
                temp_cpd.at[tuple([0] + list(parent_combo)), "prob"] = 1 - prob

            node.cpd = normalize(temp_cpd)

    def _e_step(
        self,
        samples_with_missing: List[Dict[str, int]],
        batch_size: int,
        n_workers: int
    ) -> None:
        """
        E-step: Compute expected values for missing data using multiprocessing
        """
        try:
            # Map variable names in samples
            mapped_samples = []
            for sample in samples_with_missing:
                mapped_sample = {}
                for var, val in sample.items():
                    if var not in self.forward_name_mapping:
                        raise ValueError(f"Variable {var} not found in network mapping")
                    mapped_name = self.forward_name_mapping[var]
                    mapped_sample[mapped_name] = val
                mapped_samples.append(mapped_sample)
            
            # Convert samples to DataFrame for validation
            samples_df = pd.DataFrame(mapped_samples)
            
            # Ensure all values are integers (0, 1, or 2 for missing)
            for col in samples_df.columns:
                samples_df[col] = pd.to_numeric(samples_df[col], errors='coerce').fillna(2).astype(int)
                samples_df[col] = samples_df[col].clip(0, 2)
            
            # Process samples in batches using multiprocessing
            n_samples = len(samples_df)
            batch_indices = [(i, min(i + batch_size, n_samples)) 
                           for i in range(0, n_samples, batch_size)]
            
            # Prepare batches for parallel processing
            batch_data = []
            for start_idx, end_idx in batch_indices:
                batch_samples = samples_df.iloc[start_idx:end_idx].to_dict('records')
                batch_data.append((batch_samples, self.bn_file, {
                    node_name: node.cpd.copy() 
                    for node_name, node in self.nodes.items()
                }))
            
            # Process batches in parallel
            imputed_samples = []
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(self._impute_batch, data) for data in batch_data]
                
                # Collect results with progress bar
                for future in tqdm(futures, desc="E-step batches"):
                    batch_results = future.result()
                    imputed_samples.extend(list(batch_results.values()))
            
            # Convert results to DataFrame
            final_imputed = pd.DataFrame(imputed_samples)
            
            # Reverse the mapping for the final imputed samples
            final_imputed.rename(columns=self.reverse_name_mapping, inplace=True)
            
            # Store imputed samples
            if not hasattr(self, 'imputed_samples'):
                self.imputed_samples = []
            self.imputed_samples.append(final_imputed)
            
            # Keep only last few iterations to save memory
            if len(self.imputed_samples) > 5:
                self.imputed_samples = self.imputed_samples[-5:]
            
        except Exception as e:
            self.logger.error(f"Error in E-step: {str(e)}")
            raise

    def _impute_missing_values(self, sample: Dict[str, int]) -> Dict[str, int]:
        """
        Impute missing values for a single sample using forward sampling.
        
        Args:
            sample: Dictionary mapping variable names to values (2 for missing)
            
        Returns:
            Dictionary with imputed values for missing variables
        """
        try:
            imputed = sample.copy()
            
            # Get nodes with missing values
            missing_nodes = [
                node_name for node_name, value in sample.items() 
                if value == 2
            ]
            
            # If no missing values, return as is
            if not missing_nodes:
                return imputed
            
            # Impute each missing value using forward sampling
            for node_name in missing_nodes:
                node = self.nodes[node_name]
                
                # Get parent values
                parent_values = []
                skip_node = False
                for parent in node.parent_nodes:
                    parent_val = imputed[parent.var_name]
                    if parent_val == 2:  # Skip if parent is missing
                        skip_node = True
                        break
                    parent_values.append(parent_val)
                
                if skip_node:
                    continue
                
                # Get conditional probability
                if parent_values:
                    idx = tuple([1] + parent_values)
                    try:
                        prob = float(node.cpd.loc[idx, 'prob'])
                    except KeyError:
                        prob = 0.5  # Default if combination not found
                else:
                    idx = (1,)
                    try:
                        prob = float(node.cpd.loc[idx, 'prob'])
                    except KeyError:
                        prob = 0.5  # Default if no entry
                
                # Ensure probability is valid
                prob = max(min(prob, 1.0 - 1e-10), 1e-10)
                
                # Sample value based on probability
                imputed[node_name] = np.random.binomial(1, prob)
            
            return imputed
            
        except Exception as e:
            self.logger.error(f"Error imputing missing values: {str(e)}")
            raise

    def _impute_batch(self, samples: List[Dict[str, int]]) -> List[Dict[str, int]]:
        """Impute missing values for a batch of samples."""
        return [self._impute_missing_values(sample) for sample in samples]

    def _m_step(
        self,
        samples_with_missing: List[Dict[str, int]],
        imputed_samples: pd.DataFrame,
        alpha: float,
        batch_size: int,
        n_workers: int,
        momentum: float = 0.1
    ) -> Tuple[Dict[str, Dict[int, float]], float]:
        """
        M-step: Update parameters using imputed samples and momentum with multiprocessing
        """
        # Map variable names in samples
        mapped_samples = []
        for sample in samples_with_missing:
            mapped_sample = {}
            for var, val in sample.items():
                mapped_name = next(k for k, v in self.reverse_name_mapping.items() if v == var)
                mapped_sample[mapped_name] = val
            mapped_samples.append(mapped_sample)
        
        # Convert samples to DataFrame if needed
        if isinstance(imputed_samples, list):
            imputed_samples = pd.DataFrame(mapped_samples)
        else:
            column_mapping = {v: k for k, v in self.reverse_name_mapping.items()}
            imputed_samples = imputed_samples.rename(columns=column_mapping)
        
        # Ensure all values are integers
        for col in imputed_samples.columns:
            imputed_samples[col] = pd.to_numeric(imputed_samples[col], errors='coerce').fillna(2).astype(int)
            imputed_samples[col] = imputed_samples[col].clip(0, 2)
        
        # Process nodes in parallel
        new_params = {}
        max_param_diff = 0.0
        
        # Prepare node data for parallel processing
        node_data = []
        for node_name, node in self.nodes.items():
            node_data.append((
                node_name,
                imputed_samples[node_name].values,
                pd.DataFrame([
                    imputed_samples[parent.var_name].values 
                    for parent in node.parent_nodes
                ]).T.values if node.parent_nodes else np.empty((len(imputed_samples), 0)),
                node.cpd.index,
                alpha
            ))
        
        # Process nodes in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(self._process_node_counts_stable, *data) 
                for data in node_data
            ]
            
            # Collect results with progress bar
            for future, (node_name, _, _, _, _) in tqdm(
                zip(futures, node_data), 
                desc="M-step nodes", 
                total=len(node_data)
            ):
                params, _ = future.result()
                
                # Get current parameters
                node = self.nodes[node_name]
                current_params = {
                    idx: float(prob) 
                    for idx, prob in zip(node.cpd.index, node.cpd['prob'])
                }
                
                # Apply momentum and update parameters
                updated_params = {}
                for idx, new_prob in params.items():
                    old_prob = current_params[idx]
                    updated_prob = old_prob + momentum * (new_prob - old_prob)
                    updated_prob = max(min(updated_prob, 1.0 - 1e-10), 1e-10)
                    updated_params[idx] = updated_prob
                    max_param_diff = max(max_param_diff, abs(updated_prob - old_prob))
                    
                    # Update CPD
                    node.cpd.at[idx, 'prob'] = updated_prob
                
                # Store parameters with original variable names
                original_name = self.reverse_name_mapping[node_name]
                new_params[original_name] = updated_params
        
        return new_params, max_param_diff

    def _process_node_counts_stable(
        self,
        node_name: str,
        values: np.ndarray,
        parent_values: np.ndarray,
        cpd_index: pd.Index,
        alpha: float,
        epsilon: float = 1e-10
    ) -> Tuple[Dict[tuple, float], Dict[tuple, float]]:
        """
        Process counts with improved numerical stability
        """
        try:
            counts = {idx: 0.0 for idx in cpd_index}
            params = {}
            
            # Handle no-parent case with stability
            if parent_values.shape[1] == 0:
                val_counts = np.bincount(values, minlength=2)
                
                # Apply Laplace smoothing and compute probability
                smoothed_counts = val_counts + alpha
                total = np.sum(smoothed_counts) + epsilon
                
                # Store results for both values
                counts[(1,)] = val_counts[1]
                counts[(0,)] = val_counts[0]
                
                params[(1,)] = min(max(smoothed_counts[1] / total, epsilon), 1.0 - epsilon)
                params[(0,)] = 1.0 - params[(1,)]
                
                return params, counts
            
            parent_configs = np.array([list(idx[1:]) for idx in cpd_index if idx[0] == 1])
            
            for parent_config in parent_configs:
                # Create mask for matching parent configuration
                mask = np.all(parent_values == parent_config, axis=1)
                
                # Skip if no matching samples
                if not np.any(mask):
                    continue
                
                matching_values = values[mask]
                val_counts = np.bincount(matching_values, minlength=2)
                
                # Apply Laplace smoothing and compute probability
                smoothed_counts = val_counts + alpha
                total = np.sum(smoothed_counts) + epsilon
                
                # Store results
                idx_1 = tuple([1] + list(parent_config))
                idx_0 = tuple([0] + list(parent_config))
                
                counts[idx_1] = val_counts[1]
                counts[idx_0] = val_counts[0]
                
                # Compute probabilities with numerical stability
                prob_1 = min(max(smoothed_counts[1] / total, epsilon), 1.0 - epsilon)
                params[idx_1] = prob_1
                params[idx_0] = 1.0 - prob_1
            
            return params, counts
            
        except Exception as e:
            self.logger.error(f"Error processing counts for node {node_name}: {str(e)}")
            raise

    def learn_cpds(
        self,
        samples_with_missing: List[Dict[str, int]],
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        alpha: float = 0.1,
        batch_size: int = 32,
        n_workers: int = 4,
        momentum: float = 0.1,
        patience: int = 3,
        min_iterations: int = 5
    ) -> None:
        """
        Learn conditional probability distributions using EM algorithm with momentum
        """
        self.logger.info("Starting EM algorithm...")
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Initialize storage for parameter history and log-likelihoods
        self.param_history = []
        self.log_likelihoods = []
        
        # Store initial parameters
        initial_params = {}
        for node_name, node in self.nodes.items():
            # Map node name back to original name
            original_name = self.reverse_name_mapping.get(node_name, node_name)
            initial_params[original_name] = {
                idx: float(prob.iloc[0]) if isinstance(prob, pd.Series) else float(prob)
                for idx, prob in zip(node.cpd.index, node.cpd['prob'])
            }
        self.param_history.append(initial_params)
        
        # Initialize tracking variables
        prev_ll = float('-inf')
        best_ll = float('-inf')
        best_params = None
        no_improvement_count = 0
        oscillation_count = 0
        prev_direction = 0
        
        # Compute initial log-likelihood
        initial_ll = self.compute_log_likelihood(samples_with_missing)
        self.log_likelihoods.append(initial_ll)
        best_ll = initial_ll
        best_params = initial_params
        
        # Main EM loop
        for iteration in tqdm(range(max_iterations), desc="EM Iterations"):
            try:
                # E-step: Compute expected values
                self._e_step(samples_with_missing, batch_size, n_workers)
                
                # M-step: Update parameters with momentum
                new_params, param_diff = self._m_step(
                    samples_with_missing, 
                    self.imputed_samples[-1], 
                    alpha,
                    batch_size,
                    n_workers,
                    momentum=momentum
                )
                
                # Store updated parameters
                self.param_history.append(new_params)
                
                # Compute log-likelihood using original samples
                current_ll = self.compute_log_likelihood(samples_with_missing)
                self.log_likelihoods.append(current_ll)
                
                # Check convergence
                ll_diff = current_ll - prev_ll if prev_ll != float('-inf') else float('inf')
                abs_diff = abs(ll_diff)
                
                # Update progress bar with more precision for small differences
                tqdm.write(f"\rIteration {iteration + 1}/{max_iterations}, ll={current_ll:.6f}, diff={ll_diff:.10f}")
                
                # Detect oscillation
                if iteration > 0:
                    current_direction = 1 if ll_diff > 0 else (-1 if ll_diff < 0 else 0)
                    if current_direction * prev_direction < 0:  # Direction changed
                        oscillation_count += 1
                        # Reduce momentum more aggressively when oscillating
                        momentum = max(0.01, momentum * 0.5)
                    else:
                        oscillation_count = 0
                    prev_direction = current_direction
                
                # Check for improvement
                if current_ll > best_ll:  # Remove threshold to be more sensitive
                    best_ll = current_ll
                    best_params = new_params
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    # Reduce momentum when not improving
                    momentum = max(0.01, momentum * 0.9)
                
                # Early stopping checks
                if iteration >= min_iterations:
                    # Stop if no improvement for several iterations
                    if no_improvement_count >= patience:
                        self.logger.info(f"Early stopping after {iteration + 1} iterations (no improvement for {patience} iterations)")
                        break
                    
                    # Stop if oscillating too much
                    if oscillation_count >= 3:
                        self.logger.info(f"Early stopping after {iteration + 1} iterations (detected oscillation)")
                        break
                    
                    # Stop if converged
                    if prev_ll != float('-inf'):
                        relative_diff = abs_diff / (abs(prev_ll) + 1e-10)
                        if relative_diff < convergence_threshold:
                            self.logger.info(f"Converged after {iteration + 1} iterations (relative diff: {relative_diff:.10f})")
                            break
                
                prev_ll = current_ll
                
            except Exception as e:
                self.logger.error(f"EM algorithm failed: {str(e)}")
                raise
        
        # Restore best parameters
        if best_params is not None:
            for original_name, params in best_params.items():
                if params:  # Only update if we have parameters
                    # Map original name back to node name
                    node_name = self.forward_name_mapping.get(original_name, original_name)
                    if node_name in self.nodes:
                        node = self.nodes[node_name]
                        for idx, prob in params.items():
                            prob = max(min(prob, 1.0 - 1e-10), 1e-10)
                            node.cpd.at[idx, "prob"] = prob
        
        self.logger.info("EM algorithm completed")

    def compute_log_likelihood(self, samples: List[Dict[str, int]]) -> float:
        """
        Optimized log-likelihood computation using vectorization where possible.
        Returns the average log-likelihood per sample.
        """
        total_ll = 0.0
        n_samples = len(samples)
        epsilon = 1e-10  # Small constant to prevent log(0)
        
        try:
            for sample in samples:
                sample_ll = 0.0
                for node_name, node in self.nodes.items():
                    # Get the value for this node and its parents
                    value = sample.get(node_name)
                    if value is None or value == 2:  # Skip missing values
                        continue
                        
                    # Get parent values
                    parent_values = []
                    skip_node = False
                    for parent in node.parent_nodes:
                        parent_val = sample.get(parent.var_name)
                        if parent_val is None or parent_val == 2:  # Skip if any parent has missing value
                            skip_node = True
                            break
                        parent_values.append(parent_val)
                    
                    if skip_node:
                        continue
                    
                    # Create index tuple for CPD lookup
                    idx = tuple([value] + parent_values)
                    
                    # Get probability and compute log
                    prob = node.cpd.loc[idx, 'prob']
                    if prob < epsilon:
                        prob = epsilon
                    elif prob > 1.0 - epsilon:
                        prob = 1.0 - epsilon
                        
                    sample_ll += np.log(prob)
                
                if not np.isneginf(sample_ll):  # Only add if not -inf
                    total_ll += sample_ll
            
            # Return average log-likelihood, with safeguard against all -inf
            if n_samples > 0:
                avg_ll = total_ll / n_samples
                if np.isfinite(avg_ll):
                    return avg_ll
                return -1e6  # Return a large negative number instead of -inf
            return -1e6
            
        except Exception as e:
            self.logger.error(f"Error computing log-likelihood: {str(e)}")
            return -1e6  # Return a large negative number on error

    def conditional_log_prob(self, var: str, evidence: Dict[str, int], value: int = None) -> float:
        """
        Compute conditional log probability of a variable given evidence.
        
        Args:
            var: Variable name to compute probability for
            evidence: Dictionary mapping variable names to their values
            value: Value to compute probability for (if None, returns probabilities for all values)
        
        Returns:
            float: Log probability of the variable taking the specified value given the evidence
        """
        # Create junction tree for efficient inference
        junction_tree = JunctionTree(bn=self)
        
        # Get initial junction tree
        jt = junction_tree._get_junction_tree()
        
        # Incorporate evidence
        jt = junction_tree._incorporate_evidence(jt, evidence)
        
        # Run belief propagation
        calibrated_jt = junction_tree._run_belief_propagation(jt)
        
        # Find the smallest clique containing the variable
        containing_cliques = [
            n for n in calibrated_jt.nodes()
            if var in calibrated_jt.nodes[n]["factor_vars"]
        ]
        best_clique = min(
            containing_cliques,
            key=lambda n: len(calibrated_jt.nodes[n]["factor_vars"])
        )
        
        # Get potential and marginalize to the target variable
        potential = calibrated_jt.nodes[best_clique]["potential"]
        for v in [v for v in potential.index.names if v != var]:
            potential = sumout(potential, [v])
        potential = normalize(potential)
        
        # Return log probability for specified value or all probabilities
        if value is not None:
            try:
                return np.log(potential.loc[value]["prob"])
            except KeyError:
                return float("-inf")
        else:
            return {val: np.log(prob) for val, prob in 
                    zip(potential.index, potential["prob"])}

    def _impute_batch(self, batch_data: Tuple[List[Dict[str, int]], str, Dict]) -> Dict[tuple, Dict[str, int]]:
        """
        Process a batch of samples for imputation using Gibbs sampling.
        
        Args:
            batch_data: Tuple containing (batch_samples, bn_file, current_cpds)
            
        Returns:
            Dictionary mapping original samples to their imputed versions
        """
        batch_samples, bn_file, current_cpds = batch_data
        
        # Create a local BayesNet instance with current CPDs
        local_bn = BayesNet(bn_file)
        for node_name, cpd in current_cpds.items():
            local_bn.nodes[node_name].cpd = cpd.copy()
        
        # Create junction tree once per batch
        junction_tree = JunctionTree(bn=local_bn)
        
        batch_results = {}
        
        # Process each sample in the batch
        for sample in batch_samples:
            sample_key = tuple(sample.items())
            if sample_key in self._jt_cache:
                batch_results[sample_key] = self._jt_cache[sample_key]
                continue
                
            missing_vars = [var for var, val in sample.items() if val == 2]
            if not missing_vars:
                batch_results[sample_key] = sample.copy()
                continue
                
            # Incorporate evidence from non-missing variables
            evidence = {var: val for var, val in sample.items() if val != 2}
            jt = junction_tree._get_junction_tree()
            jt = junction_tree._incorporate_evidence(jt, evidence)
            calibrated_jt = junction_tree._run_belief_propagation(jt)
            
            # Impute missing values
            imputed_sample = sample.copy()
            for var in missing_vars:
                # Find cliques containing the variable
                containing_cliques = [
                    n for n in calibrated_jt.nodes()
                    if var in calibrated_jt.nodes[n]["factor_vars"]
                ]
                
                if not containing_cliques:
                    # If no clique contains the variable, create a new factor for it
                    # This can happen if the variable is disconnected in the graph
                    p1 = 0.5  # Use uniform distribution as fallback
                    imputed_sample[var] = np.random.choice([0, 1], p=[1-p1, p1])
                    continue
                
                # Find smallest clique containing the variable
                best_clique = min(
                    containing_cliques,
                    key=lambda n: len(calibrated_jt.nodes[n]["factor_vars"])
                )
                
                # Get marginal probability
                potential = calibrated_jt.nodes[best_clique]["potential"]
                for v in [v for v in potential.index.names if v != var]:
                    potential = sumout(potential, [v])
                potential = normalize(potential)
                
                # Convert potential to probabilities
                try:
                    # Try direct access using iloc for single element Series
                    p0 = float(potential.loc[0].iloc[0] if 0 in potential.index else 0)
                    p1 = float(potential.loc[1].iloc[0] if 1 in potential.index else 0)
                except:
                    # If direct access fails, try getting values as array
                    try:
                        values = potential.values.flatten()
                        if len(values) >= 2:
                            p0, p1 = values[0], values[1]
                        else:
                            p0 = values[0] if len(values) > 0 else 0.5
                            p1 = 1 - p0
                    except:
                        # Fallback to uniform distribution
                        p0 = p1 = 0.5
                
                # Ensure valid probabilities
                if p0 + p1 == 0:
                    p0 = p1 = 0.5
                else:
                    # Normalize
                    total = p0 + p1
                    p0 /= total
                    p1 /= total
                
                # Sample value based on probabilities
                imputed_sample[var] = np.random.choice([0, 1], p=[p0, p1])
            
            # Cache and store results
            self._jt_cache[sample_key] = imputed_sample
            batch_results[sample_key] = imputed_sample
        
        return batch_results

    def _process_node_counts(
        self,
        node_name: str,
        values: np.ndarray,
        parent_values: np.ndarray,
        cpd_index: pd.Index,
        alpha: float
    ) -> Tuple[Dict[tuple, float], Dict[tuple, float]]:
        """
        Process counts for a single node using vectorized operations
        """
        # Initialize storage
        counts = {idx: 0.0 for idx in cpd_index}
        params = {}
        
        # Handle no-parent case
        if parent_values.shape[1] == 0:
            # Count occurrences of each value
            val_counts = np.bincount(values, minlength=2)
            
            # Apply Laplace smoothing and compute probability
            smoothed_counts = val_counts + alpha
            total = np.sum(smoothed_counts)
            
            # Store results for both values
            counts[(1,)] = val_counts[1]
            counts[(0,)] = val_counts[0]
            params[(1,)] = smoothed_counts[1] / total
            params[(0,)] = smoothed_counts[0] / total
            
            return params, counts
        
        # Get all possible parent configurations from CPD index
        parent_configs = np.array([list(idx[1:]) for idx in cpd_index if idx[0] == 1])
        
        # Process each parent configuration
        for parent_config in parent_configs:
            # Find samples matching this parent configuration
            mask = np.all(parent_values == parent_config, axis=1)
            if not np.any(mask):
                # If no samples match this configuration, use uniform distribution
                idx_1 = tuple([1] + list(parent_config))
                idx_0 = tuple([0] + list(parent_config))
                counts[idx_1] = 0
                counts[idx_0] = 0
                params[idx_1] = 0.5
                params[idx_0] = 0.5
                continue
            
            # Count occurrences of each value for this configuration
            matching_values = values[mask]
            val_counts = np.bincount(matching_values, minlength=2)
            
            # Apply Laplace smoothing and compute probability
            smoothed_counts = val_counts + alpha
            total = np.sum(smoothed_counts)
            
            # Store results
            idx_1 = tuple([1] + list(parent_config))
            idx_0 = tuple([0] + list(parent_config))
            
            counts[idx_1] = val_counts[1]
            counts[idx_0] = val_counts[0]
            
            params[idx_1] = smoothed_counts[1] / total
            params[idx_0] = smoothed_counts[0] / total
        
        return params, counts

def download_air_quality_dataset(output_dir: str = './data') -> str:
    """
    Download and properly convert the Air Quality dataset from UCI repository.
    
    Args:
        output_dir: Directory to save the dataset
        
    Returns:
        str: Path to the processed CSV file
    """
    csv_path = os.path.join(output_dir, "air_quality.csv")
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(csv_path):
        print(f"Air Quality dataset already exists at {csv_path}")
        return csv_path
        
    print("Downloading Air Quality dataset...")
    url = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Extract all files
            zip_ref.extractall(output_dir)
            
            # Find the data file
            data_file = os.path.join(output_dir, 'AirQualityUCI.csv')
            
            if not os.path.exists(data_file):
                raise FileNotFoundError("Could not find AirQualityUCI.csv in the zip archive")
            
            # Read and process the data properly
            with open(data_file, 'r') as f:
                lines = f.readlines()
            
            # Process and write to new CSV
            with open(csv_path, 'w', newline='') as f:
                # Write header (first line contains column names)
                header = lines[0].strip().replace(';', ',')
                f.write(header + '\n')
                
                # Process and write each data line
                for line in lines[1:]:
                    # Split by semicolon and process each field
                    fields = line.strip().split(';')
                    processed_fields = []
                    
                    for field in fields:
                        field = field.strip()
                        # Handle empty or invalid fields
                        if not field or field == '':
                            processed_fields.append('')
                        else:
                            # Convert decimal comma to point for numeric fields
                            try:
                                # Try to convert to float (handles both . and , as decimal)
                                value = float(field.replace(',', '.'))
                                if value == -200:
                                    processed_fields.append('')  # Convert -200 to empty string
                                else:
                                    processed_fields.append(str(value))
                            except ValueError:
                                # If not a number, keep as is
                                processed_fields.append(field)
                    
                    # Join fields with comma and write line
                    f.write(','.join(processed_fields) + '\n')
            
            # Clean up original file
            os.remove(data_file)
            
        print(f"Successfully downloaded and processed dataset to {csv_path}")
        return csv_path
        
    except Exception as e:
        print(f"Error processing the dataset: {e}")
        raise

def load_air_quality_data(file_path: str) -> pd.DataFrame:
    """
    Load the Air Quality dataset with proper column names.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    # Define column names
    columns = [
        'Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 
        'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 
        'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'
    ]
    
    # Read and preprocess the CSV file to remove trailing commas
    with open(file_path, 'r') as f:
        lines = [line.rstrip(',\n') + '\n' for line in f]
    
    # Use StringIO to create an in-memory file-like object
    from io import StringIO
    csv_data = StringIO(''.join(lines))
    
    # Read the preprocessed CSV data
    df = pd.read_csv(csv_data, 
                    na_values=['', 'NA', '-200'],
                    names=columns,  # Set column names
                    header=0)  # First row is header, skip it
    
    # Rename columns to match expected names
    df.columns = columns
    
    # Drop Date and Time columns
    df = df.drop(['Date', 'Time'], axis=1)
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns:", df.columns.tolist())
    
    print("\nMissing values per column:")
    for col in df.columns:
        missing_count = df[col].isna().sum()
        percentage = (missing_count / len(df)) * 100
        print(f"{col}: {missing_count:,} missing values ({percentage:.2f}%)")
    
    return df

def process_air_quality_data(file_path: str) -> List[Dict[str, int]]:
    """
    Process the Air Quality dataset for EM algorithm.
    
    Args:
        file_path: Path to the Air Quality dataset
        
    Returns:
        samples: List of dictionaries with binary encoded data (0, 1) and missing values (2)
    """
    # Read the data with proper column names
    df = load_air_quality_data(file_path)
    
    # Convert to numeric and handle missing values
    df_processed = df.copy()
    for col in df.columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Replace -200 with NaN
    df_processed = df_processed.replace(-200, np.nan)
    
    # Calculate percentage of missing values for each column
    missing_percentages = (df_processed.isna().sum() / len(df_processed)) * 100
    print("\nMissing values percentage per column:")
    for col, pct in missing_percentages.items():
        print(f"{col}: {pct:.2f}%")
    
    # For each sensor type, determine the binarization threshold based on domain knowledge
    thresholds = {
        'CO(GT)': 2.0,        # CO levels above 2.0 mg/m³ are concerning
        'PT08.S1(CO)': None,  # Will use median for sensor readings
        'NMHC(GT)': 500,      # NMHC levels above 500 µg/m³ are high
        'C6H6(GT)': 5.0,      # Benzene levels above 5.0 µg/m³ are concerning
        'PT08.S2(NMHC)': None,
        'NOx(GT)': 100,       # NOx levels above 100 ppb are high
        'PT08.S3(NOx)': None,
        'NO2(GT)': 40,        # NO2 levels above 40 µg/m³ are concerning (WHO guideline)
        'PT08.S4(NO2)': None,
        'PT08.S5(O3)': None,
        'T': 25,              # Temperature above 25°C is considered warm
        'RH': 60,             # RH above 60% is considered humid
        'AH': None           # Will use median for absolute humidity
    }
    
    # Create binary data using thresholds or medians
    binary_data = pd.DataFrame(index=df_processed.index, columns=df_processed.columns)
    
    for col in df_processed.columns:
        threshold = thresholds[col]
        if threshold is None:
            # Use median for sensor readings and undefined thresholds
            threshold = df_processed[col].median()
        
        # Convert to binary based on threshold
        binary_data[col] = (df_processed[col] > threshold).astype(int)
        
        # Mark missing values as 2 using proper loc accessor
        binary_data.loc[df_processed[col].isna(), col] = 2
        
        # Print threshold used
        print(f"{col} threshold: {threshold:.2f}")
    
    # Convert to list of dictionaries
    samples = binary_data.to_dict('records')
    
    return samples

def analyze_domain_knowledge(original_df: pd.DataFrame, imputed_df: pd.DataFrame, output_dir: str):
    """
    Analyze imputed values against domain knowledge for air quality data.
    
    Args:
        original_df: Original dataframe with missing values
        imputed_df: Dataframe with imputed values
        output_dir: Directory to save analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Known Chemical Relationships Plot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: NOx vs NO2 relationship (NOx should be higher than NO2)
    plt.subplot(2, 2, 1)
    valid_mask = ~pd.isna(original_df['NOx(GT)']) & ~pd.isna(original_df['NO2(GT)'])
    plt.scatter(original_df.loc[valid_mask, 'NOx(GT)'], 
               original_df.loc[valid_mask, 'NO2(GT)'],
               alpha=0.5, label='Original', color='blue')
    
    # Add imputed points
    missing_mask = pd.isna(original_df['NOx(GT)']) | pd.isna(original_df['NO2(GT)'])
    plt.scatter(imputed_df.loc[missing_mask, 'NOx(GT)'],
               imputed_df.loc[missing_mask, 'NO2(GT)'],
               alpha=0.5, label='Imputed', color='red')
    
    plt.plot([0, plt.gca().get_xlim()[1]], [0, plt.gca().get_xlim()[1]], 
             'k--', label='1:1 Line')
    plt.xlabel('NOx (GT)')
    plt.ylabel('NO2 (GT)')
    plt.title('NOx vs NO2 Relationship\n(NOx should be ≥ NO2)')
    plt.legend()
    
    # Plot 2: Temperature vs Relative Humidity (inverse relationship)
    plt.subplot(2, 2, 2)
    valid_mask = ~pd.isna(original_df['T']) & ~pd.isna(original_df['RH'])
    plt.scatter(original_df.loc[valid_mask, 'T'],
               original_df.loc[valid_mask, 'RH'],
               alpha=0.5, label='Original', color='blue')
    
    missing_mask = pd.isna(original_df['T']) | pd.isna(original_df['RH'])
    plt.scatter(imputed_df.loc[missing_mask, 'T'],
               imputed_df.loc[missing_mask, 'RH'],
               alpha=0.5, label='Imputed', color='red')
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Relative Humidity (%)')
    plt.title('Temperature vs RH Relationship\n(Expected Inverse Relationship)')
    plt.legend()
    
    # Plot 3: CO vs NOx (positive correlation in urban areas)
    plt.subplot(2, 2, 3)
    valid_mask = ~pd.isna(original_df['CO(GT)']) & ~pd.isna(original_df['NOx(GT)'])
    plt.scatter(original_df.loc[valid_mask, 'CO(GT)'],
               original_df.loc[valid_mask, 'NOx(GT)'],
               alpha=0.5, label='Original', color='blue')
    
    missing_mask = pd.isna(original_df['CO(GT)']) | pd.isna(original_df['NOx(GT)'])
    plt.scatter(imputed_df.loc[missing_mask, 'CO(GT)'],
               imputed_df.loc[missing_mask, 'NOx(GT)'],
               alpha=0.5, label='Imputed', color='red')
    
    plt.xlabel('CO (GT)')
    plt.ylabel('NOx (GT)')
    plt.title('CO vs NOx Relationship\n(Expected Positive Correlation)')
    plt.legend()
    
    # Plot 4: Sensor vs Ground Truth correlation
    plt.subplot(2, 2, 4)
    valid_mask = ~pd.isna(original_df['CO(GT)']) & ~pd.isna(original_df['PT08.S1(CO)'])
    plt.scatter(original_df.loc[valid_mask, 'CO(GT)'],
               original_df.loc[valid_mask, 'PT08.S1(CO)'],
               alpha=0.5, label='Original', color='blue')
    
    missing_mask = pd.isna(original_df['CO(GT)']) | pd.isna(original_df['PT08.S1(CO)'])
    plt.scatter(imputed_df.loc[missing_mask, 'CO(GT)'],
               imputed_df.loc[missing_mask, 'PT08.S1(CO)'],
               alpha=0.5, label='Imputed', color='red')
    
    plt.xlabel('CO Ground Truth')
    plt.ylabel('CO Sensor Reading')
    plt.title('Sensor vs Ground Truth\n(Expected Strong Correlation)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'domain_knowledge_analysis.png'))
    plt.close()
    
    # 2. Correlation Analysis
    # Original data correlation (excluding missing values)
    original_corr = original_df.corr()
    imputed_corr = imputed_df.corr()
    
    # Plot correlation matrices
    plt.figure(figsize=(20, 8))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(original_corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title('Original Data Correlation Matrix')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(imputed_corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title('Imputed Data Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'))
    plt.close()
    
    # Save correlation differences
    correlation_diff = imputed_corr - original_corr
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_diff, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title('Correlation Difference (Imputed - Original)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_difference.png'))
    plt.close()
    
    # Save correlation analysis to text file
    with open(os.path.join(output_dir, 'correlation_analysis.txt'), 'w') as f:
        f.write("=== Correlation Analysis ===\n\n")
        f.write("Notable correlation changes after imputation:\n")
        
        # Find significant correlation changes
        significant_changes = []
        for i in range(len(correlation_diff.index)):
            for j in range(i+1, len(correlation_diff.columns)):
                diff = correlation_diff.iloc[i, j]
                if abs(diff) >= 0.1:  # Report changes >= 0.1
                    var1 = correlation_diff.index[i]
                    var2 = correlation_diff.columns[j]
                    original = original_corr.iloc[i, j]
                    imputed = imputed_corr.iloc[i, j]
                    significant_changes.append((var1, var2, original, imputed, diff))
        
        # Sort by absolute difference
        significant_changes.sort(key=lambda x: abs(x[4]), reverse=True)
        
        for var1, var2, orig, imp, diff in significant_changes:
            f.write(f"\n{var1} vs {var2}:\n")
            f.write(f"  Original correlation: {orig:.3f}\n")
            f.write(f"  Imputed correlation:  {imp:.3f}\n")
            f.write(f"  Change:              {diff:+.3f}\n")

def convert_binary_to_continuous(binary_df: pd.DataFrame, thresholds: dict, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert binary imputed values back to continuous values using thresholds and original data statistics.
    
    Args:
        binary_df: DataFrame with binary imputed values (0/1)
        thresholds: Dictionary of thresholds used for binarization
        original_df: Original DataFrame with continuous values
        
    Returns:
        DataFrame with continuous values
    """
    continuous_df = pd.DataFrame(index=binary_df.index, columns=binary_df.columns)
    
    for col in binary_df.columns:
        # Get threshold for this variable
        threshold = thresholds[col]
        if threshold is None:
            threshold = original_df[col].median()
        
        # Get statistics for values above and below threshold
        below_threshold = original_df[col][original_df[col] <= threshold]
        above_threshold = original_df[col][original_df[col] > threshold]
        
        # Calculate representative values
        below_value = below_threshold.median() if not below_threshold.empty else threshold * 0.75
        above_value = above_threshold.median() if not above_threshold.empty else threshold * 1.25
        
        # Convert binary values to continuous
        continuous_df[col] = np.where(binary_df[col] == 0, below_value, above_value)
    
    return continuous_df

def analyze_air_quality_imputation(
    original_df: pd.DataFrame,
    imputed_samples: pd.DataFrame,
    output_dir: str
):
    """Analyze the imputation results for the air quality dataset"""
    try:
        if imputed_samples.empty:
            raise ValueError("No imputed samples available")
            
        # Create comparison plots for each variable
        os.makedirs(output_dir, exist_ok=True)
        
        # Define thresholds (same as in process_air_quality_data)
        thresholds = {
            'CO(GT)': 2.0,        # CO levels above 2.0 mg/m³ are concerning
            'PT08.S1(CO)': None,  # Will use median for sensor readings
            'NMHC(GT)': 500,      # NMHC levels above 500 µg/m³ are high
            'C6H6(GT)': 5.0,      # Benzene levels above 5.0 µg/m³ are concerning
            'PT08.S2(NMHC)': None,
            'NOx(GT)': 100,       # NOx levels above 100 ppb are high
            'PT08.S3(NOx)': None,
            'NO2(GT)': 40,        # NO2 levels above 40 µg/m³ are concerning (WHO guideline)
            'PT08.S4(NO2)': None,
            'PT08.S5(O3)': None,
            'T': 25,              # Temperature above 25°C is considered warm
            'RH': 60,             # RH above 60% is considered humid
            'AH': None           # Will use median for absolute humidity
        }
        
        # Convert binary imputed values to continuous
        continuous_imputed = convert_binary_to_continuous(imputed_samples, thresholds, original_df)
        
        # For each variable, create plots comparing original and imputed distributions
        for column in original_df.columns:
            if column in continuous_imputed.columns:
                plt.figure(figsize=(12, 6))
                
                # Get original and imputed values
                original_values = original_df[column].values
                imputed_values = continuous_imputed[column].values
                binary_imputed = imputed_samples[column].values
                
                # Create masks for valid values
                valid_mask = ~pd.isna(original_values) & (original_values != -200)
                valid_imputed = ~pd.isna(imputed_values)
                
                if valid_mask.any() and valid_imputed.any():
                    # Plot 1: Distribution comparison
                    plt.subplot(2, 1, 1)
                    
                    # Plot original distribution
                    plt.hist(original_values[valid_mask], bins=30, alpha=0.5, 
                           label='Original', color='blue', density=True)
                    
                    # Plot imputed distribution
                    plt.hist(imputed_values[valid_imputed], bins=30, alpha=0.5,
                           label='Imputed', color='red', density=True)
                    
                    # Add threshold line
                    threshold = thresholds[column]
                    if threshold is None:
                        threshold = np.median(original_values[valid_mask])
                    plt.axvline(x=threshold, color='k', linestyle='--',
                              label=f'Threshold ({threshold:.2f})')
                    
                    plt.title(f'Value Distribution - {column}')
                    plt.xlabel('Value')
                    plt.ylabel('Density')
                    plt.legend()
                    
                    # Plot 2: Statistics
                    plt.subplot(2, 1, 2)
                    stats = {
                        'Count': [np.sum(binary_imputed == 0), np.sum(binary_imputed == 1)],
                        'Mean': [np.mean(imputed_values[binary_imputed == 0]),
                                np.mean(imputed_values[binary_imputed == 1])],
                        'Median': [np.median(imputed_values[binary_imputed == 0]),
                                 np.median(imputed_values[binary_imputed == 1])],
                        'Std': [np.std(imputed_values[binary_imputed == 0]),
                               np.std(imputed_values[binary_imputed == 1])]
                    }
                    
                    # Calculate percentages
                    total = len(binary_imputed)
                    pct_0 = np.sum(binary_imputed == 0) / total * 100
                    pct_1 = np.sum(binary_imputed == 1) / total * 100
                    
                    # Create text summary
                    plt.text(0.1, 0.8,
                           f'Below Threshold ({pct_0:.1f}%):\n' +
                           f'Count: {stats["Count"][0]}\n' +
                           f'Mean: {stats["Mean"][0]:.2f}\n' +
                           f'Median: {stats["Median"][0]:.2f}\n' +
                           f'Std: {stats["Std"][0]:.2f}',
                           transform=plt.gca().transAxes,
                           bbox=dict(facecolor='white', alpha=0.8))
                    
                    plt.text(0.6, 0.8,
                           f'Above Threshold ({pct_1:.1f}%):\n' +
                           f'Count: {stats["Count"][1]}\n' +
                           f'Mean: {stats["Mean"][1]:.2f}\n' +
                           f'Median: {stats["Median"][1]:.2f}\n' +
                           f'Std: {stats["Std"][1]:.2f}',
                           transform=plt.gca().transAxes,
                           bbox=dict(facecolor='white', alpha=0.8))
                    
                    plt.axis('off')
                    
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'dist_comparison_{column}.png'))
                plt.close()
        
        # Save overall statistics
        stats_df = pd.DataFrame({
            'variable': [],
            'threshold': [],
            'below_threshold_mean': [],
            'above_threshold_mean': [],
            'below_threshold_pct': [],
            'above_threshold_pct': [],
            'total_missing': []
        })
        
        for column in original_df.columns:
            if column in continuous_imputed.columns:
                original_values = original_df[column].values
                imputed_values = continuous_imputed[column].values
                binary_imputed = imputed_samples[column].values
                
                valid_mask = ~pd.isna(original_values) & (original_values != -200)
                
                threshold = thresholds[column]
                if threshold is None:
                    threshold = np.median(original_values[valid_mask])
                
                stats_df = pd.concat([stats_df, pd.DataFrame({
                    'variable': [column],
                    'threshold': [threshold],
                    'below_threshold_mean': [np.mean(imputed_values[binary_imputed == 0])],
                    'above_threshold_mean': [np.mean(imputed_values[binary_imputed == 1])],
                    'below_threshold_pct': [np.sum(binary_imputed == 0) / len(binary_imputed) * 100],
                    'above_threshold_pct': [np.sum(binary_imputed == 1) / len(binary_imputed) * 100],
                    'total_missing': [np.sum(~valid_mask)]
                })])
        
        stats_df.to_csv(os.path.join(output_dir, 'imputation_statistics.csv'), index=False)
        
        # Add domain knowledge analysis with continuous values
        analyze_domain_knowledge(original_df, continuous_imputed, output_dir)
        
    except Exception as e:
        print(f"Error analyzing imputation results: {str(e)}")
        raise

def get_args() -> Namespace:
    """Get command line arguments."""
    parser = argparse.ArgumentParser(description='Run EM algorithm on a Bayesian network.')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='synthetic',
                      choices=['synthetic', 'air_quality'],
                      help='Dataset to use (synthetic or air_quality)')
    
    # Files and directories
    parser.add_argument('--samples_file', type=str, default='samples.txt',
                      help='File containing samples with missing values')
    parser.add_argument('--true_samples_file', type=str, default=None,
                      help='File containing true samples (for synthetic data evaluation)')
    parser.add_argument('--bn_file', type=str, default='bn.txt',
                      help='File containing Bayesian network structure')
    parser.add_argument('--output_dir', type=str, default='./output',
                      help='Directory to save output files')
    
    # EM algorithm parameters
    parser.add_argument('--max_iter', type=int, default=20,
                      help='Maximum number of EM iterations')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for processing samples')
    parser.add_argument('--n_workers', type=int, default=64,
                      help='Number of parallel workers')
    parser.add_argument('--convergence_threshold', type=float, default=1e-9,
                      help='Convergence threshold for EM algorithm')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
        
    if args.dataset == 'air_quality':
        # Download and load the Air Quality dataset
        data_file = download_air_quality_dataset("./data")
        df = load_air_quality_data(data_file)
        
        # Process air quality data
        missing_samples = process_air_quality_data(data_file)
        
        # Create Bayesian network structure for air quality data
        bn = AirQualityBayesNet()
        
        # Create a temporary file to store the network structure
        temp_bn_file = os.path.join(args.output_dir, "air_quality_bn.txt")
        
        # Write network structure to file
        bn.write_to_file(temp_bn_file)
        
        # Create EMBayesNet using the temporary file
        em_bn = EMBayesNet(temp_bn_file)
        
        # Run EM algorithm with adjusted parameters for air quality dataset
        print("\nRunning EM algorithm on air quality dataset...")
        em_bn.learn_cpds(
            samples_with_missing=missing_samples,
            max_iterations=50,  # Increased max iterations
            convergence_threshold=1e-6,  # Relaxed convergence threshold
            alpha=0.5,  # Reduced alpha for smoother updates
            batch_size=32,  # Increased batch size
            n_workers=args.n_workers,
            momentum=0.3,  # Added momentum
            patience=5,  # Added patience for early stopping
            min_iterations=10  # Minimum iterations before early stopping
        )

        # Visualize missing value patterns
        em_bn.visualize_missing_pattern(
            df=pd.DataFrame(missing_samples),
            output_file=os.path.join(args.output_dir, "air_quality_missing_pattern.png")
        )
        
        # Visualize imputed data patterns
        if em_bn.imputed_samples:
            em_bn.visualize_imputed_pattern(
                samples=[imputed for _, imputed in em_bn.imputed_samples[-1].items()],
                output_file=os.path.join(args.output_dir, "air_quality_imputed_pattern.png")
            )
        
        # Visualize convergence
        em_bn.visualize_convergence(
            output_dir=os.path.join(args.output_dir, "air_quality_convergence")
        )
        
        # Analyze imputation results
        analyze_air_quality_imputation(df, em_bn.imputed_samples[-1], args.output_dir)
        
        # Compare with original values
        em_bn.visualize_imputation_comparison(
            true_samples=[{col: int(val) for col, val in row.items() if pd.notna(val)} 
                         for _, row in df.iterrows()],
            missing_samples=missing_samples,
            output_file=os.path.join(args.output_dir, "air_quality_imputation_comparison.png")
        )
        
        # Clean up temporary file
        os.remove(temp_bn_file)
        
    else:
        # Original synthetic dataset processing
        missing_samples = read_samples(args.samples_file)
        true_samples = read_samples(args.true_samples_file) if args.true_samples_file else None
        em_bn = EMBayesNet(args.bn_file)
        
        print("\nRunning EM algorithm on synthetic dataset...")
        em_bn.learn_cpds(
            samples_with_missing=missing_samples,
            max_iterations=args.max_iter,
            convergence_threshold=args.convergence_threshold,
            alpha=0.1,
            batch_size=args.batch_size,
            n_workers=args.n_workers
        )
        
        # Visualize missing value patterns
        em_bn.visualize_missing_pattern(
            df=pd.DataFrame(missing_samples),
            output_file=os.path.join(args.output_dir, "synthetic_missing_pattern.png")
        )
        
        # Visualize imputed data patterns
        if em_bn.imputed_samples:
            em_bn.visualize_imputed_pattern(
                samples=[imputed for _, imputed in em_bn.imputed_samples[-1].items()],
                output_file=os.path.join(args.output_dir, "synthetic_imputed_pattern.png")
            )
        
        # Visualize convergence
        em_bn.visualize_convergence(
            output_dir=os.path.join(args.output_dir, "synthetic_convergence")
        )
        
        # Compare with true values if available
        if true_samples:
            # Visualize comparison between true and imputed values
            em_bn.visualize_imputation_comparison(
                true_samples=true_samples,
                missing_samples=missing_samples,
                output_file=os.path.join(args.output_dir, "synthetic_imputation_comparison.png")
            )
            
            results, detailed_metrics = em_bn.compare_with_true_values(
                true_samples=true_samples,
                missing_samples=missing_samples,
                output_dir=args.output_dir
            )
            
            # Save comparison metrics
            with open(os.path.join(args.output_dir, "synthetic_comparison_metrics.txt"), 'w') as f:
                f.write("Comparison with True Values:\n")
                f.write("\nOverall Metrics:\n")
                for var, metrics in results.items():
                    f.write(f"\n{var}:\n")
                    f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                    f.write("  Confusion Matrix:\n")
                    cm = metrics['confusion_matrix']
                    f.write(f"    TN: {cm[0,0]}, FP: {cm[0,1]}\n")
                    f.write(f"    FN: {cm[1,0]}, TP: {cm[1,1]}\n")
                
                f.write("\nDetailed Metrics:\n")
                for var, var_metrics in detailed_metrics.items():
                    f.write(f"\n{var}:\n")
                    for metric, value in var_metrics.items():
                        if isinstance(value, dict):
                            f.write(f"  {metric}:\n")
                            for k, v in value.items():
                                if isinstance(v, float):
                                    f.write(f"    {k}: {v:.4f}\n")
                        elif isinstance(value, float):
                            f.write(f"  {metric}: {value:.4f}\n")
            
            print("\nComparison metrics saved to: "
                  f"{args.output_dir}/synthetic_comparison_metrics.txt")
        
        print("\nVisualization files saved:")
        print(f"- Missing pattern: {args.output_dir}/synthetic_missing_pattern.png")
        print(f"- Imputed pattern: {args.output_dir}/synthetic_imputed_pattern.png")
        print(f"- Convergence plot: {args.output_dir}/synthetic_convergence.png")

if __name__ == "__main__":
    main()