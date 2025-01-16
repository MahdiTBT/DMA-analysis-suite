"""
Module: core.py

This module defines three classes for handling DMA master-curves and temperature dependence:

1) MasterCurve
   - Reads text-based DMA data named like 'T0.txt', 'T-10.txt', etc.
   - Cleans & organizes the data.
   - Builds a 'master curve' by shifting and aligning test data at different temperatures.

2) MastercurveFitter
   - Takes a master-curved (or otherwise frequency vs. modulus) dataset.
   - Filters out extraneous data above a 'checkpoint' modulus.
   - Fits Prony series parameters (e_i, E_0) using L-BFGS-B optimization.

3) TemperatureDependency
   - Given temperature T and shift values (log or linear),
   - Fits a polynomial relationship (e.g., degree=3).
   - Plots and returns the polynomial for evaluating shift factors at arbitrary temperatures.
"""

import os
import re
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import scipy.optimize as opt
from scipy.optimize import minimize
from sklearn.metrics import r2_score

# ------------------------------------------------------------------------------
# Global configurations (fonts, warnings, etc.)
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Matplotlib styling
plt.rc('font', family='serif', serif='Times New Roman')
plt.rcParams.update({
    'font.size': 13,         # Font size for axis labels, tick labels, legends
    'axes.titlesize': 16,    # Font size for titles
    'axes.labelsize': 16,    # Font size for axis labels
    'legend.fontsize': 13,   # Font size for legend
    'mathtext.fontset': 'cm' # Computer Modern for math text
})
plt.rcParams['font.family'] = 'serif'

# Set figure dimensions (inches)
figure_width_mm = 180       # desired width in mm
aspect_ratio = 4 / 3
figure_width_inches = figure_width_mm / 25.4
figure_height_inches = figure_width_inches / aspect_ratio
plt.rcParams['figure.figsize'] = (figure_width_inches, figure_height_inches)

# Pandas & NumPy display
pd.options.display.float_format = '{:.3f}'.format
np.set_printoptions(precision=3, suppress=True)

# ------------------------------------------------------------------------------ 
# 1) MasterCurve class
# ------------------------------------------------------------------------------
class MasterCurve:
    """
    The MasterCurve class reads, processes, and aligns DMA test data from text files
    named like 'T0.txt', 'T-10.txt', etc., culminating in a 'master curve'.

    Key Features:
    1) Columns & reading logic can be configured (init parameters).
    2) Data cleaning via `clean_dataframe()`.
    3) Objective function for shifting can be custom or default.
    4) Plots for quick check of Storage & Loss data (log10 scale).
    5) Builds a master curve by iteratively shifting data sets relative to a reference temperature.
    """

    def __init__(
        self,
        columns=None,
        skiprows: int = 2,
        skipfooter: int = 1,
        delimiter: str = r'\s+',
        engine: str = 'python',
        encoding: str = 'iso-8859-1',
        objective_func=None,
        do_plot: bool = True
    ):
        """
        Parameters
        ----------
        columns : list of str or None
            Column names for the input .txt files. If None, a default list is used.
        skiprows : int
            Number of rows to skip at the start of each file.
        skipfooter : int
            Number of rows to skip at the end of each file.
        delimiter : str
            Delimiter for file reading (regex supported).
        engine : str
            The parsing engine for pandas (commonly 'python' or 'c').
        encoding : str
            File encoding for reading the data.
        objective_func : callable or None
            Custom objective function for shifting. Must match signature:
                objective_func(shift_val, Data_ref, Data_shifting) -> float
            If None, uses self.default_objective_function.
        do_plot : bool
            Whether to produce a quick check plot for Storage & Loss after reading.
        """
        # Default columns if none provided
        self.columns = columns or [
            'Index', 'Ts', 't', 'f', 'F', 'x', 'Phase', 'F0', 'x0',
            'Tr', 'Storage', 'Loss', 'E^*', 'tan_delta', 'D_p', 'D_Dp',
            'D^*', 'Eta_p', 'Eta_Dp', 'Eta^*'
        ]
        self.skiprows = skiprows
        self.skipfooter = skipfooter
        self.delimiter = delimiter
        self.engine = engine
        self.encoding = encoding
        self.do_plot = do_plot

        # Data dictionary to store each file's DataFrame (keyed by "T{temp}C")
        self.data_dict = {}

        # Objective function for shifting (user-provided or default).
        self.objective_func = objective_func or self.default_objective_function

    def read_data(self, data_dir: Path) -> pd.DataFrame:
        """
        Reads text files in `data_dir` named like 'T0.txt', 'T-10.txt', etc.
        Applies cleaning, stores them in self.data_dict, and optionally
        generates a quick check plot (Storage, Loss in log10).

        Parameters
        ----------
        data_dir : Path
            A path object pointing to the directory with .txt data files.

        Returns
        -------
        pd.DataFrame
            DataFrame named 'File_names' containing:
            - "File Name" (string)
            - "Temperature" (int)
            - "Jupyter Name" (for referencing self.data_dict)
        """
        # Collect all .txt files that match pattern "T*.txt"
        file_list = sorted(data_dir.glob("T*.txt"))
        if not file_list:
            logging.warning(f"No files found matching 'T*.txt' in {data_dir}")
            return pd.DataFrame()

        metadata_records = []
        for file_path in file_list:
            file_name = file_path.name

            # Attempt to extract temperature from filename (e.g., T-10.txt => -10)
            match = re.findall(r"T(-?\d+)", file_name)
            if not match:
                logging.warning(f"Could not parse temperature from file name: {file_name}")
                continue

            temp = int(match[0])  # convert string to int
            metadata_records.append((file_name, temp))

        # Build a DataFrame of files & temperatures
        File_names = pd.DataFrame(metadata_records, columns=["File Name", "Temperature"])
        File_names.sort_values(by="Temperature", inplace=True, ignore_index=True)

        # Generate a "Jupyter Name" for each file and read into data_dict
        for idx, row in File_names.iterrows():
            temp = row["Temperature"]
            jupyter_name = f"T{temp}C"
            File_names.loc[idx, "Jupyter Name"] = jupyter_name

            try:
                # Read each file with user-specified parameters
                df = pd.read_csv(
                    data_dir / row["File Name"],
                    delimiter=self.delimiter,
                    skiprows=self.skiprows,
                    skipfooter=self.skipfooter,
                    encoding=self.encoding,
                    engine=self.engine,
                    names=self.columns
                )
            except Exception as e:
                logging.error(f"Error reading file {row['File Name']}: {e}")
                continue

            # Convert Ts, Tr from Celsius to Kelvin
            df['Ts'] = df['Ts'] + 273.15
            df['Tr'] = df['Tr'] + 273.15

            # Perform cleaning (remove invalid rows, etc.)
            df = self.clean_dataframe(df)

            # Store in data_dict
            self.data_dict[jupyter_name] = df

        # If plotting is enabled, produce a quick check chart
        # Generate a qualitative color palette for 22 curves
        num_curves = len(File_names)
        palette = sns.color_palette("husl", num_curves)  # HSV colormap for high contrast

        if self.do_plot and not File_names.empty:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            for idx, (row, color) in enumerate(zip(File_names.iterrows(), palette)):
                jupyter_name = row[1]["Jupyter Name"]
                temp = row[1]["Temperature"]
                df = self.data_dict[jupyter_name]

                f_log = np.log10(df['f'])
                storage_log = np.log10(df['Storage'])
                loss_log = np.log10(df['Loss'])

                axs[0].plot(10**f_log, 10**storage_log, label=f"{temp}°C", color=color, linewidth=1.5)
                axs[1].plot(10**f_log, 10**loss_log, label=f"{temp}°C", color=color, linewidth=1.5)

            # Left subplot: Storage
            #axs[0].set_title("Quick Check: Storage (Log-Log)")
            axs[0].set_xlabel(r"$f\ \mathrm{[Hz]}$")
            axs[0].set_ylabel(r"Storage Modulus $\mathrm{[GPa]}$")
            axs[0].set_xscale('log')  # Logarithmic scale on x-axis
            axs[0].set_yscale('log')  # Logarithmic scale on y-axis
            axs[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7) 

            # Right subplot: Loss
            #axs[1].set_title("Quick Check: Loss (Log-Log)")
            axs[1].set_xlabel(r"$f\ \mathrm{[Hz]}$")
            axs[1].set_ylabel(r"Loss Modulus $\mathrm{[GPa]}$")
            axs[1].set_xscale('log')  # Logarithmic scale on x-axis
            axs[1].set_yscale('log')  # Logarithmic scale on y-axis
            axs[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

            # Single legend for all curves
            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.1))

            plt.tight_layout(rect=[0, 0.07, 1, 0.95])  # Adjust layout for the legend
            plt.show()

        return File_names

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by removing rows with non-positive Storage or Loss.
        Override to add more sophisticated rules if needed.

        Parameters
        ----------
        df : pd.DataFrame
            Raw data after reading.

        Returns
        -------
        pd.DataFrame
            Cleaned data.
        """
        before_len = len(df)
        df = df[df['Storage'] > 0]
        df = df[df['Loss'] > 0]
        df.reset_index(drop=True, inplace=True)

        if len(df) < before_len:
            logging.info("Removed rows with non-positive Storage/Loss.")

        return df

    def default_objective_function(
        self,
        shift_val: float,
        Data_ref: pd.DataFrame,
        Data_shifting: pd.DataFrame
    ) -> float:
        """
        Default function measuring sum of squared relative differences
        in log10(Storage) between reference & shifted curves.

        The shifting is done by adjusting the 'f' axis of Data_shifting by shift_val,
        and interpolating to compare against the Data_ref curve.
        """
        baseline_x = Data_ref['f']
        baseline_y = Data_ref['Storage']

        curve_x = Data_shifting['f']
        curve_y = Data_shifting['Storage']

        # Shift 'curve_x' by shift_val and interpolate for direct comparison
        shifted_curve_y = np.interp(baseline_x, curve_x + shift_val, curve_y)

        # Sum of squared relative differences
        return np.sum(((shifted_curve_y - baseline_y) ** 2) / (baseline_y ** 2))

    def calculate_shift(
        self,
        Data_ref: pd.DataFrame,
        Data_shifting: pd.DataFrame
    ) -> float:
        """
        Minimizes self.objective_func to find the best horizontal shift aligning
        Data_shifting to Data_ref.

        Parameters
        ----------
        Data_ref : pd.DataFrame
        Data_shifting : pd.DataFrame

        Returns
        -------
        float
            Optimal shift value (log10 domain).
        """
        result = opt.minimize(
            self.objective_func,
            x0=0.0,  # initial guess
            args=(Data_ref, Data_shifting),
            method='Nelder-Mead'  # or another method if you prefer
        )
        return result.x[0]

    def ref_temp(self, File_names: pd.DataFrame, T_ref: float, shift: float):
        """
        Finds reference data at T_ref in File_names, converts f, Storage, Loss to log10,
        applies a base shift, returns Data_ref plus the next closest temperature data for shifting.

        Parameters
        ----------
        File_names : pd.DataFrame
            Contains "File Name", "Temperature", "Jupyter Name"
        T_ref : float
            Reference temperature (in same units as File_names['Temperature'])
        shift : float
            Initial shift to apply to Data_ref's frequency axis

        Returns
        -------
        Data_ref : pd.DataFrame or None
            The reference dataset with log10 transforms
        Data_shifting : pd.DataFrame or None
            The next-closest dataset to shift
        File_names : pd.DataFrame
            Updated (with T_ref row removed)
        T_next : float or None
            Temperature of the next dataset
        next_index : int or None
            Index of next dataset in File_names
        """
        Data_ref, Data_shifting = None, None
        T_next, next_index = None, None

        for i, row in File_names.iterrows():
            if T_ref == row['Temperature']:
                jupyter_name = row['Jupyter Name']
                df_ref = self.data_dict[jupyter_name][['Ts', 'f', 'Storage', 'Loss', 'Tr']].copy()

                # Convert to log10 domain, plus apply shift
                df_ref['f'] = np.log10(df_ref['f']) + shift
                df_ref['Storage'] = np.log10(df_ref['Storage'])
                df_ref['Loss'] = np.log10(df_ref['Loss'])
                df_ref['Tr'] = df_ref['Tr'] - 273.15  # Convert back to °C if needed

                Data_ref = df_ref

                # Remove reference row from File_names
                File_names.drop(i, inplace=True)
                File_names.reset_index(drop=True, inplace=True)

                # If there's still data left, pick the next-closest temperature
                if not File_names.empty:
                    next_index = File_names['Temperature'].sub(T_ref).abs().idxmin()
                    T_next = File_names.loc[next_index, 'Temperature']
                    jupyter_name_shift = File_names.loc[next_index, 'Jupyter Name']

                    df_shift = self.data_dict[jupyter_name_shift][['Ts', 'f', 'Storage', 'Loss', 'Tr']].copy()
                    df_shift['f'] = np.log10(df_shift['f']) + shift
                    df_shift['Storage'] = np.log10(df_shift['Storage'])
                    df_shift['Loss'] = np.log10(df_shift['Loss'])
                    df_shift['Tr'] = df_shift['Tr'] - 273.15

                    Data_shifting = df_shift
                break
        else:
            logging.warning(f"Temperature {T_ref} not found in File_names.")

        return Data_ref, Data_shifting, File_names, T_next, next_index

    def apply_shift(
        self,
        Data_shifting: pd.DataFrame,
        Data_ref: pd.DataFrame,
        next_index: int,
        File_names: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame, float):
        """
        Calculates and applies the optimal shift to Data_shifting,
        then concatenates Data_shifting with Data_ref.

        Parameters
        ----------
        Data_shifting : pd.DataFrame
        Data_ref : pd.DataFrame
        next_index : int
            Index of the row in File_names for the shifting dataset
        File_names : pd.DataFrame

        Returns
        -------
        Data_shifting : pd.DataFrame
            Shifted dataset (modified in place)
        data_final : pd.DataFrame
            Concatenated reference + shifted dataset
        shift_val : float
            The shift that was applied
        """
        shift_val = self.calculate_shift(Data_ref, Data_shifting)
        # Apply shift
        Data_shifting['f'] += shift_val
        Data_shifting['Shift_value'] = shift_val

        data_final = pd.concat([Data_ref, Data_shifting], ignore_index=True)
        return Data_shifting, data_final, shift_val

    def build_master_curve(self, path: Path) -> (pd.DataFrame, pd.DataFrame):
        """
        Main pipeline to construct the master curve:
          1) read_data() from T*.txt in `path`
          2) choose T_ref as the minimum temperature
          3) iteratively align each next-closest temperature
          4) return the final "master curve" and File_names

        Returns
        -------
        Final : pd.DataFrame
            The master curve DataFrame with log10(f), log10(Storage), log10(Loss).
        File_names : pd.DataFrame
            The original metadata (filenames, temperature, jupyter names).
        """
        File_names = self.read_data(path)
        if File_names.empty:
            logging.warning("No data loaded. Returning empty DataFrame.")
            return pd.DataFrame(), File_names

        # Use the minimum temperature in File_names as initial reference
        T_ref = File_names['Temperature'].min()
        total_shift = 0.0
        Final = pd.DataFrame()
        fnames_copy = File_names.copy()

        # Iteratively shift data sets until no more remain
        for _ in range(len(fnames_copy)):
            Data_ref, Data_shifting, fnames_copy, T_next, next_idx = self.ref_temp(
                fnames_copy, T_ref, total_shift
            )
            if Data_ref is None:
                # Something went wrong (temperature not found or no data left)
                break

            if Final.empty:
                # First time: just set Final = Data_ref
                Final = Data_ref.copy()

            # Apply shift to next dataset
            Data_shifting, data_final, shift_val = self.apply_shift(
                Data_shifting, Data_ref, next_idx, fnames_copy
            )
            Final = pd.concat([Final, Data_shifting], ignore_index=True)

            # Update reference temperature for the next iteration
            T_ref = T_next if T_next is not None else T_ref
            total_shift += shift_val

            if len(fnames_copy) == 1:
                break

        # Sort final data by frequency
        Final.sort_values(by='f', inplace=True)
        # Fill any NaNs in Shift_value column with 0
        Final['Shift_value'] = Final['Shift_value'].fillna(0)

        return Final, File_names

    def adjust_shift(
        self,
        T_ref_new: float,
        Temp_Shift: pd.DataFrame,
        Final: pd.DataFrame
    ) -> (float, float, float, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Adjusts the entire master curve to a new reference temperature T_ref_new,
        then plots:
          - log(Storage), log(Loss) before & after shift
          - Arrhenius-like plot ln(a_T) vs 1/T

        Parameters
        ----------
        T_ref_new : float
            Desired new reference temperature (in °C)
        Temp_Shift : pd.DataFrame
            Contains 'Tr' (temperature) and 'Shift_value' (log10(a_T))
        Final : pd.DataFrame
            The master curve data to be re-shifted

        Returns
        -------
        slope : float
            Slope from linear fit of ln(a_T) vs 1/T
        intercept : float
            Intercept from linear fit
        Ea : float
            Activation energy in J/mol (divide by 1000 for kJ/mol)
        Final : pd.DataFrame
            The original Final (unchanged)
        Temp_Shift : pd.DataFrame
            The shift table, partially updated
        Final_after_shift : pd.DataFrame
            The re-shifted master curve
        """
        # If T_ref_new is in Temp_Shift table, pick that shift; else interpolate
        if T_ref_new in Temp_Shift['Tr'].values:
            shift_Tref = Temp_Shift.loc[Temp_Shift['Tr'] == T_ref_new, 'Shift_value'].values[0]
        else:
            shift_Tref = np.interp(T_ref_new, Temp_Shift['Tr'], Temp_Shift['Shift_value'])

        logging.info(f"Shift for T_ref={T_ref_new} is {shift_Tref}")

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot Storage: before & after shift
        #axs[0].plot(10**Final['f'], 10**Final['Storage'], label='Before Shift')
        axs[0].plot(10**(Final['f'] - shift_Tref), 10**Final['Storage'], label=f"Master Curve at {T_ref_new}°C")
        #axs[0].set_title("Master Curve: Storage Modulus")
        axs[0].set_xlabel(r"$f\ \mathrm{[Hz]}$")
        axs[0].set_ylabel(r"Storage Modulus $\mathrm{[GPa]}$")
        axs[0].set_xscale('log')  # Logarithmic scale on x-axis
        axs[0].set_yscale('log')  # Logarithmic scale on y-axis
        axs[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        axs[0].legend()

        # Plot Loss: before & after shift
        #axs[1].plot(10**Final['f'], 10**Final['Loss'], label='Before Shift')
        axs[1].plot(10**(Final['f'] - shift_Tref), 10**Final['Loss'], label=f"Master Curve at {T_ref_new}°C")
        # also plot the initial data in first plot

        #axs[1].set_title("Loss Modulus After Shift")
        axs[1].set_xlabel(r"$f\ \mathrm{[Hz]}$")
        axs[1].set_ylabel(r"Loss Modulus $\mathrm{[GPa]}$")
        axs[1].set_xscale('log')  # Logarithmic scale on x-axis
        axs[1].set_yscale('log')  # Logarithmic scale on y-axis
        axs[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        axs[1].legend()

        plt.tight_layout()
        plt.show()

        # Arrhenius-like plot: ln(a_T) vs. 1/(T+273.15)
        plt.figure()
        
        x_vals = 1 / (Temp_Shift['Tr'] + 273.15)  # 1/T in K
        # Convert from log10 shift_value to ln(a_T)
        y_vals = np.log(10 ** Temp_Shift['Shift_value'])
        plt.scatter(x_vals, y_vals, label='Shift Values', marker='o')

        # Linear fit
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        plt.plot(x_vals, slope * x_vals + intercept, label='Linear Fit')
        plt.title("ln(a_T) vs. 1/T")
        plt.legend()
        plt.show()

        R = 8.314  # J/(mol·K)
        Ea = slope * R  # J/mol (divide by 1000 for kJ/mol)
        logging.info(f"Activation energy (kJ/mol): {Ea / 1000:.2f}")

        # Build a copy of Final, re-shift to new reference
        Final_after_shift = Final.copy()
        Final_after_shift['f'] -= shift_Tref
        Final_after_shift['Shift_value'] -= shift_Tref
        Temp_Shift['Shift_value'] -= shift_Tref

        return slope, intercept, Ea, Final, Temp_Shift, Final_after_shift

# ------------------------------------------------------------------------------ 
# 2) MastercurveFitter class
# ------------------------------------------------------------------------------
class MastercurveFitter:
    """
    Class to handle:
      1) Data filtering based on a given checkpoint
      2) Prony series fitting for the given data (Storage, Loss vs. frequency)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        checkpoint: float = 2.4e3,
        E_infty: float = 2.9e3,
        lambda_reg: float = 0.01,
        rolling_window: int = 20,
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least columns: 'f', 'Storage', 'Loss'
        checkpoint : float
            Threshold for 'Complex' modulus trimming
        E_infty : float
            Reference modulus scale (used in objective function normalization)
        lambda_reg : float
            Regularization parameter used in the objective function
        rolling_window : int
            Window size for optional rolling mean smoothing
        """
        self.df_raw = df.copy()
        self.checkpoint = checkpoint
        self.E_infty = E_infty
        self.lambda_reg = lambda_reg
        self.rolling_window = rolling_window

        # Internal placeholders
        self.df_filtered = None
        self.tau = None
        self.optimized_e = None
        self.optimized_E_0 = None
        self.result = None

    def preprocess_and_filter(self):
        """
        1) (Optional) Convert frequencies/moduli if needed (already done externally? skip it).
        2) Compute 'Complex' and 'Phase'.
        3) Apply rolling mean smoothing.
        4) Filter rows with 'Complex' >= checkpoint.
        """
        df = self.df_raw.copy()

        # Example conversions (commented if already done externally):
        # df['f']       = (10**df['f']) * 2*np.pi
        # df['Storage'] = 10**df['Storage']
        # df['Loss']    = 10**df['Loss']

        # Compute Complex modulus & phase
        df['Complex'] = np.abs(df['Storage'] + 1j * df['Loss'])
        df['Phase'] = np.angle(df['Storage'] + 1j * df['Loss'])

        # Rolling mean smoothing
        for col in ['Storage', 'Loss', 'Complex', 'Phase']:
            df[col] = df[col].rolling(window=self.rolling_window).mean()

        # Drop rows that became NaN after rolling
        df.dropna(inplace=True)

        # Filter out data points where Complex >= checkpoint
        df_filtered = df[df['Complex'] < self.checkpoint].copy()
        self.df_filtered = df_filtered

    def compute_tau(self):
        """
        Compute the array of relaxation times (tau) based on logspace over 
        the frequency range in self.df_filtered.
        """
        if self.df_filtered is None:
            raise ValueError("Must run preprocess_and_filter() before computing tau.")

        df_filtered = self.df_filtered
        f_min = df_filtered['f'].min()
        f_max = df_filtered['f'].max()

        # Construct tau array
        tau = np.logspace(
            np.ceil(np.log10(f_min)),
            np.floor(np.log10(f_max)),
            int(
                np.floor(np.log10(f_max))
                - np.ceil(np.log10(f_min))
                + 1
            )
        )
        self.tau = tau

    @staticmethod
    def prony_storage(omega, e, tau, E_0):
        """
        Prony series Storage modulus E'(ω).
        E'(ω) = E_0*(1-Σe_i) + E_0 * Σ[e_i*(τ_i²*ω²)/(1+τ_i²*ω²)].
        """
        return E_0 * (1 - np.sum(e)) + E_0 * np.sum(
            [
                e_i * ((tau_i**2) * (omega**2)) / (1 + (tau_i**2)*(omega**2))
                for e_i, tau_i in zip(e, tau)
            ],
            axis=0,
        )

    @staticmethod
    def prony_loss(omega, e, tau, E_0):
        """
        Prony series Loss modulus E''(ω).
        E''(ω) = E_0 * Σ[e_i*(τ_i*ω)/(1+τ_i²*ω²)].
        """
        return E_0 * np.sum(
            [
                e_i * (tau_i*omega) / (1 + (tau_i**2)*(omega**2))
                for e_i, tau_i in zip(e, tau)
            ],
            axis=0,
        )

    def objective_function(self, params, omega, E_storage_data, E_loss_data):
        """
        Objective function for L-BFGS-B with L2 regularization on 'e'.
        Minimizes the sum of squared errors (normalized by E_infty²) + λ * Σ(e_i²).
        """
        e = params[:-1]  # first N are e_i
        E_0 = params[-1] # last param is E_0

        # Predicted
        E_s = self.prony_storage(omega, e, self.tau, E_0)
        E_l = self.prony_loss(omega, e, self.tau, E_0)

        # L2 regularization on e
        reg_term = self.lambda_reg * np.sum(e**2)

        return (
            np.sum((E_storage_data - E_s) ** 2 / (self.E_infty ** 2))
            + np.sum((E_loss_data - E_l) ** 2 / (self.E_infty ** 2))
            + reg_term
        )

    def fit_prony_series(self):
        """
        Runs L-BFGS-B optimization to find e_i, E_0 that minimize 
        self.objective_function, respecting e_i in [0,1], E_0>=0.
        """
        if self.df_filtered is None:
            raise ValueError("Must run preprocess_and_filter() before fitting.")
        if self.tau is None:
            raise ValueError("Must run compute_tau() before fitting.")

        df_filtered = self.df_filtered
        omega = df_filtered['f'].values
        E_storage_data = df_filtered['Storage'].values
        E_loss_data = df_filtered['Loss'].values

        # Initial guesses
        initial_e   = np.ones(len(self.tau)) * 0.5
        initial_E_0 = 1.0
        initial_params = np.append(initial_e, initial_E_0)

        # Bounds: each e_i in [0,1], E_0 >= 0
        bounds = [(0, 1)] * len(self.tau) + [(0, None)]

        result = minimize(
            self.objective_function,
            initial_params,
            args=(omega, E_storage_data, E_loss_data),
            bounds=bounds,
            method='L-BFGS-B'
        )

        self.result = result
        self.optimized_e   = result.x[:-1]
        self.optimized_E_0 = result.x[-1]

        if not result.success:
            print("Warning: Prony optimization did not converge successfully.")
        else:
            print("Optimization for Prony series succeeded.")

    def plot_fitted_results(self):
        """
        Plots experimental vs. fitted Storage & Loss moduli (both in log scale).
        Shows R^2 for each.
        """
        if self.df_filtered is None or self.result is None:
            raise ValueError("Must run preprocess_and_filter(), compute_tau(), and fit first.")

        df_filtered = self.df_filtered
        omega = df_filtered['f'].values

        # Compute fitted curves
        fitted_storage = self.prony_storage(omega, self.optimized_e, self.tau, self.optimized_E_0)
        fitted_loss    = self.prony_loss(omega, self.optimized_e, self.tau, self.optimized_E_0)

        # R^2
        r2_storage = r2_score(df_filtered['Storage'], fitted_storage)
        r2_loss    = r2_score(df_filtered['Loss'], fitted_loss)

        # Plot
        plt.figure(figsize=(12, 6))

        # 1) Storage
        plt.subplot(1, 2, 1)
        plt.plot(df_filtered['f'], df_filtered['Storage'], 'b.-', label="Generated Master Curve")
        plt.plot(df_filtered['f'], fitted_storage, 'r--', label=f"Fitted Prony Series (R²={r2_storage:.3f})")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xlabel(r"$f\ \mathrm{[rad/s]}$")
        plt.ylabel(r"Storage Modulus $\mathrm{[GPa]}$")
#        plt.title("Storage Modulus vs Frequency")
        plt.legend()

        # 2) Loss
        plt.subplot(1, 2, 2)
        plt.plot(df_filtered['f'], df_filtered['Loss'], 'b.-', label="Generated Master Curve")
        plt.plot(df_filtered['f'], fitted_loss, 'r--', label=f"Fitted Prony Series (R²={r2_loss:.3f})")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xlabel(r"$f\ \mathrm{[rad/s]}$")
        plt.ylabel(r"Loss Modulus $\mathrm{[GPa]}$")
#        plt.title("Loss Modulus vs Frequency")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def get_prony_parameters(self):
        """
        Returns the Prony series parameters in a DataFrame:
            tau, e_i, and E_0 (the last is repeated in each row).
        """
        if self.optimized_e is None or self.optimized_E_0 is None:
            raise ValueError("No fitted parameters. Run fit_prony_series() first.")

        df_prony = pd.DataFrame({
            'tau': self.tau,
            'e':   self.optimized_e
        })
        df_prony['E_0'] = self.optimized_E_0
        E_0_final= self.optimized_E_0
        return df_prony , E_0_final

# ------------------------------------------------------------------------------ 
# 3) TemperatureDependency class
# ------------------------------------------------------------------------------
class TemperatureDependency:
    """
    Class to handle temperature-dependent shift factor data, 
    and perform polynomial fits of shift vs. temperature.
    """

    def __init__(self, T, shift, T_ref=20.0):
        """
        Parameters
        ----------
        T : array-like
            Temperatures in Celsius (or any consistent unit)
        shift : array-like
            Shift values (log10 or linear scale, depending on your data)
        T_ref : float
            Reference temperature (optional). Default = 20 °C
        """
        self.T = np.array(T, dtype=float)
        self.shift = np.array(shift, dtype=float)
        self.T_ref = T_ref

        self.degree = None
        self.coefficients = None
        self.r2 = None

    def fit_polynomial(self, degree=3):
        """
        Fits a polynomial of the given degree to (T, shift).

        Parameters
        ----------
        degree : int
            Polynomial degree.

        Returns
        -------
        coeffs : np.ndarray
            Polynomial coefficients [a_deg, ..., a_0]
        r2 : float
            R^2 of the fit
        """
        self.degree = degree
        # Fit polynomial
        self.coefficients = np.polyfit(self.T, self.shift, degree)

        # Evaluate polynomial at the original T
        y_fit = np.polyval(self.coefficients, self.T)
        self.r2 = r2_score(self.shift, y_fit)

        return self.coefficients, self.r2

    def plot_fit(self):
        """
        Plots the raw data (T vs. shift) and the fitted polynomial,
        with an R^2 annotation in the legend.
        """
        if self.coefficients is None:
            raise ValueError("No polynomial fit. Call fit_polynomial() first.")

        T_plot = np.linspace(self.T.min(), self.T.max(), 300)
        Y_fit_plot = np.polyval(self.coefficients, T_plot)

        plt.figure()
        plt.plot(self.T, self.shift, 'o', label='Data')
        plt.plot(T_plot, Y_fit_plot, 'r-', 
                 label=f'Poly Fit (deg={self.degree}), R^2={self.r2:.2f}')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Shift value')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_polynomial(self):
        """
        Returns the fitted polynomial as np.poly1d.

        Returns
        -------
        poly : np.poly1d
            The fitted polynomial function for shift vs. temperature.
        """
        if self.coefficients is None:
            raise ValueError("No polynomial fit. Call fit_polynomial() first.")
        return np.poly1d(self.coefficients)

    def print_coefficients(self):
        """
        Prints polynomial coefficients in descending powers: a_deg, ..., a_0.
        """
        if self.coefficients is None:
            raise ValueError("No polynomial fit. Call fit_polynomial() first.")

        print(f"Polynomial degree: {self.degree}")
        for i, coeff in enumerate(self.coefficients):
            print(f"a_{self.degree - i} = {coeff:.6f}")

# ------------------------------------------------------------------------------
# Optionally: if you want to include a main test block
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("This module defines MasterCurve, MastercurveFitter, and TemperatureDependency classes.")
    print("You can import these classes into your scripts or notebooks to read, process, and fit DMA data.")
