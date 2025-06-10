import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import requests
import joblib
import os
import shutil
import google.generativeai as genai
import sys # Import sys for potential exit if truly needed (though st.stop() is preferred in Streamlit)

# Assuming these are in your project directory (or adjust paths if they are elsewhere)
from src.merge_flow import *
from src.simulate_flow import *
from src.utils import * # This is where map_port, preprocess_dataframe, and choose_label are now expected to be

# Global constant for Gemini model
GEMINI_MODEL = "gemini-1.5-flash"

# --- Initialize session state variables if they don't exist ---
# This is the crucial part for robustness
if 'initial_setup_completed' not in st.session_state:
    st.session_state.initial_setup_completed = False
if 'setup_failed' not in st.session_state:
    st.session_state.setup_failed = False
if 'show_setup_logs' not in st.session_state:
    st.session_state.show_setup_logs = False
if 'model_state' not in st.session_state: # Initialize model_state
    st.session_state.model_state = None # Set to None initially
    
# --- Your existing cached setup function ---
@st.cache_data(show_spinner=False)
def initial_setup_cached(cache_key_for_setup):
    # These resets are crucial for the first run or after "Analyze another file"
    # Keeping them for the first setup call, but the global initialization handles subsequent ones
    st.session_state.initial_setup_completed = False
    st.session_state.setup_failed = False

    success = display_setup_logs()

    st.session_state.initial_setup_completed = success
    st.session_state.setup_failed = not success
    st.session_state.show_setup_logs = True
    return success

# --- LLM Configuration Function ---
@st.cache_resource(show_spinner="Connecting to Gemini...")
def configure_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        return model
    except Exception as e:
        print(f"Error configuring Gemini (within cache_resource): {e}") # Log to console
        return None

def display_setup_logs():
    # CICFlowMeter setup
    with st.status("Setting up CICFlowMeter-3.0...", expanded=True, state="running") as status:
        try:
            # install libpcap-dev library
            st.write(":arrow_down: Installing libpcap-dev...")
            subprocess.run(["sudo", "apt-get", "update"], check=True, capture_output=True, text=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "libpcap-dev"], check=True, capture_output=True, text=True)
            st.write(":white_check_mark: libpcap-dev installed.")

            if not os.path.exists("CICFlowMeter-3.0"):
                st.write(":arrow_down: Downloading CICFlowMeter-3.0.zip...")
                url = "https://codeberg.org/iortega/TCPDUMP_and_CICFlowMeter/archive/master:CICFlowMeters/CICFlowMeter-3.0.zip"
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open("CICFlowMeter-3.0.zip", "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.write(":white_check_mark: CICFlowMeter-3.0.zip downloaded.")

                st.write(":open_file_folder: Extracting CICFlowMeter-3.0...")
                subprocess.run(["unzip", "-o", "CICFlowMeter-3.0.zip", "-d", "CICFlowMeter-3.0"], check=True, capture_output=True, text=True)
                st.write(":white_check_mark: CICFlowMeter extracted.")

                st.write(":wrench: Configuring executable permission...")
                subprocess.run(["chmod", "+x", "CICFlowMeter-3.0/tcpdump_and_cicflowmeter/bin/CICFlowMeter"], check=True, capture_output=True, text=True)
                st.write(":white_check_mark: Permission configured")

                st.write(":wastebasket: Clearing .zip file...")
                subprocess.run(["rm", "CICFlowMeter-3.0.zip"], check=True, capture_output=True, text=True)
                st.write(":white_check_mark: CICFlowMeter-3.0.zip Cleared")
            else:
                st.write(":information_source: CICFlowMeter-3.0 existed. Skipping...")

            st.write(":file_folder: Creating data/in data/out directories...")
            os.makedirs("data/in", exist_ok=True)
            os.makedirs("data/out", exist_ok=True)
            st.write(":white_check_mark: Directories created.")

            status.update(label=":white_check_mark: CICFlowMeter Setup Complete!", state="complete", expanded=False)

        except subprocess.CalledProcessError as e:
            st.error(f":x: Error during CICFlowMeter setup. Command '{e.cmd}' returned non-zero exit status {e.returncode}. Output: {e.stdout}\nError: {e.stderr}")
            status.update(label=":x: CICFlowMeter Setup Failed", state="error", expanded=True)
            st.session_state.setup_failed = True
            st.session_state.initial_setup_completed = False
            return False
        except requests.exceptions.RequestException as e:
            st.error(f":x: Error downloading CICFlowMeter: {e}")
            status.update(label=":x: CICFlowMeter Setup Failed", state="error", expanded=True)
            st.session_state.setup_failed = True
            st.session_state.initial_setup_completed = False
            return False
        except Exception as e:
            st.error(f":x: An unexpected error occurred during setup: {e}")
            status.update(label=":x: CICFlowMeter Setup Failed", state="error", expanded=True)
            st.session_state.setup_failed = True
            st.session_state.initial_setup_completed = False
            return False

    # Classification Model setup
    with st.status("Setting up ML Model...", expanded=True, state="running") as status:
        try:
            if not os.path.exists("RandomForest400IntPortCIC1718-2.pkl"):
                st.write(":hugging_face: Downloading ML model...")
                model_url = "https://huggingface.co/Mc4minta/RandomForest400IntPortCIC1718/resolve/main/RandomForest400IntPortCIC1718-2.pkl"
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                with open("RandomForest400IntPortCIC1718-2.pkl", "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.write(":white_check_mark: ML Model downloaded.")

            status.update(label=":white_check_mark: ML Model Setup Complete", state="complete", expanded=False)
        except requests.exceptions.RequestException as e:
            st.error(f":x: Error downloading ML Model: {e}")
            status.update(label=":x: ML Model Setup Failed", state="error", expanded=True)
            st.session_state.setup_failed = True
            st.session_state.initial_setup_completed = False
            return False
        except Exception as e:
            st.error(f"An unexpected error occurred during ML Model setup: {e}")
            status.update(label=":x: ML Model Setup Failed", state="error", expanded=True)
            st.session_state.setup_failed = True
            st.session_state.initial_setup_completed = False
            return False

    # Sample PCAP setup
    with st.status("Downloading sample pcap...", expanded=True, state="running") as status:
        try:
            if not os.path.exists("sample_pcap.zip"):
                st.write(":hugging_face: Downloading sample_pcap.zip...")
                zip_url = "https://huggingface.co/Mc4minta/RandomForest400IntPortCIC1718/resolve/main/sample_pcap.zip"
                response = requests.get(zip_url, stream=True)
                response.raise_for_status()
                with open("sample_pcap.zip", "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.write(":white_check_mark: sample_pcap.zip Downloaded.")

            if not os.path.exists("sample_pcap") or not os.listdir("sample_pcap"):
                os.makedirs("sample_pcap", exist_ok=True)

                st.write(":open_file_folder: Extracting sample_pcap...")
                subprocess.run(["unzip", "-o", "sample_pcap.zip", "-d", "."], check=True, capture_output=True, text=True)
                st.write(":white_check_mark: sample_pcap extracted.")

                if os.path.exists("sample_pcap") and os.listdir("sample_pcap"):
                    pcap_files = [f for f in os.listdir("sample_pcap") if f.endswith('.pcap')]
                    st.write(f":information_source: Found {len(pcap_files)} .pcap files in sample_pcap directory.")
                else:
                    st.warning(":warning: sample_pcap directory appears to be empty after extraction.")
            else:
                pcap_files = [f for f in os.listdir("sample_pcap") if f.endswith('.pcap')]
                st.write(f":information_source: Sample PCAP directory already exists with {len(pcap_files)} .pcap files. Skipping extraction.")

            status.update(label=":white_check_mark: Sample Data Setup Complete", state="complete", expanded=False)
            return True

        except subprocess.CalledProcessError as e:
            st.error(f":x: Error extracting sample data: Command '{e.cmd}' returned {e.returncode}. Output: {e.stdout}\nError: {e.stderr}")
            status.update(label=":x: Sample Data Extraction Failed", state="error", expanded=True)
            st.session_state.setup_failed = True
            st.session_state.initial_setup_completed = False
            return False
        except requests.exceptions.RequestException as e:
            st.error(f":x: Error downloading sample data: {e}")
            status.update(label=":x: Sample Data Setup Failed", state="error", expanded=True)
            st.session_state.setup_failed = True
            st.session_state.initial_setup_completed = False
            return False
        except Exception as e:
            st.error(f":x: An unexpected error occurred during sample data setup: {e}")
            status.update(label=":x: Sample Data Setup Failed", state="error", expanded=True)
            st.session_state.setup_failed = True
            st.session_state.initial_setup_completed = False
            return False

@st.cache_data(show_spinner=False)
def initial_setup_cached():
    if st.session_state.get('initial_setup_completed', False) and not st.session_state.get('setup_failed', False):
        return True
    
    st.session_state.initial_setup_completed = False
    st.session_state.setup_failed = False
    
    success = display_setup_logs()
    
    st.session_state.initial_setup_completed = success
    st.session_state.setup_failed = not success
    st.session_state.show_setup_logs = True
    return success

def clear_uploaded_files():
    # remove pcap files
    if os.path.exists("data/in"):
        for filename in os.listdir("data/in"):
            filepath = os.path.join("data/in", filename)
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
            except Exception as e:
                st.error(f"Failed to delete {filename} from data/in: {e}")
    # remove csv files
    if os.path.exists("data/out"):
        for filename in os.listdir("data/out"):
            filepath = os.path.join("data/out", filename)
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
            except Exception as e:
                st.error(f"Failed to delete {filename} from data/out: {e}")

# --- Prediction Pipeline (YOUR APPROACH) ---
def run_prediction_pipeline(pcap_file_path, uploaded_pcap_name): # Removed 'model' parameter
    # Load the ML model *here*, just before it's used for prediction
    model_path = 'RandomForest400IntPortCIC1718-2.pkl'
    model = None # Initialize model to None
    try:
        if not os.path.exists(model_path):
            st.error(f"ML model file not found at {model_path}. Please ensure setup completed successfully.")
            return None, 0, None # Indicate failure if file isn't there
        st.status(":robot_face: Loading ML model...", state="running") # Optional status update
        model = joblib.load(model_path)
        st.status(":white_check_mark: ML Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading ML model: {e}. Cannot run prediction.")
        return None, 0, None # Indicate failure

    # --- Rest of your run_prediction_pipeline function ---
    data_in_dir = 'data/in/'
    data_out_dir = 'data/out/'

    file_name_without_ext = os.path.splitext(uploaded_pcap_name)[0]
    index_file_path = f'{data_out_dir}{file_name_without_ext}_index.csv'
    flow_file_path = f'{data_out_dir}{file_name_without_ext}_ISCX.csv'
    prediction_file_path = f'{data_out_dir}{file_name_without_ext}_prediction.csv'
    prompt_file_path = f'{data_out_dir}{file_name_without_ext}_prompt.csv' # This is the key file for LLM

    try:
        with st.status("Running analysis pipeline...", expanded=True) as pipeline_status:
            pipeline_status.write(":mag: Starting analysis...")

            # Step 1: Run CICFlowMeter to generate flow features
            pipeline_status.write(":rocket: Running CICFlowMeter to generate flow features...")
            subprocess.run("CICFlowMeter-3.0/tcpdump_and_cicflowmeter/bin/CICFlowMeter", check=True, capture_output=True, text=True)
            pipeline_status.write(":white_check_mark: CICFlowMeter finished.")

            # Step 2: Simulate flow with packet indices
            pipeline_status.write(":chart_with_upwards_trend: Extracting flows and simulating packet indices...")
            # Assuming extract_flows_from_pcap is defined elsewhere
            simulated_flows_df, total_packets = extract_flows_from_pcap(pcap_file_path)
            simulated_flows_df['Total_Packets'] = total_packets
            simulated_flows_df.to_csv(index_file_path, index=False)
            pipeline_status.write(f":white_check_mark: Simulated flows extracted. Total packets: {total_packets}")

            # Step 3: Merge packet indices flow with original flow
            pipeline_status.write(":handshake: Merging simulated and original flow data...")
            # Assuming read_flows_to_dataframe and merge_flows_and_return_dataframe are defined elsewhere
            simulated_df, _ = read_flows_to_dataframe(index_file_path, is_simulated_output=True)
            original_df, _ = read_flows_to_dataframe(flow_file_path, is_simulated_output=False)

            merged_df = merge_flows_and_return_dataframe(simulated_df, original_df)
            merged_df['Total_Packets'] = total_packets
            merged_df.to_csv(index_file_path, index=False)
            pipeline_status.write(":white_check_mark: Flow data merged successfully.")

            # --- START OF PREDICTION PIPELINE (integrated) ---
            # 1. Extract Flow-level Metadata
            flow_metadata_cols = [
                'Simulated Packet Indices',
                'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol',
                'Fwd Seg Size Min', 'Init Fwd Win Byts', 'Bwd Pkts/s', 'Flow IAT Max',
                'Flow Duration', 'Pkt Len Mean', 'Flow Pkts/s', 'Fwd Header Len',
                'TotLen Fwd Pkts', 'Pkt Size Avg', 'Init Bwd Win Byts', 'Flow IAT Mean',
                'Subflow Fwd Byts', 'Bwd Pkt Len Mean', 'Bwd Header Len',
                'Bwd Seg Size Avg', 'PSH Flag Cnt', 'Flow Byts/s', 'Fwd Pkts/s'
            ]
            flow_metadata_df = merged_df[flow_metadata_cols].copy()

            # 2. Preprocess Data for Model Prediction
            pipeline_status.write(":gear: Preprocessing data for prediction...")
            # Assuming preprocess_dataframe is defined elsewhere
            df = preprocess_dataframe(merged_df.copy())
            total_packets = df['Total_Packets'].iloc[0] if 'Total_Packets' in df.columns and not df.empty else 0
            if 'Total_Packets' in df.columns:
                df = df.drop(columns=['Total_Packets'])

            original_flow_indices = df.index

            if 'Simulated_Packet_Indices' in df.columns:
                df_for_prediction = df.drop(columns=['Simulated_Packet_Indices'])
            else:
                df_for_prediction = df.copy()
            pipeline_status.write(":white_check_mark: Data preprocessed.")

            # 3. Perform Prediction (per-flow prediction)
            pipeline_status.write(":robot_face: Performing flow-level predictions...")
            flow_predictions = model.predict(df_for_prediction) # Uses the 'model' loaded above
            pipeline_status.write(":white_check_mark: Predictions complete.")

            # 4. Link Predictions with Flow Metadata
            df_flow_level_result = flow_metadata_df.loc[original_flow_indices].copy()
            df_flow_level_result['Label'] = flow_predictions

            # 5. Explode Packet Indices
            df_prediction_expanded = df_flow_level_result.explode('Simulated Packet Indices').reset_index(drop=True)
            df_prediction_expanded = df_prediction_expanded.rename(columns={'Simulated Packet Indices': 'Packet_Indices'})
            df_prediction_expanded['Packet_Indices'] = df_prediction_expanded['Packet_Indices'].astype(int)

            # 6. Group by Packet Indices and Aggregate Features
            pipeline_status.write(":clipboard: Aggregating features by packet index...")
            # Assuming choose_label is defined elsewhere
            final_df_prediction = df_prediction_expanded.groupby('Packet_Indices').agg(
                Label=('Label', lambda x: choose_label(list(x))),
                Source_IP=('Src IP', 'first'),
                Destination_IP=('Dst IP', 'first'),
                Source_Port=('Src Port', 'first'),
                Destination_Port=('Dst Port', 'first'),
                Protocol=('Protocol', 'first'),
                Fwd_Seg_Size_Min=('Fwd Seg Size Min', 'first'),
                Init_Fwd_Win_Byts=('Init Fwd Win Byts', 'first'),
                Bwd_Pkts_s=('Bwd Pkts/s', 'first'),
                Flow_IAT_Max=('Flow IAT Max', 'first'),
                Flow_Duration=('Flow Duration', 'first'),
                Pkt_Len_Mean=('Pkt Len Mean', 'first'),
                Flow_Pkts_s=('Flow Pkts/s', 'first'),
                Fwd_Header_Len=('Fwd Header Len', 'first'),
                TotLen_Fwd_Pkts=('TotLen Fwd Pkts', 'first'),
                Pkt_Size_Avg=('Pkt Size Avg', 'first'),
                Init_Bwd_Win_Byts=('Init Bwd Win Byts', 'first'),
                Flow_IAT_Mean=('Flow IAT Mean', 'first'),
                Subflow_Fwd_Byts=('Subflow Fwd Byts', 'first'),
                Bwd_Pkt_Len_Mean=('Bwd Pkt Len Mean', 'first'),
                Bwd_Header_Len=('Bwd Header Len', 'first'),
                Bwd_Seg_Size_Avg=('Bwd Seg Size Avg', 'first'),
                PSH_Flag_Cnt=('PSH Flag Cnt', 'first'),
                Flow_Byts_s=('Flow Byts/s', 'first'),
                Fwd_Pkts_s=('Fwd Pkts/s', 'first')
            ).reset_index()
            pipeline_status.write(":white_check_mark: Aggregation complete.")

            for col in [
                'Source_Port', 'Destination_Port', 'Protocol',
                'Fwd_Seg_Size_Min', 'Init_Fwd_Win_Byts', 'Flow_Duration',
                'Fwd_Header_Len', 'TotLen_Fwd_Pkts', 'Init_Bwd_Win_Byts',
                'Subflow_Fwd_Byts', 'Bwd_Header_Len', 'PSH_Flag_Cnt'
            ]:
                final_df_prediction[col] = final_df_prediction[col].astype('Int64')

            # 7. Adjust Packet Indices to be 1-based
            final_df_prediction['Packet_Indices'] = final_df_prediction['Packet_Indices'] + 1
            final_df_prediction['Packet_Indices'] = final_df_prediction['Packet_Indices'].astype(int)

            # 8. Handle Missing Indices and Add Features
            full_indices = set(range(1, total_packets + 1))
            existing_indices = set(final_df_prediction['Packet_Indices'])
            missing_indices = sorted(list(full_indices - existing_indices))

            df_missing = pd.DataFrame({
                'Packet_Indices': missing_indices,
                'Label': ['Benign'] * len(missing_indices),
                'Source_IP': [np.nan] * len(missing_indices),
                'Destination_IP': [np.nan] * len(missing_indices),
                'Source_Port': [pd.NA] * len(missing_indices),
                'Destination_Port': [pd.NA] * len(missing_indices),
                'Protocol': [pd.NA] * len(missing_indices),
                'Fwd_Seg_Size_Min': [pd.NA] * len(missing_indices),
                'Init_Fwd_Win_Byts': [pd.NA] * len(missing_indices),
                'Bwd_Pkts_s': [np.nan] * len(missing_indices),
                'Flow_IAT_Max': [np.nan] * len(missing_indices),
                'Flow_Duration': [pd.NA] * len(missing_indices),
                'Pkt_Len_Mean': [np.nan] * len(missing_indices),
                'Flow_Pkts_s': [np.nan] * len(missing_indices),
                'Fwd_Header_Len': [pd.NA] * len(missing_indices),
                'TotLen_Fwd_Pkts': [pd.NA] * len(missing_indices),
                'Pkt_Size_Avg': [np.nan] * len(missing_indices),
                'Init_Bwd_Win_Byts': [pd.NA] * len(missing_indices),
                'Flow_IAT_Mean': [np.nan] * len(missing_indices),
                'Subflow_Fwd_Byts': [pd.NA] * len(missing_indices),
                'Bwd_Pkt_Len_Mean': [np.nan] * len(missing_indices),
                'Bwd_Header_Len': [pd.NA] * len(missing_indices),
                'Bwd_Seg_Size_Avg': [np.nan] * len(missing_indices),
                'PSH_Flag_Cnt': [pd.NA] * len(missing_indices),
                'Flow_Byts_s': [np.nan] * len(missing_indices),
                'Fwd_Pkts_s': [np.nan] * len(missing_indices),
            })

            for col in [
                'Source_Port', 'Destination_Port', 'Protocol',
                'Fwd_Seg_Size_Min', 'Init_Fwd_Win_Byts', 'Flow_Duration',
                'Fwd_Header_Len', 'TotLen_Fwd_Pkts', 'Init_Bwd_Win_Byts',
                'Subflow_Fwd_Byts', 'Bwd_Header_Len', 'PSH_Flag_Cnt'
            ]:
                df_missing[col] = df_missing[col].astype('Int64')
            df_missing['Packet_Indices'] = df_missing['Packet_Indices'].astype(int)

            # 9. Ensure Consistent Column Order
            final_columns_order = [
                'Packet_Indices', 'Label', 'Source_IP', 'Destination_IP',
                'Source_Port', 'Destination_Port', 'Protocol',
                'Fwd_Seg_Size_Min', 'Init_Fwd_Win_Byts', 'Bwd_Pkts_s', 'Flow_IAT_Max',
                'Flow_Duration', 'Pkt_Len_Mean', 'Flow_Pkts_s', 'Fwd_Header_Len',
                'TotLen_Fwd_Pkts', 'Pkt_Size_Avg', 'Init_Bwd_Win_Byts', 'Flow_IAT_Mean',
                'Subflow_Fwd_Byts', 'Bwd_Pkt_Len_Mean', 'Bwd_Header_Len',
                'Bwd_Seg_Size_Avg', 'PSH_Flag_Cnt', 'Flow_Byts_s', 'Fwd_Pkts_s'
            ]
            final_df_prediction = final_df_prediction.reindex(columns=final_columns_order)
            df_missing = df_missing.reindex(columns=final_columns_order)

            # 10. Concatenate and Sort Final DataFrame
            final_df_prediction = pd.concat([final_df_prediction, df_missing], ignore_index=True)
            final_df_prediction.sort_values(by='Packet_Indices', inplace=True)
            final_df_prediction.reset_index(drop=True, inplace=True)

            final_df_prediction.to_csv(prompt_file_path, index=False) # This is the file with all details for LLM

            columns_to_keep_for_display = [
                'Packet_Indices',
                'Label',
                'Source_IP',
                'Destination_IP',
                'Source_Port',
                'Destination_Port',
                'Protocol',
            ]
            display_df = final_df_prediction[columns_to_keep_for_display].copy()
            display_df.to_csv(prediction_file_path, index=False)

            pipeline_status.update(label=":white_check_mark: Analysis Complete!", state="complete", expanded=False)
            return display_df, total_packets, prompt_file_path # Return prompt_file_path
    except subprocess.CalledProcessError as e:
        st.error(f":x: Pipeline Error during command execution: {e.cmd} returned {e.returncode}. Output: {e.stdout}\nError: {e.stderr}")
        return None, 0, None
    except Exception as e:
        st.error(f":x: An unexpected error occurred during the analysis pipeline: {e}")
        return None, 0, None


# --- Function to generate the initial prompt for a single flow (for LLM) ---
def create_gemini_prompt_for_streamlit(df_row):
    prompt = f"""
I have a network flow extracted from a .pcap file using CICFlowMeter, predicted by a machine learning model.
This flow was classified as: **{df_row['Label']}**.

Here are the flow details:
- **Packet Index**: {df_row['Packet_Indices']}
- **Source IP**: {df_row['Source_IP']}
- **Destination IP**: {df_row['Destination_IP']}
- **Destination Port**: {df_row['Destination_Port']}
- **Protocol**: {df_row['Protocol']}
- **Minimum segment size observed in the forward direction**: {df_row['Fwd_Seg_Size_Min']} bytes
- **Total initial window bytes in forward direction**: {df_row['Init_Fwd_Win_Byts']} bytes
- **Backward packets per second**: {df_row['Bwd_Pkts_s']}
- **Maximum time between two packets (Flow IAT Max)**: {df_row['Flow_IAT_Max']} ms
- **Flow Duration**: {df_row['Flow_Duration']} ms
- **Mean packet length**: {df_row['Pkt_Len_Mean']} bytes
- **Flow packets per second**: {df_row['Flow_Pkts_s']}
- **Forward header length**: {df_row['Fwd_Header_Len']} bytes
- **Total length of forward packets**: {df_row['TotLen_Fwd_Pkts']} bytes
- **Average packet size**: {df_row['Pkt_Size_Avg']} bytes
- **Total initial window bytes in backward direction**: {df_row['Init_Bwd_Win_Byts']} bytes
- **Mean time between two packets (Flow IAT Mean)**: {df_row['Flow_IAT_Mean']} ms
- **Average forward subflow bytes**: {df_row['Subflow_Fwd_Byts']} bytes
- **Mean backward packet length**: {df_row['Bwd_Pkt_Len_Mean']} bytes
- **Backward header length**: {df_row['Bwd_Header_Len']} bytes
- **Average segment size in backward direction**: {df_row['Bwd_Seg_Size_Avg']} bytes
- **PSH Flag Count**: {df_row['PSH_Flag_Cnt']}
- **Flow bytes per second**: {df_row['Flow_Byts_s']}
- **Forward packets per second**: {df_row['Fwd_Pkts_s']}

Given these details, please explain:
1.  **Why** might the model classify this flow as **{df_row['Label']}**? Elaborate on the features that strongly suggest this classification.
2.  What **suspicious behaviors** or **flow characteristics** directly support this classification, if any?
3.  What **insights** does this specific prediction provide about the network activity?
4.  What parts of the original **.pcap file** (e.g., specific filters to apply in Wireshark/tcpdump, packet types) should I examine further to confirm or understand this flow better?
5.  Based on this classification, what are the **immediate next steps** for investigation or mitigation from a cybersecurity perspective?
"""
    return prompt.strip()

# --- Streamlit Main Function ---
def main():
    # 1. Set page config FIRST
    st.set_page_config(
        page_title="Malicious .PCAP File Classifier",
        page_icon=":peacock:",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # 2. Then perform API key check and Gemini model initialization
    # It's better to fetch the API key here
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("🚨 GOOGLE_API_KEY environment variable not set.")
        st.warning("Please set it before running the app. E.g.:")
        st.code("export GOOGLE_API_KEY='YOUR_API_KEY'  # For Linux terminal")
        st.code("os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY' # For Google Colab cell")
        st.info("The application cannot proceed without the API key.")
        st.stop() # Halt execution if API key is missing

    # Initialize Gemini model (this will be called once per session due to st.cache_resource)
    # Pass the fetched api_key
    gemini_model = configure_gemini(api_key) 
    if gemini_model is None:
        # configure_gemini already displayed error, so just stop
        st.info("Gemini model could not be initialized. Please check your API key and connection.")
        st.stop() # Halt execution if Gemini model couldn't be configured

    # Initialize session states (these can be after set_page_config)
    if 'initial_setup_completed' not in st.session_state:
        st.session_state.initial_setup_completed = False
    if 'setup_failed' not in st.session_state:
        st.session_state.setup_failed = False
    if 'show_setup_logs' not in st.session_state:
        st.session_state.show_setup_logs = False
    if 'proceed_clicked' not in st.session_state:
        st.session_state.proceed_clicked = False
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'file_selected_successfully' not in st.session_state:
        st.session_state.file_selected_successfully = False
    if 'selected_filename' not in st.session_state:
        st.session_state.selected_filename = None
    if 'prediction_results_df' not in st.session_state:
        st.session_state.prediction_results_df = None
    if 'total_packets' not in st.session_state:
        st.session_state.total_packets = 0
    if 'prompt_csv_path' not in st.session_state: # New: Store path to the full data for LLM
        st.session_state.prompt_csv_path = None
    
    # LLM specific states
    if 'llm_chat_history' not in st.session_state:
        st.session_state.llm_chat_history = {} # Stores chat history per Packet_Indices
    if 'current_llm_flow_index' not in st.session_state:
        st.session_state.current_llm_flow_index = None
    if 'show_llm_chat' not in st.session_state:
        st.session_state.show_llm_chat = False


    st.markdown("""
        <h1 style='text-align: center; color: #b5213b;'>
            AI Builders 2025
        </h1> 
        <h1 style='text-align: center;'>
            Malicious <span style='color: #074ec0;'>.pcap</span> File Classifier
        </h1>
        <h3 style='text-align: center;'><span style='color: #1abc9c;'>
            By: MiN - Vibrant Peacock</span> 🦚
        </h3>
        """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        div.stButton > button {
            display: block;
            margin: 0 auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Setup Logic ---
    # The rest of your main function logic can remain largely the same from here
    if not st.session_state.initial_setup_completed and not st.session_state.setup_failed:
        if st.button("Start Setup"):
            initial_setup_cached()
            st.rerun()

    elif st.session_state.setup_failed:
        st.warning("Setup failed. Please try again.")
        if st.button("Start Setup"):
            initial_setup_cached()
            st.rerun()

    if st.session_state.initial_setup_completed and not st.session_state.proceed_clicked:
        st.success(":tada: Setup Completed")
        if st.button("Proceed"):
            st.session_state.show_setup_logs = False
            st.session_state.proceed_clicked = True
            st.rerun()

    # --- Main Application Logic (after setup and proceed) ---
    if st.session_state.initial_setup_completed and st.session_state.proceed_clicked:
        if not st.session_state.show_results:
            st.info(":file_folder: Choose a file for analysis.")

            # --- File Selection Options ---
            selection_method = st.radio(
                "How would you like to provide the PCAP file?",
                ("Upload a PCAP file", "Choose from Sample Data"),
                key="selection_method"
            )

            pcap_to_analyze_bytes = None # Renamed to clearly indicate bytes data
            pcap_filename_for_analysis = None # Renamed to clearly indicate filename for the pipeline

            if selection_method == "Upload a PCAP file":
                uploaded_file = st.file_uploader(
                    "Choose a PCAP file", accept_multiple_files=False, type=["pcap"]
                )
                if uploaded_file is not None:
                    pcap_to_analyze_bytes = uploaded_file.read()
                    pcap_filename_for_analysis = uploaded_file.name

            elif selection_method == "Choose from Sample Data":
                sample_data_dir = "sample_pcap"
                if os.path.exists(sample_data_dir):
                    pcap_files = [f for f in os.listdir(sample_data_dir) if f.endswith('.pcap')]
                    if pcap_files:
                        selected_sample_file = st.selectbox(
                            "Select a sample PCAP file:",
                            ["-- Select a file --"] + sorted(pcap_files),
                            key="selected_sample_file"
                        )
                        if selected_sample_file != "-- Select a file --":
                            pcap_filename_for_analysis = selected_sample_file
                            sample_file_path = os.path.join(sample_data_dir, pcap_filename_for_analysis)
                            with open(sample_file_path, "rb") as f:
                                pcap_to_analyze_bytes = f.read()
                    else:
                        st.warning("No .pcap files found in the 'sample_pcap' folder.")
                else:
                    st.error("The 'sample_pcap' folder does not exist. Please create it and add .pcap files.")

            # --- Analysis Button ---
            if pcap_to_analyze_bytes is not None and pcap_filename_for_analysis is not None:
                if st.button("Start Analysis", key="start_analysis_button"):
                    # Save the chosen PCAP (uploaded or sample) to data/in
                    os.makedirs('data/in', exist_ok=True)
                    target_pcap_path = os.path.join("data/in", pcap_filename_for_analysis)
                    try:
                        with open(target_pcap_path, "wb") as f:
                            f.write(pcap_to_analyze_bytes) # Use the bytes data here
                        mb_size = len(pcap_to_analyze_bytes) / (1024 * 1024)
                        st.session_state.selected_filename = pcap_filename_for_analysis
                        st.success(f":file_folder: '{pcap_filename_for_analysis}' size {mb_size:.2f} MB selected. Starting analysis...")
                        st.session_state.file_selected_successfully = True
                        
                        results_df, total_p, prompt_csv_path = run_prediction_pipeline(target_pcap_path, pcap_filename_for_analysis)
                        if results_df is not None:
                            st.session_state.prediction_results_df = results_df
                            st.session_state.total_packets = total_p
                            st.session_state.prompt_csv_path = prompt_csv_path
                            st.session_state.show_results = True
                            st.rerun()
                        else:
                            st.error("Analysis failed. Please check the logs above.")
                            st.session_state.file_selected_successfully = False
                            st.session_state.show_results = False
                            clear_uploaded_files()

                    except Exception as e:
                        st.error(f"Error during file processing or analysis: {e}")
                        st.session_state.file_selected_successfully = False
                        st.session_state.show_results = False
                        clear_uploaded_files()

        # Display results once available
        if st.session_state.show_results and st.session_state.prediction_results_df is not None:
            selected_filename = st.session_state.selected_filename
            total_packets = st.session_state.total_packets
            st.subheader(f"Analysis Results for '{selected_filename}'")
            st.info(f"Total packets analyzed: **{total_packets}**")
            
            prediction_df = st.session_state.prediction_results_df
            
            if "show_df" not in st.session_state:
                st.session_state.show_df = False
            if "show_predictions" not in st.session_state:
                st.session_state.show_predictions = True

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Show Packet-Level Details"):
                    st.session_state.show_df = True
                    st.session_state.show_predictions = False
                    st.session_state.show_llm_chat = False # Hide LLM when showing details
            with col2:
                if st.button("Show Prediction Summary"):
                    st.session_state.show_df = False
                    st.session_state.show_predictions = True
                    st.session_state.show_llm_chat = False # Hide LLM when showing summary

            if st.session_state.show_predictions:
                st.write("### Prediction Summary (Packet Count by Label)")
                if not prediction_df.empty:
                    prediction_counts = prediction_df['Label'].value_counts().sort_index()
                    st.bar_chart(prediction_counts)
                    st.dataframe(prediction_counts.reset_index().rename(columns={'index': 'Label', 'Label': 'Count'}), use_container_width=True)
                else:
                    st.warning("No predictions to display.")
            
            if st.session_state.show_df:
                st.write("### Packet-Level Prediction Details")
                st.dataframe(prediction_df, use_container_width=True)

            output_prediction_file = os.path.join('data', 'out', os.path.splitext(selected_filename)[0] + '_prediction.csv')
            if os.path.exists(output_prediction_file):
                with open(output_prediction_file, "rb") as file:
                    btn = st.download_button(
                        label="Download Packet Predictions (CSV)",
                        data=file,
                        file_name=os.path.basename(output_prediction_file),
                        mime="text/csv",
                        help="Download the CSV file containing packet-level predictions."
                    )
            
            st.markdown("---")
            st.subheader("🤖 LLM-Powered Flow Analysis")

            # Load the full prompt data for LLM if not already loaded
            if 'full_prompt_df' not in st.session_state and st.session_state.prompt_csv_path:
                try:
                    st.session_state.full_prompt_df = pd.read_csv(st.session_state.prompt_csv_path)
                except FileNotFoundError:
                    st.error(f"LLM data file not found: {st.session_state.prompt_csv_path}")
                    st.session_state.full_prompt_df = pd.DataFrame()
                except Exception as e:
                    st.error(f"Error loading LLM data: {e}")
                    st.session_state.full_prompt_df = pd.DataFrame()

            if not st.session_state.get('full_prompt_df', pd.DataFrame()).empty:
                unique_labels = st.session_state.full_prompt_df['Label'].unique().tolist()
                label_filter = st.multiselect(
                    "Filter flows by Label for LLM analysis:",
                    options=["All"] + sorted(unique_labels),
                    default=["All"],
                    key="llm_label_filter"
                )

                filtered_llm_flows = st.session_state.full_prompt_df.copy()
                if "All" not in label_filter:
                    filtered_llm_flows = filtered_llm_flows[filtered_llm_flows['Label'].isin(label_filter)]
                
                # Exclude Benign from default selection if not explicitly chosen
                if "All" in label_filter:
                    filtered_llm_flows_options = filtered_llm_flows
                else:
                    filtered_llm_flows_options = filtered_llm_flows[filtered_llm_flows['Label'] != 'Benign']

                # Display a dropdown for Packet_Indices of non-benign flows
                if not filtered_llm_flows_options.empty:
                    # Create options with more descriptive labels
                    llm_flow_options = [
                        f"Packet {row['Packet_Indices']} (Label: {row['Label']}, Src: {row['Source_IP']}, Dst: {row['Destination_IP']}:{row['Destination_Port']})"
                        for index, row in filtered_llm_flows_options.sort_values(by='Packet_Indices').iterrows()
                    ]
                    
                    selected_flow_option = st.selectbox(
                        "Select a flow to get LLM insights:",
                        options=["-- Select a flow --"] + llm_flow_options,
                        key="llm_flow_selection"
                    )

                    # Extract Packet_Indices from the selected option
                    selected_packet_index = None
                    if selected_flow_option != "-- Select a flow --":
                        selected_packet_index = int(selected_flow_option.split(" (Label:")[0].replace("Packet ", ""))
                        
                    if selected_packet_index:
                        # Get the row corresponding to the selected packet index
                        selected_flow_row = st.session_state.full_prompt_df[
                            st.session_state.full_prompt_df['Packet_Indices'] == selected_packet_index
                        ].iloc[0]

                        # Ensure the chat history for this flow is initialized
                        if selected_packet_index not in st.session_state.llm_chat_history:
                            st.session_state.llm_chat_history[selected_packet_index] = []
                            # Automatically generate initial explanation if not already done
                            with st.spinner(f"Generating initial analysis for Packet {selected_packet_index}..."):
                                initial_prompt_text = create_gemini_prompt_for_streamlit(selected_flow_row)
                                try:
                                    # Use the gemini_model object from main's scope
                                    response = gemini_model.generate_content(initial_prompt_text)
                                    st.session_state.llm_chat_history[selected_packet_index].append({"role": "user", "content": initial_prompt_text})
                                    st.session_state.llm_chat_history[selected_packet_index].append({"role": "model", "content": response.text})
                                    st.session_state.current_llm_flow_index = selected_packet_index
                                    st.session_state.show_llm_chat = True
                                    # Rerun to show the initial chat immediately
                                    st.rerun() 
                                except Exception as e:
                                    st.error(f"Error generating initial LLM response: {e}")
                                    st.session_state.show_llm_chat = False
                        
                        st.session_state.current_llm_flow_index = selected_packet_index
                        st.session_state.show_llm_chat = True

                        if st.session_state.show_llm_chat and st.session_state.current_llm_flow_index == selected_packet_index:
                            st.write(f"### Chat for Packet {selected_packet_index} ({selected_flow_row['Label']} traffic)")
                            
                            # Display chat messages from history
                            for message in st.session_state.llm_chat_history[selected_packet_index]:
                                with st.chat_message(message["role"]):
                                    st.markdown(message["content"])

                            # Chat input for follow-up questions
                            prompt_chat_input = st.chat_input("Ask a follow-up question about this flow:")
                            if prompt_chat_input:
                                st.session_state.llm_chat_history[selected_packet_index].append({"role": "user", "content": prompt_chat_input})
                                with st.chat_message("user"):
                                    st.markdown(prompt_chat_input)

                                with st.chat_message("model"):
                                    with st.spinner("Thinking..."):
                                        try:
                                            # Reconstruct history for current chat session
                                            current_chat_history = [
                                                {"role": "user", "parts": [msg["content"]]} if msg["role"] == "user" else {"role": "model", "parts": [msg["content"]]}
                                                for msg in st.session_state.llm_chat_history[selected_packet_index] if msg["role"] in ["user", "model"]
                                            ]
                                            
                                            # Start a new chat with the current history
                                            chat_session = gemini_model.start_chat(history=current_chat_history[:-1]) # Exclude the latest user message from history for send_message
                                            
                                            full_response = chat_session.send_message(prompt_chat_input)
                                            st.markdown(full_response.text)
                                            st.session_state.llm_chat_history[selected_packet_index].append({"role": "model", "content": full_response.text})
                                            
                                        except Exception as e:
                                            st.error(f"Error communicating with LLM: {e}")
                                            st.session_state.llm_chat_history[selected_packet_index].append({"role": "model", "content": f"Error: {e}"}) # Add error to history
                                st.rerun() # Rerun to display new message
                    else:
                        st.session_state.show_llm_chat = False # Hide chat if no flow is selected
                        st.session_state.current_llm_flow_index = None
                else:
                    st.info("No flows matching the selected labels to analyze with LLM.")
            else:
                st.warning("No flow data available for LLM analysis. Please ensure a PCAP file was analyzed successfully.")

            # --- Bottom navigation buttons ---
            st.markdown("---")
            if st.button("Analyze another file", key="analyze_another_file_button"):
                clear_uploaded_files()
                # Clear all relevant session states for a fresh start
                for key in list(st.session_state.keys()):
                    if key.startswith(('show_', 'file_selected', 'selected_filename', 
                                       'prediction_results_df', 'total_packets', 'prompt_csv_path',
                                       'llm_')): # Clear LLM related states too
                        del st.session_state[key]
                st.session_state.proceed_clicked = True # Keep this true if we want to immediately go to file selection
                st.rerun()


if __name__ == "__main__":
    main()