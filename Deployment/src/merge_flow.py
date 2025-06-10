# define function for combining simulated and original flow

import pandas as pd
import ipaddress # For robust IP address comparison

# --- Start: Helper Functions (from previous scripts, for self-contained use) ---

def generate_flow_key(packet_components):
    """
    Generates a unique key for a flow based on the 5-tuple,
    mimicking CICFlowMeter's Java BasicPacketInfo.generateFlowId() logic
    for canonicalizing IP addresses and ports.

    This function uses the raw protocol number, matching BasicPacketInfo.java.
    Any mapping of non-TCP/UDP/ICMP protocols to '0' happens *after* Flow ID generation
    in CICFlowMeter's pipeline (e.g., for display in FlowFeature.featureValue2String),
    and is NOT part of the Flow ID itself.

    Args:
        packet_components (tuple): A tuple containing (src_ip_str, dst_ip_str, src_port, dst_port, protocol_int)
    Returns:
        tuple: Canonical 5-tuple (normalized_src_ip, normalized_dst_ip, normalized_src_port, normalized_dst_port, normalized_protocol_int)
    """
    src_ip_str, dst_ip_str, src_port, dst_port, protocol_int = packet_components

    # Use ipaddress for robust IP comparison, mirroring Java's byte-by-byte comparison
    try:
        src_ip_obj = ipaddress.ip_address(src_ip_str)
        dst_ip_obj = ipaddress.ip_address(dst_ip_str)
    except ValueError:
        # Fallback for invalid IPs if any, should not happen with valid PCAP data
        return (src_ip_str, dst_ip_str, src_port, dst_port, protocol_int)


    # Determine 'forward' based on IP comparison: canonical_src_ip will be the "smaller" IP
    if src_ip_obj < dst_ip_obj:
        normalized_src_ip = src_ip_str
        normalized_dst_ip = dst_ip_str
        normalized_src_port = src_port
        normalized_dst_port = dst_port
    elif dst_ip_obj < src_ip_obj:
        # Swap IPs and their corresponding ports for normalization
        normalized_src_ip = dst_ip_str
        normalized_dst_ip = src_ip_str
        normalized_src_port = dst_port
        normalized_dst_port = src_port
    else: # IPs are equal (e.g., multicast or broadcast)
        # If IPs are the same, Java's logic does NOT swap ports based on IP.
        # It keeps original src/dst IPs and ports as they are.
        normalized_src_ip = src_ip_str
        normalized_dst_ip = dst_ip_str
        normalized_src_port = src_port
        normalized_dst_port = dst_port

    # The canonical 5-tuple key for the hash map
    return (normalized_src_ip, normalized_dst_ip, normalized_src_port, normalized_dst_port, protocol_int)

def parse_flow_id_string(flow_id_str):
    """
    Parses a Flow ID string (e.g., 'IP1-IP2-Port1-Port2-Protocol') into its components.
    Returns a tuple (src_ip, dst_ip, src_port, dst_port, protocol_int) or None if parsing fails.
    """
    parts = flow_id_str.split('-')
    if len(parts) == 5:
        try:
            return (parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4]))
        except ValueError:
            return None
    return None

def read_flows_to_dataframe(filepath: str, is_simulated_output: bool = False) -> pd.DataFrame:
    """
    Reads flow data from a CSV file into a Pandas DataFrame.
    Adds a 'Canonical_Flow_ID' column for merging.
    Parses 'Packet Indices' and 'Packet Timestamps' for simulated output.
    This function keeps all original rows and does not deduplicate based on Canonical Flow ID.

    Args:
        filepath (str): Path to the CSV file.
        is_simulated_output (bool): True if reading our simulated output (with 'Packet Indices' column),
                                    False if reading original CICFlowMeter output.

    Returns:
        pd.DataFrame: The loaded DataFrame with an added 'Canonical_Flow_ID' column.
                      Returns an empty DataFrame if the file is not found or an error occurs.
    """
    try:
        df = pd.read_csv(filepath)

        # Create a new column with parsed components for canonicalization
        df['Parsed_Flow_Components'] = df['Flow ID'].apply(parse_flow_id_string)

        # Filter out rows where parsing failed
        df = df.dropna(subset=['Parsed_Flow_Components'])

        # Apply canonicalization to create 'Canonical_Flow_ID'
        df['Canonical_Flow_ID'] = df['Parsed_Flow_Components'].apply(generate_flow_key).apply(lambda x: "-".join(map(str, x)))

        # Clean up temporary column
        df = df.drop(columns=['Parsed_Flow_Components'])

        total_packets = 0
        # Special handling for our simulated output's 'Packet Indices' and 'Packet Timestamps'
        if is_simulated_output:
            if 'Total_Packets' in df.columns:
                total_packets = df['Total_Packets'].iloc[0]
            if 'Packet Indices' in df.columns:
                try:
                    df['Packet Indices'] = df['Packet Indices'].apply(eval)
                except Exception as e:
                    print(f"Warning: Could not parse 'Packet Indices' in {filepath}: {e}")
                    df['Packet Indices'] = [[]] * len(df) # Assign empty list on error
            if 'Packet Timestamps' in df.columns:
                try:
                    df['Packet Timestamps'] = df['Packet Timestamps'].apply(eval)
                except Exception as e:
                    print(f"Warning: Could not parse 'Packet Timestamps' in {filepath}: {e}")
                    df['Packet Timestamps'] = [[]] * len(df) # Assign empty list on error

        print(f"Successfully loaded {len(df)} flows from '{filepath}'.")
        return df,total_packets
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        print(f"Error reading or processing CSV file '{filepath}': {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# --- End: Helper Functions ---

def merge_flows_and_return_dataframe(simulated_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges a DataFrame of simulated flows with a DataFrame of original CICFlowMeter flows
    based on a canonical flow ID. It adds the 'Simulated Packet Indices' column to the
    original flow data where a match is found.

    Args:
        simulated_df (pd.DataFrame): DataFrame containing flows generated by the simulation.
                                     Expected to have 'Canonical_Flow_ID' and 'Packet Indices'.
        original_df (pd.DataFrame): DataFrame containing flows from the original CICFlowMeter.
                                    Expected to have 'Canonical_Flow_ID' and original flow features.

    Returns:
        pd.DataFrame: A merged DataFrame containing original CICFlowMeter features
                      and 'Simulated Packet Indices' for matching flows.
                      Returns an empty DataFrame if inputs are invalid.
    """
    if simulated_df.empty or original_df.empty:
        print("Cannot merge flows due to empty input DataFrames.")
        return pd.DataFrame()

    # Perform a left merge on the 'Canonical_Flow_ID'
    # Keep all rows from original_df, and add matching data from simulated_df.
    # Select only 'Canonical_Flow_ID' and 'Packet Indices' from the simulated_df.
    merged_df = pd.merge(
        original_df,
        simulated_df[['Canonical_Flow_ID', 'Packet Indices']],
        on='Canonical_Flow_ID',
        how='left',
        suffixes=('_original', '_simulated')
    )

    # Rename the new column for clarity
    merged_df = merged_df.rename(columns={
        'Packet Indices': 'Simulated Packet Indices'
    })

    # Drop the 'Canonical_Flow_ID' column, as it's only for merging
    merged_df = merged_df.drop(columns=['Canonical_Flow_ID'])

    print("\n--- Merged Flow Data (Original CICFlowMeter with Simulated Packet Indices) ---")
    print(f"Total rows in merged DataFrame: {len(merged_df)}")
    print("Head of the merged DataFrame:")
    print(merged_df.head())

    print("\nColumns in the merged DataFrame:")
    print(merged_df.columns.tolist())

    return merged_df