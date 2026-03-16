import pandas as pd
import os
import glob

def integrate_dataset():
    base_path = "/home/aziz/Desktop/fatigue_detection/datasets/archive/fatigueset"
    output_dir = "/home/aziz/Desktop/fatigue_detection/datasets"
    output_file = os.path.join(output_dir, "data_selected1.csv")
    
    selected_files = [
        "chest_physiology_summary.csv",
        "exp_fatigue.csv",
        "exp_markers.csv",
        "wrist_acc.csv",
        "wrist_eda.csv",
        "wrist_hr.csv",
        "wrist_ibi.csv",
        "wrist_skin_temperature.csv"
    ]
    
    all_data = []
    
    # Participants 01 to 12
    participants = [f"{i:02d}" for i in range(1, 13)]
    # Sessions 01 to 03
    sessions = [f"{i:02d}" for i in range(1, 4)]
    
    for p_id in participants:
        for s_id in sessions:
            session_path = os.path.join(base_path, p_id, s_id)
            if not os.path.exists(session_path):
                print(f"Skipping {p_id}/{s_id}: path not found")
                continue
            
            print(f"Processing Participant {p_id}, Session {s_id}...")
            
            # Use a dictionary to store dataframes for the current session
            dfs = {}
            
            # Load exp_markers first to get start_experiment timestamp
            markers_file = os.path.join(session_path, "exp_markers.csv")
            if os.path.exists(markers_file):
                df_markers = pd.read_csv(markers_file)
                # Rename utcTime to timestamp for merging
                df_markers = df_markers.rename(columns={'utcTime': 'timestamp'})
                dfs['exp_markers'] = df_markers
                
                # Get start_experiment timestamp
                start_row = df_markers[df_markers['eventMarker'] == 'start_experiment']
                if not start_row.empty:
                    start_ts = start_row.iloc[0]['timestamp']
                else:
                    start_ts = None
            else:
                start_ts = None
            
            for file_name in selected_files:
                if file_name == "exp_markers.csv":
                    continue # Already handled or will be handled
                
                file_path = os.path.join(session_path, file_name)
                if not os.path.exists(file_path):
                    continue
                
                df = pd.read_csv(file_path)
                
                # Special handling for exp_fatigue.csv (relative timestamps)
                if file_name == "exp_fatigue.csv":
                    if start_ts is not None:
                        # calculate timestamp from physicalFatigueAnswerTime (assuming it's seconds since start)
                        # The README doesn't specify unit but head showed float values like 226.38
                        # We'll use fatigueSurveySubmissionTime as the primary timestamp for the survey result
                        df['timestamp'] = (start_ts + (df['fatigueSurveySubmissionTime'] * 1000)).astype(int)
                    else:
                        print(f"Warning: No start_experiment marker for {p_id}/{s_id}, skipping exp_fatigue alignment")
                
                # Add prefixes to columns except timestamp
                cols_to_rename = {col: f"{file_name.split('.')[0]}_{col}" for col in df.columns if col != 'timestamp'}
                df = df.rename(columns=cols_to_rename)
                
                # Ensure timestamp is integer
                if 'timestamp' in df.columns:
                    df['timestamp'] = df['timestamp'].astype(float).astype(int)
                
                dfs[file_name] = df
            
            # Merge all dataframes in the session
            if not dfs:
                continue
                
            session_df = None
            for name, df in dfs.items():
                if session_df is None:
                    session_df = df
                else:
                    # Outer join on timestamp
                    if 'timestamp' in df.columns and 'timestamp' in session_df.columns:
                        session_df = pd.merge(session_df, df, on='timestamp', how='outer')
                    else:
                        # If no timestamp (shouldn't happen with these files according to README), 
                        # just concat or skip? README says all have timestamp.
                        pass
            
            if session_df is not None:
                session_df['participant_id'] = p_id
                session_df['session_id'] = s_id
                all_data.append(session_df)

    if all_data:
        print("Concatenating all sessions...")
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Reorder columns to have ID info first
        cols = ['participant_id', 'session_id', 'timestamp']
        other_cols = [c for c in final_df.columns if c not in cols]
        final_df = final_df[cols + other_cols]
        
        # Sort by participant, session, and timestamp
        final_df = final_df.sort_values(['participant_id', 'session_id', 'timestamp'])
        
        print(f"Saving to {output_file}...")
        final_df.to_csv(output_file, index=False)
        print("Done!")
    else:
        print("No data collected.")

if __name__ == "__main__":
    integrate_dataset()
