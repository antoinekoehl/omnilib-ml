import os
import subprocess
from typing import Tuple, List, Dict
import json
from dataclasses import dataclass
import pandas as pd

@dataclass
class TangoParams:
    """Class to hold TANGO parameters"""
    nt: str = "N"  # C-terminus protection
    ct: str = "N"  # N-terminus protection
    ph: str = "7.4"  # pH
    te: str = "303"  # Temperature in Kelvin
    io: str = "0.05"  # Ionic strength
    tf: str = "0"  # TFE percentage
    stab: str = "-10"  # Protein stability

class NanobodyAggregationPredictor:
    def __init__(self, tango_executable: str, output_dir: str):
        """
        Initialize predictor with path to TANGO executable and output directory
        
        Args:
            tango_executable: Path to TANGO executable
            output_dir: Directory where TANGO output files will be saved
        """
        self.tango_path = tango_executable
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        self.framework_segments = [
            "QVQLQESGGGLVQAGGSLRLSCAASG",  # Framework 1
            "MGWYRQAPGKERE",                # Framework 2
            "YADSVKGRFTISRDNAANTVYLQMNSLKPEDTAVYYC",  # Framework 3
            "WGQGTQVTVSS"                   # Framework 4
        ]
        self.params = TangoParams()
        
    def assemble_sequence(self, cdr1: str, cdr2: str, cdr3: str, 
                          fw1: str = None, fw2: str = None, 
                          fw3: str = None, fw4: str = None) -> str:
        """
        Assemble full nanobody sequence from CDRs and framework segments
        
        Args:
            cdr1: CDR1 sequence
            cdr2: CDR2 sequence
            cdr3: CDR3 sequence
            fw1: Optional Framework 1 override
            fw2: Optional Framework 2 override
            fw3: Optional Framework 3 override
            fw4: Optional Framework 4 override
            
        Returns:
            Complete nanobody sequence
        """
        # Use provided frameworks or defaults
        framework1 = fw1 if fw1 is not None else self.framework_segments[0]
        framework2 = fw2 if fw2 is not None else self.framework_segments[1]
        framework3 = fw3 if fw3 is not None else self.framework_segments[2]
        framework4 = fw4 if fw4 is not None else self.framework_segments[3]
        
        segments = [
            framework1,     # Framework 1
            cdr1,          # CDR1
            framework2,     # Framework 2
            cdr2,          # CDR2
            framework3,     # Framework 3
            cdr3,          # CDR3
            framework4      # Framework 4
        ]
        
        return ''.join(segments)
        
    def create_tango_batch_file(self, sequence: str, identifier: str) -> str:
        """
        Create content for TANGO batch file
        
        Args:
            sequence: Full protein sequence
            identifier: Unique identifier for this run
            
        Returns:
            Content for batch file
        """
        tango_cmd = [
            self.tango_path,
            identifier,
            f'nt="{self.params.nt}"',
            f'ct="{self.params.ct}"',
            f'ph="{self.params.ph}"',
            f'te="{self.params.te}"',
            f'io="{self.params.io}"',
            f'tf="{self.params.tf}"',
            f'stab="{self.params.stab}"',
            f'seq="{sequence}"'
        ]
        return ' '.join(tango_cmd)
        
    def calculate_region_scores(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> Dict:
        """
        Calculate aggregation metrics for a specific region
        
        Args:
            df: DataFrame with TANGO results
            start_idx: Start index of region (1-based)
            end_idx: End index of region (1-based)
            
        Returns:
            Dictionary with region metrics
        """
        region_df = df[(df['position'] >= start_idx) & (df['position'] <= end_idx)]
        
        return {
            'length': len(region_df),
            'total_agg': region_df['agg'].sum(),
            'mean_agg': region_df['agg'].mean(),
            'max_agg': region_df['agg'].max(),
            'normalized_agg': region_df['agg'].sum() / len(region_df),
            'sequence': ''.join(region_df['aa'].tolist())
        }

    def parse_tango_output(self, output_file: str, cdr1: str, cdr2: str, cdr3: str, 
                           fw1: str = None, fw2: str = None, 
                           fw3: str = None, fw4: str = None) -> Dict:
        """
        Parse TANGO output file
        
        Args:
            output_file: Path to TANGO output file
            cdr1: CDR1 sequence
            cdr2: CDR2 sequence
            cdr3: CDR3 sequence
            fw1: Framework 1 sequence used
            fw2: Framework 2 sequence used
            fw3: Framework 3 sequence used
            fw4: Framework 4 sequence used
            
        Returns:
            Dictionary containing aggregation metrics
        """
        # Use provided frameworks or defaults
        fw1 = fw1 if fw1 is not None else self.framework_segments[0]
        fw2 = fw2 if fw2 is not None else self.framework_segments[1]
        fw3 = fw3 if fw3 is not None else self.framework_segments[2]
        fw4 = fw4 if fw4 is not None else self.framework_segments[3]
        
        df = pd.read_csv(output_file, sep='\t', header=0)
        df.columns = ['position', 'aa', 'beta', 'turn', 'helix', 'agg', 'helix_agg']
        
        # Calculate overall metrics
        metrics = {
            'max_agg': df['agg'].max(),
            'mean_agg': df['agg'].mean(),
            'agg_regions': self._find_aggregation_regions(df),
            'total_agg_score': df['agg'].sum(),
            'normalized_agg_score': df['agg'].sum() / len(df),
            'residue_scores': df.to_dict('records'),
            'cdr1': cdr1,
            'cdr2': cdr2,
            'cdr3': cdr3
        }
        
        # Calculate region-specific scores
        current_pos = 1
        
        # Framework 1
        fw1_len = len(fw1)
        metrics['framework1'] = self.calculate_region_scores(df, current_pos, current_pos + fw1_len - 1)
        current_pos += fw1_len
        
        # CDR1
        cdr1_len = len(cdr1)
        metrics['cdr1_scores'] = self.calculate_region_scores(df, current_pos, current_pos + cdr1_len - 1)
        current_pos += cdr1_len
        
        # Framework 2
        fw2_len = len(fw2)
        metrics['framework2'] = self.calculate_region_scores(df, current_pos, current_pos + fw2_len - 1)
        current_pos += fw2_len
        
        # CDR2
        cdr2_len = len(cdr2)
        metrics['cdr2_scores'] = self.calculate_region_scores(df, current_pos, current_pos + cdr2_len - 1)
        current_pos += cdr2_len
        
        # Framework 3
        fw3_len = len(fw3)
        metrics['framework3'] = self.calculate_region_scores(df, current_pos, current_pos + fw3_len - 1)
        current_pos += fw3_len
        
        # CDR3
        cdr3_len = len(cdr3)
        metrics['cdr3_scores'] = self.calculate_region_scores(df, current_pos, current_pos + cdr3_len - 1)
        current_pos += cdr3_len
        
        # Framework 4
        fw4_len = len(fw4)
        metrics['framework4'] = self.calculate_region_scores(df, current_pos, current_pos + fw4_len - 1)
        
        # Calculate overall framework score
        total_fw_agg = (metrics['framework1']['total_agg'] + 
                       metrics['framework2']['total_agg'] + 
                       metrics['framework3']['total_agg'] + 
                       metrics['framework4']['total_agg'])
        total_fw_len = (metrics['framework1']['length'] + 
                       metrics['framework2']['length'] + 
                       metrics['framework3']['length'] + 
                       metrics['framework4']['length'])
        
        metrics['framework_overall'] = {
            'total_agg': total_fw_agg,
            'length': total_fw_len,
            'normalized_agg': total_fw_agg / total_fw_len
        }
        
        return metrics
        
    def _find_aggregation_regions(self, df: pd.DataFrame, threshold: float = 5.0) -> List[Dict]:
        """
        Identify continuous regions with aggregation propensity above threshold
        
        Args:
            df: DataFrame with TANGO results
            threshold: Aggregation score threshold
            
        Returns:
            List of dictionaries containing region information
        """
        regions = []
        current_region = None
        
        for index, row in df.iterrows():
            if row['agg'] >= threshold:
                if current_region is None:
                    current_region = {'start': index + 1, 'sequence': row['aa']}
                else:
                    current_region['sequence'] += row['aa']
            elif current_region is not None:
                current_region['end'] = index
                current_region['score'] = df.loc[current_region['start']-1:index-1, 'agg'].mean()
                regions.append(current_region)
                current_region = None
                
        return regions
        
    def predict(self, cdr1: str, cdr2: str, cdr3: str, identifier: str = "nb",
                  fw1: str = None, fw2: str = None, 
                  fw3: str = None, fw4: str = None) -> Dict:
        """
        Run full prediction pipeline for a single nanobody
        
        Args:
            cdr1: CDR1 sequence
            cdr2: CDR2 sequence
            cdr3: CDR3 sequence
            identifier: Unique identifier for this run
            fw1: Optional Framework 1 override
            fw2: Optional Framework 2 override
            fw3: Optional Framework 3 override
            fw4: Optional Framework 4 override
            
        Returns:
            Dictionary containing prediction results
        """
        # Track which frameworks were used
        used_frameworks = {
            'fw1': fw1 if fw1 is not None else self.framework_segments[0],
            'fw2': fw2 if fw2 is not None else self.framework_segments[1],
            'fw3': fw3 if fw3 is not None else self.framework_segments[2],
            'fw4': fw4 if fw4 is not None else self.framework_segments[3]
        }
        
        # Assemble sequence
        full_sequence = self.assemble_sequence(cdr1, cdr2, cdr3, fw1, fw2, fw3, fw4)
        
        # Create batch file in output directory
        batch_content = self.create_tango_batch_file(full_sequence, identifier)
        batch_file = os.path.join(self.output_dir, f"{identifier}_tango.sh")
        
        # Write and make executable
        with open(batch_file, 'w') as f:
            f.write(batch_content)
        os.chmod(batch_file, 0o755)  # Make executable
        
        #print(f"Created batch file {batch_file} with content:\n{batch_content}")
        
        # Run the batch file
        process = subprocess.Popen(
            ['bash', batch_file],
            cwd=self.output_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        #print(f"Batch execution stdout:\n{stdout}")
        if stderr:
            print(f"Batch execution stderr:\n{stderr}")
        
        if process.returncode != 0:
            raise RuntimeError(f"TANGO batch file execution failed with return code {process.returncode}")
        
        # Look for output file
        output_file = os.path.join(self.output_dir, f"{identifier}.txt")
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Expected output file {output_file} not found after TANGO execution")
        
        # Parse output with the actual frameworks used
        results = self.parse_tango_output(
            output_file, cdr1, cdr2, cdr3,
            fw1=used_frameworks['fw1'],
            fw2=used_frameworks['fw2'],
            fw3=used_frameworks['fw3'],
            fw4=used_frameworks['fw4']
        )
        
        # Add sequence information
        results['sequence'] = full_sequence
        results['frameworks_used'] = used_frameworks
        
        # Parse output
        results = self.parse_tango_output(output_file, cdr1, cdr2, cdr3)
        
        # Add sequence information
        results['sequence'] = full_sequence
        results['frameworks_used'] = used_frameworks
        
        return results