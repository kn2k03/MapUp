from typing import Dict, List
import pandas as pd
import re
from itertools import permutation
from geopy.distance import geodesic
import polyline
def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    result = []
    for i in range(0, len(lst), n):
        group = lst[i:i+n]
        for j in range(len(group) - 1, -1, -1):
            result.append(group[j])
    return result

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    result = {}
    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = []
        result[length].append(string)
    return dict(sorted(result.items()))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    flattened = {}

    def flatten(current_dict, parent_key=''):
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                flatten(value, new_key)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    flatten({f"{key}[{i}]": item}, new_key)
            else:
                flattened[new_key] = value

    flatten(nested_dict)
    return flattened

def unique_permutations(nums: List[int]) -> List[List[int]]:
    perm = permutations(nums)
    unique_perm = list(set(perm))
    return [list(p) for p in unique_perm]




def find_all_dates(text: str) -> List[str]:
    pattern = r'\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    dates = re.findall(pattern, text)
    return dates

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    decoded_coords = polyline.decode(polyline_str)
    df = pd.DataFrame(decoded_coords, columns=['latitude', 'longitude'])
    df['distance'] = 0.0
    for i in range(1, len(df)):
        coord1 = (df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude'])
        coord2 = (df.loc[i, 'latitude'], df.loc[i, 'longitude'])
        df.loc[i, 'distance'] = geodesic(coord1, coord2).meters
    return df

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    result = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            result[i][j] = row_sum + col_sum
    
    return result

def time_check(df: pd.DataFrame) -> pd.Series:
    df['startTime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['endTime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    grouped = df.groupby(['id', 'id_2'])
    
    def check_completeness(group):
        total_time = (group['endTime'] - group['startTime']).sum()
        return total_time == pd.Timedelta(days=7)

    return grouped.apply(check_completeness)


