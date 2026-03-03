"""
Utility functions for the Market Maven project.


"""
from datetime import date


def get_current_date():
    """
    Returns the current date in the format 'YYYY-MM-DD'.
    """
    return date.today().strftime('%Y-%m-%d')


def check_equal_occurrences(df, column_name):
    """
    Check if all values in the specified column occur the same number of times. Especially 
    useful when checking if all asssets have the same number of 
    observations in a dataframe.

    Args:
        df (pd.DataFrame): The dataframe to check.
        column_name (str): The column name to analyze.

    Returns:
        bool: True if all values occur the same number of times, False otherwise.
    """
    value_counts = df[column_name].value_counts()
    return value_counts.nunique() == 1
