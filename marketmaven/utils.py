"""
Utility functions for the Market Maven project.


"""
from datetime import date

def get_current_date():
    """
    Returns the current date in the format 'YYYY-MM-DD'.
    """
    return date.today().strftime('%Y-%m-%d')