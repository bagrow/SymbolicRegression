import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import copy

def is_number(x):
    """Check if x is a number.

    Parameters
    ----------
    x : str
        A string that may or may not
        be a number.

    Returns
    -------
    bool
        True if number. Else False.
    """

    try:
        float(x)   # this will fail if not a number
        return True

    except ValueError:
        return False


def remove_duplicates(x1, x2):
    """Take two equal length lists where x1 has many duplicate values
    find the first instance of a duplicate and the last. Remove the
    intermediate values from both lists.

    Parameters
    ----------
    x1, x2 : lists
        x1 is the one with duplicates to remove

    Returns
    -------
    x1, x2 : lists
        Subsets of the original x1 and x2.
    """

    assert len(x1) == len(x2), 'x1 and x2 must be the same length'

    # This list of indices will be the elements of x1 and x2
    # to be kept.
    indices = []

    # prev is the previous value in the list
    prev = None

    # duplicates is true if the loop is currently
    # in a sequence of duplicate values.
    duplicates = False

    for i, x in enumerate(x1):

        # if first element of x1
        if prev is None:

            # keep this element
            indices.append(i)

        else:

            # if we find a duplicate
            if x == prev:
                duplicates = True
                continue

            else:

                # if sequence of duplicates has ended
                if duplicates:

                    # keep this element
                    indices.append(i-1)
                    duplicates = False

                # keep this element
                indices.append(i)

        # update previous element
        prev = x

    indices = np.array(indices)

    return x1[indices], x2[indices]


def write_table_with_bold_rows(df, filename, bold_rows):
    """Write a table to a .xls file with rows bolded.

    Parameters
    ----------
    df : pd.DataFrame
        The table to write.
    filename : str
        The location and name of file to
        be saved.
    bold_rows : list
        The rows to be bolded.
    """

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(filename+'.xls', engine='xlsxwriter')

    # Get the xlsxwriter workbook and worksheet objects.
    workbook  = writer.book
    worksheet = workbook.add_worksheet('Sheet1')

    # Add a header format.
    bold_format = workbook.add_format({'bold': True})

    # Write the column names.
    for k, val in enumerate(df.columns.values):
        worksheet.write(0, k, val)
    
    # Write the rest of the table.
    for i, row in enumerate(df.values):
        for j, val in enumerate(row):
            
            # Bold value if in the correct row.
            if i in bold_rows:
                worksheet.write(i+1, j, val, bold_format)
            
            else:
                worksheet.write(i+1, j, val)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def find_nearest(value, data):
    """Find index so that data[index] is
    closer to value than anther other.

    Parameters
    ----------
    value : float
        The value we want to find 
        an index for.
    data : 1D list
        The data to find the index
        that has the closest value.

    Returns
    -------
    index : int
        The index of data such that
        data[index] is closest to
        value.
    """

    adata = np.array(data)
    diff = np.abs(data-value)
    index = np.argmin(diff)

    return index


def organize_data(x_data, y_data):
    """Find shortests row in x_data. Get
    indices for other rows that have closest
    values.

    Parameters
    ----------
    x_data : 2D list
        Each row may have a different length. This
        data will be used to decide which indices to
        keep
    y_data : 2D list
        Each row may have a different length. The
        number of rows is expected to be the same
        as for x_data.

    Returns
    -------
    2D np.array
        Reduced versions of x_data and y_data
    """

    assert len(x_data) == len(y_data), 'x_data and y_data have different number of rows '+len(x_data)+' '+len(y_data)

    assert [len(row) for row in x_data] == [len(row) for row in y_data], 'x_data and y_data have different number of columns '+str([len(row) for row in x_data]) + ' ' + str([len(row) for row in y_data])

    # find shortest row in x_data
    min_length_index = np.argmin([len(x) for x in x_data])

    _x_data = copy.copy(x_data)
    _y_data = copy.copy(y_data)

    # Go through the shortest row and find the
    # nearest value (in each other row) to the
    # elements of the shortest row.
    for i, row in enumerate(_x_data):

        print('.', end='')

        # Don't change anything about
        # the shortest row.
        if i == min_length_index:
            continue

        # Build a new row.
        new_row_x = []
        new_row_y = []

        # For each element of the shortest row
        for value in x_data[min_length_index]:

            # Ge the index of the value that is closest
            index = find_nearest(value, row)

            # Put this value in the new row.
            # Use the same index for y_data
            new_row_x.append(row[index])
            new_row_y.append(y_data[i][index])

        # Apply the new row
        _x_data[i] = new_row_x
        _y_data[i] = new_row_y

    return np.array(_x_data), np.array(_y_data)
