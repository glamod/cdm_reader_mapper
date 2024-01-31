r"""
Get data section.

Created on Tue Apr 30 09:38:17 2019

Splits string reports in sections using a data model layout.

Input and output are simple pandas dataframes, with the output dataframe
column names being the section names

To work with a pandas TextParser, loop through this module.

Internally works assuming highest complexity in the input data model:
multiple non sequential sections

DEV NOTES:

1) make sure we use Series when working with Series, DataFrames otherwise...
like now:

 threads[thread_id]['data'] = pd.Series(threads[thread_id]['parent_data'][0].str[0:section_len]) instead of:
 threads[thread_id]['data'] = pd.DataFrame(threads[thread_id]['parent_data'][0].str[0:section_len])

on data import in import_data.py, we use pd.read_fwf because is more general
use, also support to chunking would make converting to series a bit dirty...

2) Can we extend (do we need to?) this to reading sequential sections with
no sentinals? apparently (see td11) we are already able to do that:

 provided the section is in a sequential parsing_order group

@author: iregon

Have to documents the threads approach!!!!

"""

from __future__ import annotations

import logging
from copy import deepcopy

import pandas as pd


#   ---------------------------------------------------------------------------
#   FUNCTIONS TO PERFORM INITIAL SEPARATION OF SECTIONS: MAIN IS GET_SECTIONS()
#   ---------------------------------------------------------------------------
def extract_data():
    """DOCUMENTATION."""
    section_len = section_lens.get(threads[thread_id]["section"])
    if section_len:
        threads[thread_id]["data"] = pd.Series(
            threads[thread_id]["parent_data"][0].str[0:section_len]
        )  # object consistency needed here
        threads[thread_id]["modulo"] = pd.DataFrame(
            threads[thread_id]["parent_data"][0].str[section_len:]
        )  # object consistency needed here
    else:
        threads[thread_id]["data"] = pd.Series(
            threads[thread_id]["parent_data"][0].str[0:]
        )  # threads[thread_id]['parent_data'].copy()
        # Could even be like with section_len (None in section_len will read to the end)
        threads[thread_id]["modulo"] = pd.DataFrame(
            columns=[0], dtype=object
        )  # Just for consistency
    del threads[thread_id]["parent_data"]


def add_next_children():
    """DOCUMENTATION."""
    global children_parsing_order, branch_parsing_order, children_group_type, children_group_number
    children_parsing_order = deepcopy(threads[thread_id]["parsing_order"])
    branch_parsing_order = deepcopy(threads[thread_id]["parsing_order"])
    children_group_type = list(children_parsing_order[0])[0]
    children_group_number = threads[thread_id]["children_group_number"]
    threads[thread_id]["children_no"] = 0
    threads[thread_id]["children"] = []
    add_children()


def add_higher_group_children():
    """DOCUMENTATION."""
    global children_parsing_order, branch_parsing_order, children_group_type, children_group_number
    children_parsing_order = deepcopy(threads[thread_id]["parsing_order"])
    children_parsing_order.pop(0)  # Move to next group of sections
    if len(children_parsing_order) > 0:
        branch_parsing_order = deepcopy(threads[thread_id]["parsing_order"])
        branch_parsing_order.pop(0)
        children_group_type = list(children_parsing_order[0])[0]
        children_group_number = threads[thread_id]["children_group_number"] + 1
        add_children()


def add_children():
    """DOCUMENTATION."""
    if children_group_type == "s":
        add_static_children()
    else:
        add_dynamic_children()


def add_static_children():
    """DOCUMENTATION."""
    threads[thread_id]["children_no"] += 1
    children_thread_id = str(children_group_number) + str(0) + thread_id
    threads[thread_id]["children"].append(children_thread_id)
    # Now build children's thread
    children_section = children_parsing_order[0][children_group_type].pop(0)
    grandchildren_group_number = children_group_number
    if len(children_parsing_order[0][children_group_type]) == 0:
        children_parsing_order.pop(0)
        if len(children_parsing_order) > 0:
            grandchildren_group_number += 1
        else:
            grandchildren_group_number = None
    threads[children_thread_id] = {"parsing_order": children_parsing_order}
    threads[children_thread_id]["group_number"] = children_group_number
    threads[children_thread_id]["group_type"] = children_group_type
    threads[children_thread_id]["section"] = children_section
    threads[children_thread_id]["parent_data"] = threads[thread_id]["modulo"]
    threads[thread_id]["modulo"].iloc[0:0]  # Remove reports from modulo
    threads[children_thread_id]["children_group_number"] = grandchildren_group_number


def add_dynamic_children():
    """DOCUMENTATION."""
    for i in range(0, len(children_parsing_order[0][children_group_type])):
        branch_i_parsing_order = deepcopy(branch_parsing_order)
        children_thread_id = str(children_group_number) + str(i + 1) + thread_id
        # Now build children's thread
        children_section = children_parsing_order[0][children_group_type].pop(0)
        children_idx = (
            threads[thread_id]["modulo"]
            .loc[
                threads[thread_id]["modulo"][0].str[
                    0 : sentinals_lens.get(children_section)
                ]
                == sentinals.get(children_section)
            ]
            .index
        )
        if len(children_idx) == 0:
            continue
        threads[thread_id]["children"].append(children_thread_id)
        threads[thread_id]["children_no"] += 1
        branch_i_parsing_order[0][children_group_type].remove(children_section)
        grandchildren_group_number = children_group_number
        if (
            len(branch_i_parsing_order[0][children_group_type]) == 0
            or children_group_type == "e"
        ):
            branch_i_parsing_order.pop(0)
            if len(children_parsing_order) > 0:
                grandchildren_group_number += 1
            else:
                grandchildren_group_number = None
        threads[children_thread_id] = {"parsing_order": branch_i_parsing_order}
        threads[children_thread_id]["group_number"] = children_group_number
        threads[children_thread_id]["group_type"] = children_group_type
        threads[children_thread_id]["section"] = children_section
        threads[children_thread_id]["parent_data"] = threads[thread_id]["modulo"].loc[
            children_idx
        ]
        threads[thread_id]["modulo"].drop(children_idx, inplace=True)
        threads[children_thread_id][
            "children_group_number"
        ] = grandchildren_group_number
    if (len(threads[thread_id]["modulo"])) > 0:
        add_higher_group_children()


def extract_sections(string_df):
    """DOCUMENTATION."""
    # threads elements:
    #    'parsing_order'            What needs to be applied to current parent data
    #    'group_number'             Order in the global parsing order
    #    'group_type'               Is it sequential, exclusive or optional
    #    'section'                  Section name to be extracted from parent_data to data
    #    'parent_data'              Inital data from which section must be extracted
    #    'data'                     Section data extracted from parent_data
    #    'modulo'                   Reminder of parent_data after extracting section (data)
    #    'children_no'              Number of children threads to build, based on next parsing order list element. Resets to number of active children
    #    'children'                 Thread id for every child
    #    'children_group_number'    Group number (in the global parsing order, of the children)
    global sentinals, section_lens, sentinal_lens, parsing_order
    global children_group_type
    global threads
    global thread_id
    global group_type

    # Initial "node': input data
    threads = dict()
    thread_id = "00"
    threads_queue = [thread_id]
    threads[thread_id] = {"parsing_order": parsing_order}
    threads[thread_id]["group_number"] = 0
    threads[thread_id]["group_type"] = None
    threads[thread_id]["section"] = None
    threads[thread_id]["parent_data"] = string_df
    threads[thread_id]["data"] = None
    threads[thread_id]["modulo"] = threads[thread_id]["parent_data"]
    del threads[thread_id]["parent_data"]
    threads[thread_id]["children_group_number"] = 1
    add_next_children()
    threads_queue.extend(threads[thread_id]["children"])
    threads_queue.remove(thread_id)
    # And now, once initialized, let it grow:
    logging.info("Processing section partitioning threads")
    while threads_queue:
        thread_id = threads_queue[0]
        logging.info(f"{thread_id} ...")
        group_type = threads[thread_id]["group_type"]
        # get section data
        extract_data()
        # kill thread if nothing there
        if len(threads[thread_id]["data"]) == 0:
            del threads[thread_id]
            logging.info(f"{thread_id} deleted: no data")
            threads_queue.pop(0)
            continue
        # build children threads
        if (
            len(threads[thread_id]["parsing_order"]) > 0
            and len(threads[thread_id]["modulo"]) > 0
        ):
            add_next_children()
            threads_queue.extend(threads[thread_id]["children"])
            # del threads[thread_id]['modulo'] # not until we control what to do whit leftovers....
        threads_queue.pop(0)
        logging.info("done")
    section_dict = dict()
    section_groups = [d[x] for d in parsing_order for x in d.keys()]
    sections = [item for sublist in section_groups for item in sublist]

    for section in sections:
        section_dict[section] = pd.Series()
        thread_ids = [x for x in threads.keys() if threads[x]["section"] == section]
        for thread_id in thread_ids:
            section_dict[section] = pd.concat(
                [section_dict[section], threads[thread_id]["data"]], ignore_index=False
            )
        section_dict[section].sort_index(inplace=True)
    return section_dict


#   ---------------------------------------------------------------------------
#   MAIN
#   ---------------------------------------------------------------------------
def get_sections(string_df, schema, read_sections):
    """Get sections from pd.DataFrame.

    Returns a pandas dataframe with a report per row
    and the report sections split along the columns.
    Each section is a block string and only the sections
    listed in read_sections parameter are output.

    Parameters
    ----------
    string_df : pd.DataFrame
        Pandas dataframe with a unique column with
        the reports as a block string

    schema : dict
        Data source data model schema

    read_sections : list
        Sections to output from the complete report


    Returns
    -------
    pd.DataFrame
        Dataframe with the report sections split
        along the columns.


    """
    global sentinals, section_lens, sentinals_lens
    global parsing_order
    # Proceed to split sections if more than one
    # else return section in a named column
    if len(schema["sections"].keys()) > 1:
        section_lens = {
            section: schema["sections"][section]["header"].get("length")
            for section in schema["sections"].keys()
        }
        sentinals = {
            section: schema["sections"][section]["header"].get("sentinal")
            for section in schema["sections"].keys()
        }
        sentinals_lens = {
            section: len(sentinals.get(section)) if sentinals.get(section) else 0
            for section in sentinals.keys()
        }
        parsing_order = schema["header"]["parsing_order"]
        # Get sections separated: section dict has a key:value pair for each
        # section in the data model. If the section does not exist in the data,
        # the value is an empty pd.Series
        section_dict = extract_sections(string_df)
        # Paste in order (as read_sections) in a single dataframe with columns
        # named as sections:
        # - Drop unwanted sections
        # - Keep requested but non-existent sections
        df_out = pd.DataFrame()
        for section in read_sections:
            df_out = pd.concat(
                [df_out, section_dict[section].rename(section)], sort=False, axis=1
            )
    else:
        df_out = string_df
        df_out.columns = read_sections

    return df_out
