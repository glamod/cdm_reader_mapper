"""
metmetpy platfrom_type icoads_r3000_NRT library.

Created on Tue Jun 25 09:07:05 2019

@author: iregon
"""

from __future__ import annotations

import re

from .. import properties


def isNum(X):
    """DOCUMENTATION."""
    try:
        a = X.isnumeric()  # & len(re.sub("[0-9]", "", X)) == 0
        # re.sub("(,[ ]*!.*)$", "", X)
        # [re.sub("(,[ ]*!.*)$", "", x) for x in strings] # for a list of strings
    except Exception:
        a = False
    return a


def deck_792_imma1(data):
    """DOCUMENTATION."""
    sid = "103"
    pt = "5"
    buoys = "6"
    regex = re.compile("^[0-9]+$")  # is numeric
    id_col = properties.metadata_datamodels.get("id").get("imma1")
    sid_col = properties.metadata_datamodels.get("source").get("imma1")
    pt_col = properties.metadata_datamodels.get("platform").get("imma1")

    loc = (
        (data[id_col].str.match(regex))
        & (data[id_col].apply(len) != 7)
        & (data[id_col].apply(len) != 5)
        & (data[id_col].str.startswith("7") is False)
        & (data[sid_col] == sid)
        & (data[pt_col] == pt)
    )

    try:
        data.loc[data[pt_col][loc], pt_col] = buoys
    except KeyError:
        pass

    # data[pt_col].loc[loc] = buoys
    # (data[id_col].apply(len) != 7)  # length not eq. to 7
    # (data[id_col].apply(len) != 5)  # length not eq. to 5
    # (data[id_col].str.startswith("7") == False) # doesn't start with 7
    # is.buoy <- ifelse(!grepl("[A-Z,a-z]",df$id) & df$yr>=1980 & !(df$dck %in% numeric.ships) & grepl("[0-9]",df$id) & df$newpt==5, TRUE, FALSE)
    # is.buoy <- ifelse(df$dck %in% c(792,992) & is.buoy & nchar(df$id) == 7, FALSE, is.buoy)
    # is.buoy <- ifelse(df$dck %in% c(700,792,992) & is.buoy & substr(df$id,1,1) == "7" & nchar(df$id) == 5, FALSE, is.buoy)
    # df$newpt<-ifelse(is.buoy, 6, df$newpt)
    return data


def deck_992_imma1(data):
    """DOCUMENTATION."""
    sid = "114"
    pt = "5"
    lv = "4"  # light vessels
    buoys = "6"
    regex = re.compile("^6202+$")
    id_col = properties.metadata_datamodels.get("id").get("imma1")
    sid_col = properties.metadata_datamodels.get("source").get("imma1")
    pt_col = properties.metadata_datamodels.get("platform").get("imma1")

    # loc = (data[id_col].str.match(regex)) & (data[id_col].apply(ilen) == 7) & \
    #     (data[sid_col] == sid) & (data[pt_col] == pt)
    loc = (
        (data[id_col].str.match(regex))
        & (data[id_col].str.len() == 7)
        & (data[sid_col] == sid)
        & (data[pt_col] == pt)
    )
    # ifelse(df$dck==992 & substr(df$id, 1, 4) == "6202" & nchar(df$id) == 7, 4, df$newpt)

    try:
        data.loc[data[pt_col][loc], pt_col] = lv
    except KeyError:
        pass

    # light vessels in dck 992
    regex = re.compile("^[0-9]+$")  # is numeric
    # loc = (data[id_col].str.match(regex)) & (data[id_col].apply(ilen) != 7) & \
    #     (data[id_col].apply(len) != 5) & \
    #         (data[id_col].str.startswith("7") == False) & \
    #             (data[sid_col] == sid ) & (data[pt_col] == pt)
    loc = (
        (data[id_col].str.match(regex))
        & (data[id_col].str.len() != 7)
        & (data[id_col].str.len() != 5)
        & (data[id_col].str.startswith("7") is False)
        & (data[sid_col] == sid)
        & (data[pt_col] == pt)
    )
    try:
        data.loc[data[pt_col][loc], pt_col] = buoys
    except KeyError:
        pass
    return data
