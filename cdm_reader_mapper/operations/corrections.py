"""
Common Data Model (CDM) pandas correction operators.

Created on Tue Jun 21 15:32:29 2022

@author: sbiri
"""
from __future__ import annotations

import logging
import math
import os

import numpy as np
import pandas as pd
from textdistance import levenshtein


# %% extract NOC_corrections/duplicates
def gen_files(data, dataset, correction_path, yr, mo):
    """DOCUMENTATION."""
    prepend = (
        dataset.split("_")[0] + "-" + dataset.split("_")[1][1:6].replace(".", "") + "-"
    )
    for f in [
        "duplicates",
        "duplicate_flags",
        "id",
        "longitude",
        "latitude",
        "timestamp",
    ]:
        os.makedirs(os.path.join(correction_path, f), exist_ok=True)
    df = pd.DataFrame(
        {
            "UID": data[("c98", "UID")],
            "ID": data[("core", "ID")],
            "LON": data[("core", "LON")],
            "LAT": data[("core", "LAT")],
        }
    )
    df["UID"] = df["UID"].apply(lambda x: f"{prepend+x}")
    hours = np.round(data[("core", "HR")], decimals=0).astype("Int64")
    minutes = np.round(60 * np.fmod(data[("core", "HR")], 1)).astype("Int64")
    df["TM"] = pd.to_datetime(
        pd.DataFrame(
            {
                "YR": int(yr) * np.ones(hours.shape, dtype=int),
                "MO": int(mo) * np.ones(hours.shape, dtype=int),
                "DY": data[("core", "DY")].astype("Int64"),
                "H": hours,
                "M": minutes,
            }
        )
        .astype(str)
        .apply("-".join, axis=1)
        .values,
        format="%Y-%m-%d-%H-%M",
        errors="coerce",
    )
    df["TM"] = df["TM"].apply(lambda x: f"{x}+00:00")
    df["flag"] = 0
    # df['UID'] = df['UID'].apply(lambda x: f"{prepend+x}")
    # %% duplicates
    fn = os.path.join(correction_path, "duplicates", yr + "-" + mo + ".txt.gz")
    fn_f = os.path.join(correction_path, "duplicate_flags", yr + "-" + mo + ".txt.gz")
    if not df.empty:
        dup, dup_f = get_dup(data, dataset)
        if os.path.exists(fn):
            df1 = pd.read_csv(
                fn,
                delimiter="|",
                dtype="object",
                header=None,
                usecols=[0, 1, 2],
                names=["UID", "UID_d", "flag"],
                quotechar=None,
                quoting=3,
            )
            os.remove(fn)
            pd.concat([df1.astype(str), dup.astype(str)]).drop_duplicates(
                subset="UID", keep="last"
            ).to_csv(fn, sep="|", header=False, index=False, compression="infer")
        else:
            dup.astype(str).to_csv(
                fn, sep="|", header=False, index=False, compression="infer"
            )
        if os.path.exists(fn_f):
            df1 = pd.read_csv(
                fn_f,
                delimiter="|",
                dtype="object",
                header=None,
                usecols=[0, 1, 2],
                names=["UID", "dup_flag", "flag"],
                quotechar=None,
                quoting=3,
            )
            os.remove(fn_f)
            pd.concat([df1.astype(str), dup_f.astype(str)]).drop_duplicates(
                subset="UID", keep="last"
            ).to_csv(fn_f, sep="|", header=False, index=False, compression="infer")
        else:
            dup_f.astype(str).to_csv(
                fn_f, sep="|", header=False, index=False, compression="infer"
            )
    # %% id
    fn = os.path.join(correction_path, "id", yr + "-" + mo + ".txt.gz")
    if not df.empty:
        # print("ID not empty")
        if os.path.exists(fn):
            df1 = pd.read_csv(
                fn,
                delimiter="|",
                dtype="object",
                header=None,
                usecols=[0, 1, 2],
                names=["UID", "ID", "flag"],
                quotechar=None,
                quoting=3,
            )
            os.remove(fn)
            pd.concat(
                [df1.astype(str), df[["UID", "ID", "flag"]].astype(str)]
            ).drop_duplicates(subset="UID", keep="last").to_csv(
                fn, sep="|", header=False, index=False, compression="infer"
            )
        else:
            df[["UID", "ID", "flag"]].astype(str).to_csv(
                fn, sep="|", header=False, index=False, compression="infer"
            )
    # %% longitude
    fn = os.path.join(correction_path, "longitude", yr + "-" + mo + ".txt.gz")
    if not df.empty:
        if os.path.exists(fn):
            df1 = pd.read_csv(
                fn,
                delimiter="|",
                dtype="object",
                header=None,
                usecols=[0, 1, 2],
                names=["UID", "LON", "flag"],
                quotechar=None,
                quoting=3,
            )
            os.remove(fn)
            pd.concat(
                [df1.astype(str), df[["UID", "LON", "flag"]].astype(str)]
            ).drop_duplicates(subset="UID", keep="last").to_csv(
                fn, sep="|", header=False, index=False, compression="infer"
            )
        else:
            df[["UID", "LON", "flag"]].astype(str).to_csv(
                fn, sep="|", header=False, index=False, compression="infer"
            )
    # latitude
    fn = os.path.join(correction_path, "latitude", yr + "-" + mo + ".txt.gz")
    if not df.empty:
        if os.path.exists(fn):
            df1 = pd.read_csv(
                fn,
                delimiter="|",
                dtype="object",
                header=None,
                usecols=[0, 1, 2],
                names=["UID", "LAT", "flag"],
                quotechar=None,
                quoting=3,
            )
            os.remove(fn)
            pd.concat(
                [df1.astype(str), df[["UID", "LAT", "flag"]].astype(str)]
            ).drop_duplicates(subset="UID", keep="last").to_csv(
                fn, sep="|", header=False, index=False, compression="infer"
            )
        else:
            df[["UID", "LAT", "flag"]].astype(str).to_csv(
                fn, sep="|", header=False, index=False, compression="infer"
            )
    # timestamp
    fn = os.path.join(correction_path, "timestamp", yr + "-" + mo + ".txt.gz")
    if not df.empty:
        if os.path.exists(fn):
            df1 = pd.read_csv(
                fn,
                delimiter="|",
                dtype="object",
                header=None,
                usecols=[0, 1, 2],
                names=["UID", "TM", "flag"],
                quotechar=None,
                quoting=3,
            )
            os.remove(fn)
            pd.concat(
                [df1.astype(str), df[["UID", "TM", "flag"]].astype(str)]
            ).drop_duplicates(subset="UID", keep="last").to_csv(
                fn, sep="|", header=False, index=False, compression="infer"
            )
        else:
            df[["UID", "TM", "flag"]].astype(str).to_csv(
                fn, sep="|", header=False, index=False, compression="infer"
            )


def corrections(data, dataset, correction_path, yr, mo):
    """DOCUMENTATION."""
    logging.basicConfig(
        format="%(levelname)s\t[%(asctime)s](%(filename)s)\t%(message)s",
        level=logging.INFO,
        datefmt="%Y%m%d %H:%M:%S",
        filename=None,
    )
    for f in [
        "duplicates",
        "duplicate_flags",
        "id",
        "longitude",
        "latitude",
        "timestamp",
    ]:
        os.makedirs(os.path.join(correction_path, f), exist_ok=True)

    gen_files(data.copy(), dataset, correction_path, yr, mo)


def split_list(n):
    """DOCUMENTATION."""
    return [(x + 1) for x, y in zip(n, n[1:]) if y - x != 1]


def convert_longitude(lon):
    """Convert longitude to -180 to 180."""
    if lon > 180:
        return -180 + math.fmod(lon, 180)
    return lon


def get_dup(data, dataset):
    """
    Check for duplicates.

    In a subset of dataframe that contains UID, ID, LON, LAT, DY, HR
    checks for duplicates with tolerances for 2 digits in strings
    and depending on variable for numeric variables/columns given in tol


    Parameters
    ----------
    data : pd.dataframe
        DESCRIPTION.
    dataset : str
        DESCRIPTION.
    tol : pd.series
        DESCRIPTION

    Returns
    -------
    None.

    """
    prepend = (
        dataset.split("_")[0] + "-" + dataset.split("_")[1][1:6].replace(".", "") + "-"
    )
    df = pd.DataFrame(
        {
            "UID": data[("c98", "UID")],
            "ID": data[("core", "ID")],
            "LON": data[("core", "LON")],
            "LAT": data[("core", "LAT")],
            "DY": data[("core", "DY")],
            "HR": data[("core", "HR")],
        }
    )
    df["UID"] = df["UID"].apply(lambda x: f"{prepend+x}")
    # round lon, lat to one digit
    df[["LON", "LAT"]] = df[["LON", "LAT"]].astype(float).round(1)
    # convert longitdute to -180-180
    df["LON"] = df["LON"].apply(convert_longitude)
    tol = pd.Series([2, 0, 0.05, 0.05, 0, 0])
    tol.index = ["UID", "ID", "LON", "LAT", "DY", "HR"]
    df_dup = df.copy()
    df_dup["flag"] = 0
    # first flag pos & time
    df_dup = df_dup.sort_values(by=["LON", "LAT", "DY", "HR"])
    tmp_id = pd.DataFrame()
    tmp_id["ID"] = df_dup["ID"].copy()
    tmp_id["ID_s"] = df_dup["ID"].shift().astype(str)
    tmp_id = tmp_id.assign(
        distance=[*map(levenshtein.distance, tmp_id.ID, tmp_id.ID_s)]
    )
    tmp_uid = pd.DataFrame()
    tmp_uid["UID"] = df_dup["UID"].copy()
    tmp_uid["UID_s"] = df_dup["UID"].shift().astype(str)
    tmp_uid = tmp_uid.assign(
        distance=[*map(levenshtein.distance, tmp_uid.UID, tmp_uid.UID_s)]
    )
    loc = (
        (abs(df_dup["LON"] - df_dup["LON"].shift()) <= tol["LON"])
        & (abs(df_dup["LAT"] - df_dup["LAT"].shift()) <= tol["LAT"])
        & (abs(df_dup["DY"] - df_dup["DY"].shift()) <= tol["DY"])
        & (abs(df_dup["HR"] - df_dup["HR"].shift()) <= tol["HR"])
        & (tmp_id["distance"] <= tol["ID"])
        & (tmp_uid["distance"] <= tol["UID"])
    )
    df_dup["flag"] = df_dup["flag"].where(~loc, 1)
    dup_flag = pd.DataFrame({"UID": df["UID"].copy(), "dup_flag": 0, "flag": 1})
    dup_flag["dup_flag"] = dup_flag["dup_flag"].where(~loc, 1)
    #   %%
    dup_list = (
        df_dup.sort_values(by=["LON", "LAT", "DY", "HR"])
        .loc[df_dup["flag"] == 1]
        .index.to_list()
    )
    #  %% find consecutive indices in list of duplicates
    lst = dup_list.copy()
    lst.sort()
    ind = split_list(lst)
    pv = 0
    dup_out = pd.DataFrame()
    # find consecutive indices
    for index in ind:
        nlst = [x for x in lst[pv:] if x < index]
        pv += len(nlst)
        nlst.insert(0, nlst[0] - 1)
        # choose the first duplicate to keep #!THiS SHOULD IMPROVE
        dup_flag["dup_flag"][nlst[0]] = 3
        # generate all combinations of consecutive values and add to df
        for fe in nlst:
            r_nlst = list(nlst)
            r_nlst.remove(fe)
            tmp = "{" + ",".join(df["UID"].loc[r_nlst]) + "}"
            dup_out = dup_out.append(
                {"UID": df["UID"].loc[fe], "UID_d": tmp, "flag": 1},
                ignore_index=True,
            )
    return dup_out, dup_flag
