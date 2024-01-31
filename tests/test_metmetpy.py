from __future__ import annotations

import pytest  # noqa

from cdm_reader_mapper.metmetpy import (
    correct_datetime,
    correct_pt,
    validate_datetime,
    validate_id,
)

from ._data import data_df, data_pa

dataset = "icoads_r3000_d714"
deck = "992"


def test_correct_datetime_pandas():
    correct_datetime.correct(
        data=data_df,
        data_model="imma1",
        deck=deck,
    )


def test_correct_datetime_parser():
    correct_datetime.correct(
        data=data_pa,
        data_model="imma1",
        deck=deck,
    )


def test_validate_datetime_pandas():
    validate_datetime.validate(
        data=data_df,
        data_model="imma1",
        dck=deck,
    )


def test_validate_datetime_parser():
    validate_datetime.validate(
        data=data_pa,
        data_model="imma1",
        dck=deck,
    )


def test_correct_pt_pandas():
    correct_pt.correct(
        data_df,
        dataset="icoads_r3000",
        data_model="imma1",
        deck=deck,
    )


def test_correct_pt_parser():
    correct_pt.correct(
        data_pa,
        dataset="icoads_r3000",
        data_model="imma1",
        deck=deck,
    )


def test_validate_id_pandas():
    validate_id.validate(
        data=data_df,
        dataset="icoads_r3000",
        data_model="imma1",
        dck=deck,
    )


def test_validate_id_parser():
    validate_id.validate(
        data=data_pa,
        dataset="icoads_r3000",
        data_model="imma1",
        dck=deck,
    )
