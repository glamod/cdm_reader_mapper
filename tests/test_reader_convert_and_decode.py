from __future__ import annotations

import pandas as pd
import pytest

from decimal import Decimal

from cdm_reader_mapper.mdf_reader.utils.convert_and_decode import (
    max_decimal_places,
    to_numeric,
    Decoders,
    Converters,
    convert_and_decode,
)
from cdm_reader_mapper.mdf_reader import properties


@pytest.fixture
def sample_series():
    return pd.Series(["A", "Z", "10", "1Z"])


@pytest.fixture
def numeric_series():
    return pd.Series(["1", "2 ", "3", "False", "bad"], dtype="object", name="NUM")


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "NUM": ["1", "2 ", "3", "False", "bad"],  # object type
            "KEY": ["a", "b", "c", "d", "e"],  # for decoder
        }
    )


def test_max_decimal_places():
    assert max_decimal_places(Decimal("1"), Decimal("2.34")) == 2
    assert max_decimal_places(Decimal("1.200"), Decimal("3.4")) == 3
    assert max_decimal_places(Decimal("5")) == 0


@pytest.mark.parametrize(
    "value, scale, offset, expected",
    [
        ("10", Decimal("0.1"), Decimal("0"), Decimal("1.0")),
        ("10", Decimal("1"), Decimal("5"), Decimal("15")),
        ("3.5", Decimal("2"), Decimal("1.00"), Decimal("8.00")),
        (" 2 ", Decimal("1"), Decimal("0"), Decimal("2")),
        ("", Decimal("1"), Decimal("0"), False),
        ("abc", Decimal("1"), Decimal("0"), False),
    ],
)
def test_to_numeric_valid(value, scale, offset, expected):
    assert to_numeric(value, scale, offset) == expected


def test_to_numeric_boolean_passthrough():
    assert to_numeric(True, Decimal("1"), Decimal("0")) is True
    assert to_numeric(False, Decimal("1"), Decimal("0")) is False


def test_to_numeric_space_replacement():
    assert to_numeric("1 2", Decimal("1"), Decimal("0")) == Decimal("102")


def test_to_numeric_precision_preserved():
    result = to_numeric("1.234", Decimal("0.1"), Decimal("0.00"))
    assert result == Decimal("0.123")


def test_base36_decoding_basic(sample_series):
    dec = Decoders(dtype="key")
    decoder = dec.decoder()

    result = decoder(sample_series)

    assert list(result) == ["10", "35", "36", "71"]


def test_base36_preserves_boolean():
    series = pd.Series(["True", "False", "A"])
    dec = Decoders(dtype="key")

    result = dec.decoder()(series)

    assert result.tolist() == [True, False, "10"]


def test_converter_numeric(numeric_series):
    conv = Converters(dtype=next(iter(properties.numeric_types)))
    func = conv.converter()

    result = func(numeric_series)

    assert result.iloc[0] == Decimal("1")
    assert result.iloc[1] == Decimal("2")
    assert result.iloc[2] == Decimal("3")
    assert result.iloc[3] is False
    assert result.iloc[4] is False


def test_numeric_with_scale_offset():
    conv = Converters(dtype="float")
    series = pd.Series(["1", "2"])

    result = conv.object_to_numeric(series, scale=10, offset=5)

    assert result.tolist() == [Decimal("15"), Decimal("25")]


def test_preprocessing_function_pppp():
    conv = Converters(dtype=next(iter(properties.numeric_types)))
    series = pd.Series(["0123"], name="PPPP")

    result = conv.object_to_numeric(series)

    assert result.iloc[0] == Decimal("10123")


def test_object_to_object_strip():
    conv = Converters(dtype="object")
    series = pd.Series([" a ", "", "   ", "b"])

    result = conv.object_to_object(series)

    assert result.tolist() == ["a", None, None, "b"]


def test_object_to_object_disable_strip():
    conv = Converters(dtype="object")
    series = pd.Series([" a ", "b "])

    result = conv.object_to_object(series, disable_white_strip="l")

    assert result.tolist() == [" a", "b"]


def test_object_to_datetime():
    conv = Converters(dtype="datetime")
    series = pd.Series(["20240101", "bad"])

    result = conv.object_to_datetime(series)

    assert pd.notna(result.iloc[0])
    assert pd.isna(result.iloc[1])


def test_unknown_dtype_raises():
    with pytest.raises(KeyError):
        Converters("unknown").converter()


def test_convert_and_decode_basic():
    df = pd.DataFrame({"A": ["1", "2", "3"], "B": ["x", "y", "z"]})

    converter_dict = {
        "A": lambda s: s.apply(lambda x: Decimal(x) * 2),
        "B": lambda s: s.str.upper(),
    }
    converter_kwargs = {"A": {}, "B": {}}

    decoder_dict = {"A": lambda s: s.apply(lambda x: str(int(x) + 1))}

    out = convert_and_decode(
        df.copy(),
        convert_flag=True,
        decode_flag=True,
        converter_dict=converter_dict,
        converter_kwargs=converter_kwargs,
        decoder_dict=decoder_dict,
    )

    assert out["A"].iloc[0] == Decimal(4)
    assert out["A"].iloc[1] == Decimal(6)
    assert out["B"].iloc[0] == "X"


def test_convert_and_decode_with_converters_and_decoders(sample_df):
    df = sample_df.copy()

    conv = Converters(dtype="int")
    converter_dict = {"NUM": conv.converter()}
    converter_kwargs = {"NUM": {}}

    dec = Decoders(dtype="key")
    decoder_dict = {"KEY": dec.decoder()}

    out = convert_and_decode(
        df,
        convert_flag=True,
        decode_flag=True,
        converter_dict=converter_dict,
        converter_kwargs=converter_kwargs,
        decoder_dict=decoder_dict,
    )

    expected_nums = [Decimal("1"), Decimal("2"), Decimal("3"), False, False]
    for i, val in enumerate(expected_nums):
        assert out["NUM"].iloc[i] == val

    expected_keys = ["10", "11", "12", "13", "14"]
    for i, val in enumerate(expected_keys):
        assert out["KEY"].iloc[i] == val
